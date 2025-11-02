"""
generate_articles_faster_whisper_v2.py

This script extends ``generate_articles_faster_whisper.py`` by adding a
robust fallback mechanism for fetching YouTube transcripts. On some
systems, the ``youtube-transcript-api`` package may not provide the
``list_transcripts`` method (for example, when an outdated version is
installed). To support these environments, the helper
``try_youtube_transcript`` first checks for the presence of
``list_transcripts`` and, if absent, falls back to using
``YouTubeTranscriptApi.get_transcript``. The rest of the functionality
remains the same: the script retrieves the latest videos from a given
YouTube channel, attempts to extract or transcribe audio, summarises
the text, and saves it as Markdown.

Usage is identical to the previous script. See ``config.json`` for
configuration details. Ensure that the ``YOUTUBE_API_KEY`` environment
variable is set to a valid YouTube Data API key before running.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from pytube import YouTube
from faster_whisper import WhisperModel

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
import nltk

# Language for tokenisation and stopwords (Sumy)
LANGUAGE = "english"


def ensure_nltk_data() -> None:
    """Ensure required NLTK data is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.corpus.stopwords.words(LANGUAGE)
    except LookupError:
        nltk.download("stopwords")


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_youtube_client(api_key: str):
    """Create a YouTube API client."""
    return build("youtube", "v3", developerKey=api_key)


def resolve_channel_id(youtube, identifier: str) -> str:
    """Resolve a channel handle, name, or ID to a channel ID."""
    identifier = identifier.strip()
    if identifier.startswith("UC"):
        return identifier
    search_response = (
        youtube.search()
        .list(
            q=identifier,
            part="id,snippet",
            type="channel",
            maxResults=1,
        )
        .execute()
    )
    items = search_response.get("items", [])
    if not items:
        raise ValueError(f"Could not resolve channel identifier '{identifier}'")
    return items[0]["id"]["channelId"]


def fetch_latest_videos(
    youtube, channel_id: str, max_results: int = 5
) -> List[Dict[str, Any]]:
    """Fetch the latest videos for a channel."""
    search_response = (
        youtube.search()
        .list(
            channelId=channel_id,
            part="id,snippet",
            order="date",
            maxResults=max_results,
            type="video",
        )
        .execute()
    )
    return search_response.get("items", [])


def try_youtube_transcript(
    video_id: str, languages: Optional[List[str]] = None
) -> Optional[str]:
    """
    Attempt to retrieve a video's transcript. If the installed version of
    youtube-transcript-api supports ``list_transcripts``, it will be
    used to find preferred languages; otherwise, this function falls back
    to ``get_transcript``.

    :param video_id: The YouTube video ID.
    :param languages: A list of language codes to prioritise (e.g., ["ja", "en"]).
    :returns: The transcript text, or None if unavailable.
    """
    if languages is None:
        languages = ["ja", "en"]
    # Newer versions expose list_transcripts
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            transcript_list = None
        if transcript_list is not None:
            # Try preferred languages first
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    segments = transcript.fetch()
                    return " ".join(segment["text"] for segment in segments)
                except NoTranscriptFound:
                    continue
            # Fallback to any available transcript
            try:
                transcript = transcript_list.find_transcript([
                    t.language_code for t in transcript_list
                ])
                segments = transcript.fetch()
                return " ".join(segment["text"] for segment in segments)
            except NoTranscriptFound:
                pass
    # Fallback path: try get_transcript directly for each language
    for lang in languages:
        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            return " ".join(segment["text"] for segment in segments)
        except Exception:
            continue
    # Try without specifying languages
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(segment["text"] for segment in segments)
    except Exception:
        return None


def download_audio(video_id: str, temp_dir: str) -> str:
    """Download only the audio of a YouTube video."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    audio_stream = (
        yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    )
    if audio_stream is None:
        raise RuntimeError(f"No audio stream found for video {video_id}")
    filename = f"{video_id}.mp4"
    return audio_stream.download(output_path=temp_dir, filename=filename)


def transcribe_audio(file_path: str, model: WhisperModel) -> str:
    """Transcribe an audio file using faster-whisper."""
    segments, _ = model.transcribe(file_path, beam_size=5)
    return " ".join([segment.text.strip() for segment in segments])


def summarise_text(text: str, sentences: int = 5) -> str:
    """Summarise a text using Sumy (LSA with Luhn fallback)."""
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summariser = LsaSummarizer(stemmer)
    summariser.stop_words = nltk.corpus.stopwords.words(LANGUAGE)
    summary = summariser(parser.document, sentences)
    if not summary:
        summariser = LuhnSummarizer(stemmer)
        summariser.stop_words = nltk.corpus.stopwords.words(LANGUAGE)
        summary = summariser(parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary)


def write_markdown(
    video: Dict[str, Any],
    summary: str,
    output_dir: str,
    transcript_source: str,
) -> None:
    """Write the summary to a Markdown file."""
    vid = video["id"]["videoId"]
    snippet = video["snippet"]
    title = snippet.get("title", "").strip()
    channel_title = snippet.get("channelTitle", "").strip()
    published_at = snippet.get("publishedAt", "").strip()
    url = f"https://www.youtube.com/watch?v={vid}"
    date_str = published_at
    try:
        date_obj = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{vid}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- **チャンネル**: {channel_title}\n")
        f.write(f"- **公開日**: {date_str}\n")
        f.write(f"- **動画URL**: {url}\n")
        f.write(f"- **文字起こし元**: {transcript_source}\n\n")
        f.write("## 要約\n\n")
        f.write(summary)


def main() -> None:
    """Entry point."""
    ensure_nltk_data()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY environment variable must be set")
    config = load_config()
    identifier = config.get("channel_identifier")
    if not identifier:
        raise RuntimeError("'channel_identifier' must be provided in config.json")
    max_results = int(config.get("max_results", 5))
    posts_dir = config.get("posts_dir", "posts")
    model_size = config.get("whisper_model", "small")
    print(f"Loading faster-whisper model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    youtube = get_youtube_client(api_key)
    channel_id = resolve_channel_id(youtube, identifier)
    print(f"Resolved channel '{identifier}' to ID '{channel_id}'")
    videos = fetch_latest_videos(youtube, channel_id, max_results=max_results)
    if not videos:
        print("No videos found.")
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        for video in videos:
            vid = video["id"]["videoId"]
            print(f"Processing {vid}...")
            transcript = try_youtube_transcript(vid)
            transcript_source = "YouTube captions"
            if not transcript:
                print("No captions available. Transcribing audio with faster-whisper...")
                try:
                    audio_path = download_audio(vid, temp_dir)
                    transcript = transcribe_audio(audio_path, model)
                    transcript_source = "faster-whisper transcription"
                except Exception as e:
                    print(f"Transcription failed for {vid}: {e}")
                    continue
            if not transcript:
                print(f"No transcript available for {vid}, skipping...")
                continue
            summary = summarise_text(transcript)
            write_markdown(video, summary, posts_dir, transcript_source)
            print(f"Generated summary for {vid} -> {posts_dir}/{vid}.md")


if __name__ == "__main__":
    main()