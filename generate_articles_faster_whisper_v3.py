"""
generate_articles_faster_whisper_v3.py

This script builds upon ``generate_articles_faster_whisper_v2.py`` by
adding a robust fallback for downloading audio tracks when ``pytube``
fails. Many users experience HTTP 400 errors when using ``pytube`` to
download streams due to upstream changes on YouTube. To mitigate this,
``download_audio`` first tries ``pytube``; if it raises an exception or
no audio stream is found, the script falls back to using the
well-maintained ``yt-dlp`` library to download the best available
audio track. This makes the script more resilient to changes in
YouTube's internal APIs.

Other functionality remains the same: the script retrieves the latest
videos from a specified YouTube channel, attempts to extract
transcripts via ``youtube-transcript-api`` with a fallback to
``get_transcript`` when necessary, performs automatic speech
recognition using ``faster-whisper`` for videos without captions,
summarises the resulting text with Sumy, and saves the summary as a
Markdown file.

Before running, ensure you have installed the following dependencies:

  - google-api-python-client
  - youtube-transcript-api
  - pytube
  - faster-whisper
  - yt-dlp
  - sumy
  - nltk

and set the environment variable ``YOUTUBE_API_KEY`` to a valid YouTube
Data API key. The configuration (channel handle, number of videos to
process, output directory, Whisper model size) is read from
``config.json``.
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
from yt_dlp import YoutubeDL

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
import nltk

# Language for Sumy
LANGUAGE = "english"


def ensure_nltk_data() -> None:
    """Download NLTK data files if they are not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.corpus.stopwords.words(LANGUAGE)
    except LookupError:
        nltk.download("stopwords")


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration parameters from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_youtube_client(api_key: str):
    """Create a YouTube Data API client using the provided API key."""
    return build("youtube", "v3", developerKey=api_key)


def resolve_channel_id(youtube, identifier: str) -> str:
    """Resolve a channel handle or name to an actual channel ID."""
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
    """Fetch the latest videos from a channel."""
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
    Attempt to fetch a transcript via youtube-transcript-api. If
    ``list_transcripts`` is unavailable, fall back to ``get_transcript``.

    :param video_id: The YouTube video ID.
    :param languages: Optional list of language codes in preference order.
    :returns: Concatenated transcript text, or None if unavailable.
    """
    if languages is None:
        languages = ["ja", "en"]
    # If list_transcripts exists (newer versions), use it
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            transcript_list = None
        if transcript_list is not None:
            # Try preferred languages
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
    # Fallback: try get_transcript for each language
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
    """
    Download the audio for a given YouTube video ID. This function
    attempts to use ``pytube`` first. If ``pytube`` fails (e.g. due to
    HTTP 400 errors) or no audio stream is found, it falls back to
    ``yt-dlp`` to fetch the best available audio track. The downloaded
    file path is returned.

    :param video_id: The YouTube video ID.
    :param temp_dir: Temporary directory to store the downloaded file.
    :returns: Path to the downloaded audio file.
    :raises: RuntimeError if both download methods fail.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    # First attempt: pytube
    try:
        yt = YouTube(url)
        audio_stream = (
            yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        )
        if audio_stream is not None:
            filename = f"{video_id}.mp4"
            return audio_stream.download(output_path=temp_dir, filename=filename)
    except Exception:
        # If pytube fails, we'll try yt-dlp below
        pass
    # Second attempt: yt-dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "outtmpl": os.path.join(temp_dir, f"{video_id}.%(ext)s"),
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            ext = info_dict.get("ext", "m4a")
            return os.path.join(temp_dir, f"{video_id}.{ext}")
    except Exception as e:
        raise RuntimeError(f"Audio download failed for {video_id}: {e}")


def transcribe_audio(file_path: str, model: WhisperModel) -> str:
    """Transcribe an audio file using faster-whisper."""
    segments, _ = model.transcribe(file_path, beam_size=5)
    return " ".join([segment.text.strip() for segment in segments])


def summarise_text(text: str, sentences: int = 5) -> str:
    """Summarise the given text using Sumy's LSA summariser with Luhn fallback."""
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
    """Write the summary and metadata to a Markdown file."""
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
    """Main entry point for the script."""
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