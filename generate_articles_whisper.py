"""
generate_articles_whisper.py

このスクリプトは YouTube 動画から字幕または音声認識を通じてテキストを抽出し、
要約を生成して Markdown 記事として保存します。
"""

import json
import os
import shutil
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
import whisper

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
import nltk

LANGUAGE = "english"

def ensure_nltk_data() -> None:
    """Ensure NLTK tokenizer and stopwords are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.corpus.stopwords.words(LANGUAGE)
    except LookupError:
        nltk.download("stopwords")

def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_youtube_client(api_key: str):
    """Build a YouTube Data API client."""
    return build("youtube", "v3", developerKey=api_key)

def resolve_channel_id(youtube, identifier: str) -> str:
    """Resolve a YouTube handle or name to a channel ID."""
    identifier = identifier.strip()
    if identifier.startswith("UC"):
        return identifier
    search_response = (
        youtube.search()
        .list(q=identifier, part="id,snippet", type="channel", maxResults=1)
        .execute()
    )
    items = search_response.get("items", [])
    if not items:
        raise ValueError(f"Could not resolve channel identifier '{identifier}'")
    return items[0]["id"]["channelId"]

def fetch_latest_videos(
    youtube, channel_id: str, max_results: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve recent videos from a channel."""
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

def try_youtube_transcript(video_id: str, languages: Optional[List[str]] = None) -> Optional[str]:
    """Attempt to fetch captions via youtube-transcript-api."""
    if languages is None:
        languages = ["ja", "en"]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    # Try preferred languages
    for lang in languages:
        try:
            transcript = transcript_list.find_transcript([lang])
            segments = transcript.fetch()
            return " ".join(segment["text"] for segment in segments)
        except NoTranscriptFound:
            continue
    # Fallback to any transcript
    try:
        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        segments = transcript.fetch()
        return " ".join(segment["text"] for segment in segments)
    except NoTranscriptFound:
        return None

def download_audio(video_id: str, temp_dir: str) -> str:
    """Download the audio track of a YouTube video to a temporary file."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    audio_stream = (
        yt.streams.filter(only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )
    if audio_stream is None:
        raise RuntimeError(f"No audio stream found for video {video_id}")
    filename = f"{video_id}.mp4"
    output_path = audio_stream.download(output_path=temp_dir, filename=filename)
    return output_path

def transcribe_audio(file_path: str, whisper_model) -> str:
    """Transcribe audio file using Whisper."""
    result = whisper_model.transcribe(file_path, task="transcribe")
    return result.get("text", "").strip()

def summarise_text(text: str, sentences: int = 5) -> str:
    """Summarise text using Sumy (LSA with Luhn fallback)."""
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summariser = LsaSummarizer(stemmer)
    summariser.stop_words = nltk.corpus.stopwords.words(LANGUAGE)
    summary_sentences = summariser(parser.document, sentences)
    if not summary_sentences:
        summariser = LuhnSummarizer(stemmer)
        summariser.stop_words = nltk.corpus.stopwords.words(LANGUAGE)
        summary_sentences = summariser(parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary_sentences)

def write_markdown(video: Dict[str, Any], summary: str, output_dir: str, transcript_source: str) -> None:
    """Write the Markdown article."""
    vid = video["id"]["videoId"]
    snippet = video["snippet"]
    title = snippet.get("title", "").strip()
    channel_title = snippet.get("channelTitle", "").strip()
    published_at = snippet.get("publishedAt", "").strip()
    url = f"https://www.youtube.com/watch?v={vid}"
    # Format publication date
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
    """Main function."""
    ensure_nltk_data()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY environment variable must be set.")
    config = load_config()
    channel_identifier = config.get("channel_identifier")
    if not channel_identifier:
        raise RuntimeError("'channel_identifier' must be defined in config.json.")
    max_results = int(config.get("max_results", 5))
    posts_dir = config.get("posts_dir", "posts")
    whisper_model_name = config.get("whisper_model", "small")
    # Load Whisper
    print(f"Loading Whisper model '{whisper_model_name}'…")
    whisper_model = whisper.load_model(whisper_model_name)
    youtube = get_youtube_client(api_key)
    channel_id = resolve_channel_id(youtube, channel_identifier)
    print(f"Resolved channel '{channel_identifier}' to ID '{channel_id}'.")
    videos = fetch_latest_videos(youtube, channel_id, max_results=max_results)
    if not videos:
        print("No videos found for the specified channel.")
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        for video in videos:
            vid = video["id"]["videoId"]
            print(f"Processing {vid}…")
            transcript = try_youtube_transcript(vid)
            transcript_source = "YouTube captions"
            if not transcript:
                print("No captions available. Downloading audio and transcribing with Whisper…")
                try:
                    audio_path = download_audio(vid, temp_dir)
                    transcript = transcribe_audio(audio_path, whisper_model)
                    transcript_source = "Whisper transcription"
                except Exception as e:
                    print(f"Transcription failed for {vid}: {e}")
                    continue
            if not transcript:
                print(f"No transcript obtained for {vid}, skipping…")
                continue
            summary = summarise_text(transcript)
            write_markdown(video, summary, posts_dir, transcript_source)
            print(f"Wrote summary for {vid} to {posts_dir}/{vid}.md")

if __name__ == "__main__":
    main()
