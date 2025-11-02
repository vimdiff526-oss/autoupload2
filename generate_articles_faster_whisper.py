"""
generate_articles_faster_whisper.py

このスクリプトは、YouTube 動画から字幕を取得できない場合でも
自動で音声をダウンロードして文字起こしを行い、要約付き記事を
作成するためのツールです。音声の文字起こしには CTranslate2
ベースの `faster-whisper` ライブラリを用いるため、PyTorch
依存の `openai-whisper` が動作しない環境でも利用できます。

主な処理の流れは以下の通りです。

1. YouTube Data API を用いて指定チャンネルの最新動画を取得する。
2. `youtube-transcript-api` を試して字幕を取得し、存在すればそのテキストを用いる。
3. 字幕がない場合は `pytube` で音声のみをダウンロードし、
   `faster-whisper` で文字起こしを行う。
4. Sumy を用いてテキストを要約し、Markdown 形式の記事を
   指定ディレクトリに保存する。

設定はリポジトリ直下の `config.json` から読み込み、API キーは
環境変数 `YOUTUBE_API_KEY` から取得します。必要なライブラリは
`requirements_faster_whisper.txt` にまとめてあります。

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
from faster_whisper import WhisperModel

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
import nltk

# 言語設定（Sumy で使用）
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
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_youtube_client(api_key: str):
    """Create a YouTube Data API client using the provided API key."""
    return build("youtube", "v3", developerKey=api_key)


def resolve_channel_id(youtube, identifier: str) -> str:
    """
    Resolve a channel identifier. If ``identifier`` starts with "UC" it is
    assumed to be a channel ID and is returned directly. If it starts with
    "@" it is treated as a handle and the corresponding channel ID is
    looked up via the search API. Otherwise it is searched for as a
    channel name.
    """
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
    """
    Fetch the most recent videos from a given YouTube channel ID.

    Returns a list of search result items containing video IDs and metadata.
    """
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
    Attempt to fetch captions for a video using ``youtube-transcript-api``.
    Returns the concatenated transcript text if available, otherwise None.
    """
    if languages is None:
        languages = ["ja", "en"]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    # Try to find a transcript in one of the preferred languages
    for lang in languages:
        try:
            transcript = transcript_list.find_transcript([lang])
            segments = transcript.fetch()
            return " ".join(segment["text"] for segment in segments)
        except NoTranscriptFound:
            continue
    # Fallback: pick any available transcript
    try:
        transcript = transcript_list.find_transcript([
            t.language_code for t in transcript_list
        ])
        segments = transcript.fetch()
        return " ".join(segment["text"] for segment in segments)
    except NoTranscriptFound:
        return None


def download_audio(video_id: str, temp_dir: str) -> str:
    """
    Download the audio track of a YouTube video to a temporary file.
    Returns the path to the downloaded file.
    """
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
    """
    Transcribe an audio file using faster-whisper's WhisperModel.
    Returns the transcribed text.
    """
    # The model's transcribe method returns a generator of segments; join texts
    segments, _ = model.transcribe(file_path, beam_size=5)
    return " ".join([segment.text.strip() for segment in segments])


def summarise_text(text: str, sentences: int = 5) -> str:
    """
    Summarise a block of text using Sumy. By default LSA summarisation is used
    with Luhn as fallback.
    """
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
    """
    Construct a Markdown article for a video and write it to a file in
    ``output_dir``. The filename is based on the video ID.
    """
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
    """
    Main entry point. Loads configuration, initialises services and processes
    videos to generate summary articles.
    """
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
    # Load Whisper model lazily using faster-whisper. Choose model size
    model_size = config.get("whisper_model", "small")
    print(f"Loading faster-whisper model '{model_size}'...")
    # Compute device: use CPU to avoid CUDA requirement; compute_type can be 'int8' or 'float16'
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    youtube = get_youtube_client(api_key)
    channel_id = resolve_channel_id(youtube, identifier)
    print(f"Resolved channel '{identifier}' to ID '{channel_id}'")
    videos = fetch_latest_videos(youtube, channel_id, max_results=max_results)
    if not videos:
        print("No videos found")
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        for video in videos:
            vid = video["id"]["videoId"]
            print(f"Processing {vid}...")
            transcript = try_youtube_transcript(vid)
            transcript_source = "YouTube captions"
            if not transcript:
                # Fall back to audio transcription via faster-whisper
                print("No captions available. Downloading audio and transcribing...")
                try:
                    audio_path = download_audio(vid, temp_dir)
                    transcript = transcribe_audio(audio_path, whisper_model)
                    transcript_source = "faster-whisper transcription"
                except Exception as e:
                    print(f"Transcription failed for {vid}: {e}")
                    continue
            if not transcript:
                print(f"No transcript obtained for {vid}, skipping...")
                continue
            summary = summarise_text(transcript)
            write_markdown(video, summary, posts_dir, transcript_source)
            print(f"Generated summary for {vid} -> {posts_dir}/{vid}.md")


if __name__ == "__main__":
    main()