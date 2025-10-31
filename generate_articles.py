"""
Script to fetch transcripts from YouTube, summarise them and generate
Markdown articles. The script uses the YouTube Data API to discover
recent videos on a channel (or handle), downloads the video's
transcript using the open-source `youtube-transcript-api` library and
summarises the transcript using the Sumy library.  Articles are
written to the directory specified in the configuration file.

Usage::

    export YOUTUBE_API_KEY=<your API key>
    python generate_articles.py

The configuration for the script is stored in ``config.json``.

Note:
    - This script does not perform any authentication against note.com;
      instead it focuses purely on gathering and summarising content.
    - It does not commit files to GitHub; you should run it locally or
      in CI and commit the resulting Markdown files manually.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
import nltk

LANGUAGE = "english"  # Sumy supports multiple languages; english works reasonably for mixed language transcripts

def download_punkt() -> None:
    """Ensure the punkt tokenizer models are downloaded for NLTK."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_youtube_client(api_key: str):
    """Create a YouTube API client using the provided API key."""
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
    # Search for channel handle or name
    search_response = youtube.search().list(
        q=identifier,
        part="id,snippet",
        type="channel",
        maxResults=1
    ).execute()
    items = search_response.get("items", [])
    if not items:
        raise ValueError(f"Could not resolve channel identifier '{identifier}'")
    return items[0]["id"]["channelId"]


def fetch_latest_videos(youtube, channel_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch the most recent videos from a given YouTube channel ID.

    Returns a list of search result items containing video IDs and metadata.
    """
    search_response = youtube.search().list(
        channelId=channel_id,
        part="id,snippet",
        order="date",
        maxResults=max_results,
        type="video"
    ).execute()
    return search_response.get("items", [])


def fetch_transcript(video_id: str, languages: List[str] = None) -> str:
    """
    Retrieve the transcript for a video. If no transcript is available
    this function returns ``None``.

    :param video_id: The ID of the YouTube video.
    :param languages: Preferred languages for the transcript. If None a
        default list of common languages is tried.
    """
    if languages is None:
        languages = ["ja", "en"]  # try Japanese then English
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        # Try to find a transcript in one of the preferred languages
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except NoTranscriptFound:
                continue
        if transcript is None:
            # Fallback to any transcript (including auto-generated)
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        fetched = transcript.fetch()
        return " ".join([segment["text"] for segment in fetched])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None


def summarise_text(text: str, sentences: int = 5) -> str:
    """
    Summarise a block of text using an extractive summariser. We default
    to LSA summarisation but fall back to Luhn summarisation for
    robustness if LSA yields no output. ``sentences`` controls the number
    of sentences in the summary.
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


def write_markdown(video: Dict[str, Any], summary: str, output_dir: str) -> None:
    """
    Construct a Markdown article for a video and write it to a file in
    ``output_dir``. The filename is based on the video ID.
    """
    vid = video["id"]["videoId"]
    snippet = video["snippet"]
    title = snippet["title"].strip()
    channel_title = snippet.get("channelTitle", "")
    published_at = snippet.get("publishedAt", "")
    url = f"https://www.youtube.com/watch?v={vid}"
    date_obj = None
    if published_at:
        try:
            date_obj = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            date_obj = None
    date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else published_at
    md_content = f"""# {title}

- **チャンネル**: {channel_title}
- **公開日**: {date_str}
- **動画URL**: {url}

## 要約

{summary}
"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{vid}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)


def main():
    # Ensure necessary NLTK data is available
    download_punkt()
    nltk.download("stopwords")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY environment variable must be set")
    config = load_config()
    identifier = config.get("channel_identifier")
    if not identifier:
        raise RuntimeError("'channel_identifier' must be provided in config.json")
    max_results = int(config.get("max_results", 5))
    output_dir = config.get("posts_dir", "posts")
    youtube = get_youtube_client(api_key)
    channel_id = resolve_channel_id(youtube, identifier)
    videos = fetch_latest_videos(youtube, channel_id, max_results=max_results)
    for video in videos:
        vid = video["id"]["videoId"]
        transcript = fetch_transcript(vid)
        if not transcript:
            print(f"No transcript available for {vid}, skipping…")
            continue
        summary = summarise_text(transcript)
        write_markdown(video, summary, output_dir)
        print(f"Generated summary for {vid}")


if __name__ == "__main__":
    main()
