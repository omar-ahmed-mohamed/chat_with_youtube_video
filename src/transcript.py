from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

def extract_video_id(video_url):
    parsed_url = urlparse(video_url.strip())
    host = parsed_url.netloc.lower().replace("www.", "")

    if host in ("youtube.com", "m.youtube.com"):
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
            if video_id:
                return video_id

        path_parts = parsed_url.path.strip("/").split("/")

        if (
            len(path_parts) >= 2
            and path_parts[0] in ("shorts", "embed", "live")
        ):
            return path_parts[1]

    if host == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]

        if video_id:
            return video_id

    raise ValueError("Invalid or unsupported YouTube URL")

def fetch_transcript(video_id):
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "ar"],
        )

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video")

    except NoTranscriptFound:
        raise ValueError(
            "No English or Arabic transcript was found"
        )

    except VideoUnavailable:
        raise ValueError("The YouTube video is unavailable")

    chunks = []

    for item in fetched_transcript:
        minutes = int(item.start // 60)
        seconds = int(item.start % 60)

        chunks.append({
            "video_id": video_id,
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "start_seconds": item.start,
            "duration": item.duration,
            "text": item.text.replace("\n", " ").strip(),
        })

    return chunks

def create_timestamp_url(video_id, seconds):
    adjusted_seconds = max(0, int(seconds) - 5)

    return (
        "https://www.youtube.com/watch"
        f"?v={video_id}&t={adjusted_seconds}s"
    )

