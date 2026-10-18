"""Phase 1: Playlist and video discovery."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .db import save_video
from .models import Video, VideoStatus


YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    match = YOUTUBE_URL_RE.search(url)
    return match.group(1) if match else None


def parse_duration_to_seconds(duration_str: str | None) -> int:
    """Parse ISO 8601 duration or HH:MM:SS / MM:SS to seconds."""
    if not duration_str:
        return 0
    d = duration_str.strip()
    # ISO 8601 e.g. PT1H23M45S
    if d.startswith("PT"):
        total = 0
        h = re.search(r"(\d+)H", d)
        m = re.search(r"(\d+)M", d)
        s = re.search(r"(\d+)S", d)
        if h:
            total += int(h.group(1)) * 3600
        if m:
            total += int(m.group(1)) * 60
        if s:
            total += int(s.group(1))
        return total
    # Colon-separated
    parts = d.split(":")
    parts = [p for p in parts if p]
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 1:
        return int(parts[0])
    return 0


def discover_from_ytdlp(
    playlist_url: str,
    limit: int | None = None,
) -> list[Video]:
    """Use yt-dlp to list videos in a playlist without downloading."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--quiet",
        playlist_url,
    ]
    if limit:
        cmd.extend(["--playlist-end", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    import json
    data = json.loads(result.stdout)
    entries = data.get("entries", [])
    videos: list[Video] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(entries, start=1):
        vid = entry.get("id")
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        duration = entry.get("duration")
        if duration is None:
            duration = parse_duration_to_seconds(entry.get("duration_string"))
        else:
            duration = int(duration)

        video = Video(
            video_id=vid,
            title=entry.get("title", "Untitled"),
            url=f"https://www.youtube.com/watch?v={vid}",
            duration_seconds=duration,
            channel=entry.get("channel") or entry.get("uploader") or "Unknown",
            playlist_order=idx,
            status=VideoStatus.DISCOVERED,
        )
        videos.append(video)

    return videos


def discover_from_file(
    filepath: Path,
) -> list[Video]:
    """Read a text file with one YouTube URL per line."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    videos: list[Video] = []
    seen_ids: set[str] = set()
    order = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vid = extract_video_id(line)
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        order += 1
        # We don't know duration/title yet — mark for enrichment
        videos.append(
            Video(
                video_id=vid,
                title="",
                url=f"https://www.youtube.com/watch?v={vid}",
                duration_seconds=0,
                channel="",
                playlist_order=order,
                status=VideoStatus.DISCOVERED,
            )
        )

    return videos


def enrich_video_metadata(video: Video) -> Video:
    """Use yt-dlp to fetch metadata for a single video."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-single-json",
        "--quiet",
        video.url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        video.status = VideoStatus.INACCESSIBLE
        video.error_message = result.stderr.strip()[:500]
        return video

    import json
    data = json.loads(result.stdout)
    video.title = data.get("title", video.title) or "Untitled"
    video.channel = data.get("channel") or data.get("uploader") or "Unknown"
    duration = data.get("duration")
    if duration is not None:
        video.duration_seconds = int(duration)
    else:
        video.duration_seconds = parse_duration_to_seconds(data.get("duration_string"))
    return video


def discover_playlist(
    source: str,
    limit: int | None = None,
    data_dir: Path = Path("data"),
) -> list[Video]:
    """Main discovery entry point.

    Args:
        source: YouTube playlist URL or path to a text file.
        limit: Optional max videos to discover.

    Returns:
        List of discovered Video objects.
    """
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        videos = discover_from_file(source_path)
    else:
        videos = discover_from_ytdlp(source, limit=limit)

    # If from file, we need metadata; fetch with yt-dlp in batches
    if source_path.exists():
        from rich.console import Console
        console = Console()
        console.print(f"[cyan]Enriching metadata for {len(videos)} videos...[/cyan]")
        for v in videos:
            v = enrich_video_metadata(v)
            save_video(v, data_dir=data_dir)
    else:
        for v in videos:
            save_video(v, data_dir=data_dir)

    return videos
