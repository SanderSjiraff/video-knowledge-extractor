"""Phase 1: Playlist and video discovery."""

from __future__ import annotations

import re
import subprocess
import sys
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Any

from .config import Config
from .db import save_video
from .models import Video, VideoStatus


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.scheme not in {"http", "https"}:
        return None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
            candidate = parts[1]
        else:
            return None
    else:
        return None
    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None


def detect_platform(url: str) -> str | None:
    """Return the platform for a URL: 'youtube', 'tiktok', 'instagram', or None."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}:
        return "youtube"
    if host == "tiktok.com" or host.endswith(".tiktok.com"):
        return "tiktok"
    if host in {"instagram.com", "instagr.am"} or host.endswith(".instagram.com"):
        return "instagram"
    return None


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")


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
    _validate_limit(limit)
    parsed = urlparse(playlist_url)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower().removeprefix("www.") not in {
        "youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"
    }:
        raise ValueError("source must be an HTTP(S) YouTube URL or an existing text file")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--quiet",
        playlist_url,
    ]
    if limit is not None:
        cmd.extend(["--playlist-end", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    import json
    data = json.loads(result.stdout)
    entries = data.get("entries")
    if entries is None:
        entries = [data] if data.get("id") else []
    videos: list[Video] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        vid = entry.get("id")
        if not isinstance(vid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) or vid in seen_ids:
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


def _yt_dlp_single_video(url: str) -> dict | None:
    """Fetch a single video's metadata with yt-dlp (any supported platform)."""
    import json

    cmd = [sys.executable, "-m", "yt_dlp", "--dump-single-json", "--quiet", url]
    if Config.COOKIES_FILE:
        cmd.extend(["--cookies", Config.COOKIES_FILE])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except ValueError:
        return None


def discover_from_file(filepath: Path, limit: int | None = None) -> list[Video]:
    """Read a text file with one video URL per line (YouTube, TikTok, Instagram).

    YouTube URLs are resolved from the URL itself (no network needed).
    TikTok/Instagram URLs are resolved through yt-dlp so a stable video id
    can be used as the primary key.
    """
    _validate_limit(limit)
    lines = filepath.read_text(encoding="utf-8").splitlines()
    videos: list[Video] = []
    seen_ids: set[str] = set()
    order = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        platform = detect_platform(line)
        if platform is None:
            continue

        if platform == "youtube":
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
                    platform="youtube",
                )
            )
        else:
            # TikTok / Instagram: resolve a stable id via yt-dlp metadata.
            meta = _yt_dlp_single_video(line)
            vid = meta.get("id") if meta else None
            if not isinstance(vid, str) or vid in seen_ids:
                continue
            seen_ids.add(vid)
            order += 1
            duration = meta.get("duration")
            videos.append(
                Video(
                    video_id=vid,
                    title=meta.get("title", "Untitled") or "Untitled",
                    url=line,
                    duration_seconds=int(duration) if isinstance(duration, (int, float)) else 0,
                    channel=meta.get("channel") or meta.get("uploader") or "Unknown",
                    playlist_order=order,
                    status=VideoStatus.DISCOVERED,
                    platform=platform,
                )
            )

        if limit is not None and len(videos) >= limit:
            break

    return videos


def enrich_video_metadata(video: Video) -> Video:
    """Use yt-dlp to fetch metadata for a single video."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-single-json",
        "--quiet",
        video.url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
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
    _validate_limit(limit)
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        videos = discover_from_file(source_path, limit=limit)
    else:
        videos = discover_from_ytdlp(source, limit=limit)

    # If from file, fetch metadata with yt-dlp for entries still missing it
    # (YouTube URLs have no title yet; TikTok/Instagram were resolved above).
    if source_path.exists():
        from rich.console import Console
        console = Console()
        to_enrich = [v for v in videos if not v.title]
        if to_enrich:
            console.print(f"[cyan]Enriching metadata for {len(to_enrich)} videos...[/cyan]")
        for v in videos:
            if not v.title:
                v = enrich_video_metadata(v)
            save_video(v, data_dir=data_dir)
    else:
        for v in videos:
            save_video(v, data_dir=data_dir)

    return videos
