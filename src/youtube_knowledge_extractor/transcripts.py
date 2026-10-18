"""Phase 2: Transcript extraction."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    VideoUnavailable,
    NoTranscriptFound,
    IpBlocked,
    RequestBlocked,
)

from rich.console import Console

from .config import Config
from .db import (
    get_transcript,
    save_transcript,
    update_video_status,
)
from .models import Transcript, TranscriptSegment, Video, VideoStatus


console = Console()

# Preferred language priority: Norwegian variants, then English
PREFERRED_LANGS = [
    "no", "nb", "nn",  # Norwegian
    "en", "en-US", "en-GB", "en-CA",  # English
]


def _fetch_with_ytdlp(video_id: str) -> Optional[Transcript]:
    """Fallback: use yt-dlp to download subtitles as JSON3."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try preferred languages
        for lang in ["en"] + PREFERRED_LANGS:
            outtmpl = Path(tmpdir) / "%(title)s"
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs", lang,
                "--sub-format", "json3",
                "-o", str(outtmpl),
                "--quiet",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if result.returncode != 0:
                continue
            json_files = list(Path(tmpdir).glob(f"*{video_id}*.{lang}.json3")) or list(Path(tmpdir).glob("*.json3"))
            if not json_files:
                continue
            try:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                events = data.get("events", [])
                segments: list[TranscriptSegment] = []
                for event in events:
                    start_ms = event.get("tStartMs", 0)
                    dur_ms = event.get("dDurationMs", 0)
                    segs = event.get("segs", [])
                    text = "".join(seg.get("utf8", "") for seg in segs if seg.get("utf8"))
                    if text.strip():
                        segments.append(
                            TranscriptSegment(
                                start=start_ms / 1000.0,
                                duration=dur_ms / 1000.0,
                                text=text.strip(),
                            )
                        )
                return Transcript(
                    video_id=video_id,
                    language=lang,
                    is_generated=True,  # yt-dlp auto-subs are generated
                    segments=segments,
                )
            except Exception:
                continue
    return None


def fetch_transcript(video: Video, force: bool = False) -> Optional[Transcript]:
    """Fetch transcript for a video.

    Returns Transcript or None if unavailable.
    Caches result in DB and local JSON.
    """
    # Check cache unless force
    if not force:
        cached = get_transcript(video.video_id, data_dir=Config.DATA_DIR)
        if cached is not None:
            console.print(f"  [green]Cached transcript for {video.video_id}[/green]")
            return cached

    try:
        # youtube-transcript-api v1.x: instantiate the API and use .list()
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video.video_id)
        try:
            ytt = transcript_list.find_transcript(PREFERRED_LANGS)
        except NoTranscriptFound:
            # Fallback to any available
            all_trans = list(transcript_list)
            if not all_trans:
                raise NoTranscriptFound(video.video_id)
            ytt = all_trans[0]

        raw_segments = ytt.fetch()
        is_generated = getattr(ytt, "is_generated", True)
        language = getattr(ytt, "language_code", "unknown")

        segments = [
            TranscriptSegment(
                start=getattr(seg, "start", 0.0),
                duration=getattr(seg, "duration", 0.0),
                text=getattr(seg, "text", ""),
            )
            for seg in raw_segments
        ]

        transcript = Transcript(
            video_id=video.video_id,
            language=language,
            is_generated=is_generated,
            segments=segments,
        )

        # Save to DB
        save_transcript(transcript, data_dir=Config.DATA_DIR)
        # Save raw JSON for inspection
        raw_path = Config.TRANSCRIPTS_DIR / f"{video.video_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "video_id": video.video_id,
                    "language": language,
                    "is_generated": is_generated,
                    "segments": [
                        {"start": s.start, "duration": s.duration, "text": s.text}
                        for s in segments
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        update_video_status(
            video.video_id, VideoStatus.TRANSCRIPT_OK, data_dir=Config.DATA_DIR
        )
        console.print(f"  [green]Transcript OK for {video.video_id} ({transcript.word_count} words, {language})[/green]")
        return transcript

    except (IpBlocked, RequestBlocked, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception) as e:
        error_name = type(e).__name__
        console.print(f"  [yellow]youtube-transcript-api {error_name} for {video.video_id}, trying yt-dlp fallback...[/yellow]")
        fallback = _fetch_with_ytdlp(video.video_id)
        if fallback is not None:
            save_transcript(fallback, data_dir=Config.DATA_DIR)
            raw_path = Config.TRANSCRIPTS_DIR / f"{video.video_id}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "video_id": video.video_id,
                        "language": fallback.language,
                        "is_generated": fallback.is_generated,
                        "segments": [
                            {"start": s.start, "duration": s.duration, "text": s.text}
                            for s in fallback.segments
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            update_video_status(
                video.video_id, VideoStatus.TRANSCRIPT_OK, data_dir=Config.DATA_DIR
            )
            console.print(f"  [green]Transcript OK for {video.video_id} ({fallback.word_count} words, {fallback.language}, yt-dlp fallback)[/green]")
            return fallback

        if isinstance(e, (TranscriptsDisabled, NoTranscriptFound)):
            error_msg = str(e) if str(e) else "No transcript available."
        elif isinstance(e, (IpBlocked, RequestBlocked)):
            error_msg = "IP blocked and yt-dlp fallback failed."
        elif isinstance(e, VideoUnavailable):
            error_msg = "Video unavailable."
        else:
            error_msg = str(e)[:500] if str(e) else "Unknown error fetching transcript."

        update_video_status(
            video.video_id,
            VideoStatus.TRANSCRIPT_UNAVAILABLE,
            error_message=error_msg,
            data_dir=Config.DATA_DIR,
        )
        console.print(f"  [red]Transcript unavailable for {video.video_id}: {error_msg}[/red]")

    return None


def process_transcripts(videos: list[Video], force: bool = False) -> dict[str, int]:
    """Fetch transcripts for a list of videos.

    Returns counts of successes and failures.
    """
    success = 0
    failed = 0
    for video in videos:
        if video.status == VideoStatus.INACCESSIBLE:
            continue
        if not force:
            existing = get_transcript(video.video_id, data_dir=Config.DATA_DIR)
            if existing is not None:
                success += 1
                continue

        result = fetch_transcript(video, force=force)
        if result is not None:
            success += 1
        else:
            failed += 1
        time.sleep(0.5)  # Respect rate limits

    return {"success": success, "failed": failed}
