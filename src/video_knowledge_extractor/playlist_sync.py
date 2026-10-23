"""Automatic playlist sync: find new videos in a YouTube playlist and queue them.

Designed to run as a one-shot command (e.g. from a Windows scheduled task):

    python src/cli.py sync-playlist "https://www.youtube.com/playlist?list=..."

It fetches the full playlist, compares it with the library (the ``videos``
table) and the job queue, and enqueues only videos that are genuinely new.
A running worker (``python src/cli.py worker``) picks the queued jobs up and
runs the full pipeline (transcribe -> analyze -> embed).  Re-running the sync
is safe: videos already in the library (or already in the queue) are never
queued again, so nothing is analyzed twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import get_connection, save_video
from .discovery import discover_from_ytdlp
from .jobs import enqueue_job, init_jobs_table
from .models import Video, VideoStatus


#: Job-queue source labels (shown in the dashboard queue history).
SYNC_SOURCE = "playlist-sync"
RETRY_SOURCE = "playlist-sync-retry"

#: Library statuses that count as failed — worth retrying with --retry-failed.
FAILED_STATUSES = {
    VideoStatus.INACCESSIBLE.value,
    VideoStatus.ANALYSIS_FAILED.value,
    VideoStatus.TRANSCRIPT_UNAVAILABLE.value,
}


@dataclass
class SyncResult:
    """What one sync run found and did."""

    playlist_url: str
    dry_run: bool = False
    fetched: list[Video] = field(default_factory=list)
    new: list[Video] = field(default_factory=list)        # enqueued (or would be)
    retried: list[Video] = field(default_factory=list)    # failed known videos, enqueued
    known: list[Video] = field(default_factory=list)      # already in the library, skipped
    in_queue: list[Video] = field(default_factory=list)   # already queued, not yet processed
    jobs: list[dict[str, Any]] = field(default_factory=list)  # actual enqueued job records


def _known_video_ids(data_dir: Path) -> dict[str, str]:
    """Map video_id -> status for every video already in the library."""
    with get_connection(data_dir) as conn:
        rows = conn.execute("SELECT video_id, status FROM videos").fetchall()
    return {r[0]: r[1] for r in rows}


def _queued_identifiers(data_dir: Path) -> set[str]:
    """Video ids and URLs of jobs that have not reached a terminal state.

    Used to avoid double-enqueueing a video that the worker has already
    picked up (it may not be in the videos table yet).
    """
    init_jobs_table(data_dir)
    identifiers: set[str] = set()
    with get_connection(data_dir) as conn:
        rows = conn.execute(
            "SELECT video_id, url FROM jobs "
            "WHERE status NOT IN ('done', 'failed', 'cancelled')"
        ).fetchall()
    for r in rows:
        if r[0]:
            identifiers.add(r[0])
        if r[1]:
            identifiers.add(r[1])
    return identifiers


def sync_playlist(
    playlist_url: str,
    data_dir: Path = Path("data"),
    limit: int | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
    options: dict[str, Any] | None = None,
) -> SyncResult:
    """Fetch *playlist_url* and enqueue the videos that are new to the library.

    Args:
        playlist_url: Public YouTube playlist URL.
        data_dir: VKE data directory (holds knowledge_base.db).
        limit: Only consider the first N videos of the playlist.
        dry_run: Compute the diff without saving or enqueuing anything.
        retry_failed: Also re-queue known videos whose earlier processing
            ended in a failed status (inaccessible / analysis_failed /
            transcript_unavailable).
        options: Pipeline options for the new jobs (default: the queue's
            standard options — transcribe + analyze + embed).

    Returns:
        A SyncResult with the fetched list and what was enqueued/skipped.

    Raises:
        ValueError: The URL is not a YouTube playlist/URL.
        RuntimeError: yt-dlp failed (network, private playlist, ...).
    """
    result = SyncResult(playlist_url=playlist_url, dry_run=dry_run)
    result.fetched = discover_from_ytdlp(playlist_url, limit=limit)

    known = _known_video_ids(data_dir)
    queued = _queued_identifiers(data_dir)

    for v in result.fetched:
        if v.video_id in known:
            if retry_failed and known[v.video_id] in FAILED_STATUSES and v.video_id not in queued:
                result.retried.append(v)
            else:
                result.known.append(v)
        elif v.video_id in queued or v.url in queued:
            result.in_queue.append(v)
        else:
            result.new.append(v)

    def _enqueue(v: Video, source: str) -> None:
        save_video(v, data_dir=data_dir)
        result.jobs.append(
            enqueue_job(
                source=source,
                url=v.url,
                platform="youtube",
                title=v.title,
                options=options,
                data_dir=data_dir,
            )
        )

    if not dry_run:
        for v in result.new:
            _enqueue(v, SYNC_SOURCE)
        for v in result.retried:
            _enqueue(v, RETRY_SOURCE)

    return result
