"""Background worker: pushes queued jobs through the processing pipeline.

The worker is a separate long-running process (``python src/cli.py worker``).
It polls the SQLite job queue and, for each job, runs:

    discover  ->  transcribe  ->  [OCR]  ->  [vision]  ->  analyze  ->  [embed]

which stages run is controlled by the job's options.  OCR/vision/embed
failures are non-fatal (recorded on the job); a failed download or
transcription stops the job as failed.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from .config import Config
from .db import (
    get_transcript,
    get_video,
    init_db,
    save_video,
)
from .discovery import (
    _yt_dlp_single_video,
    enrich_video_metadata,
    extract_video_id,
)
from .jobs import claim_next_job, fail_job, update_job
from .models import Video, VideoStatus

console = Console()


class JobAborted(Exception):
    """Raised to abort a job with a user-readable reason."""


def resolve_video(url: str, platform: str) -> Video:
    """Turn a submitted URL (or local file path) into a Video record.

    YouTube ids come straight from the URL (metadata is enriched
    best-effort).  TikTok/Instagram need one yt-dlp metadata call to get a
    stable id.  Local file paths are hashed into a stable id.
    """
    if platform == "local":
        path = Path(url)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        return Video(
            video_id=f"local-{digest}",
            title=path.stem,
            url=str(path.resolve()),
            duration_seconds=0,
            channel="Lokal fil",
            playlist_order=0,
            status=VideoStatus.DISCOVERED,
            platform="local",
        )

    if platform == "youtube":
        vid = extract_video_id(url)
        if not vid:
            raise JobAborted("Gjente YouTube-lenke")
        video = Video(
            video_id=vid,
            title="",
            url=f"https://www.youtube.com/watch?v={vid}",
            duration_seconds=0,
            channel="",
            playlist_order=0,
            status=VideoStatus.DISCOVERED,
            platform="youtube",
        )
        # Metadata (title/channel/duration) is best-effort; a transient
        # yt-dlp failure should not kill the whole job.
        try:
            enrich_video_metadata(video)
        except Exception:  # noqa: BLE001
            pass
        if video.status == VideoStatus.INACCESSIBLE:
            # Don't keep the inaccessible flag from a flaky metadata fetch.
            video.status = VideoStatus.DISCOVERED
            video.error_message = None
        return video

    # tiktok / instagram: stable id via yt-dlp metadata
    meta = _yt_dlp_single_video(url)
    vid = meta.get("id") if meta else None
    if not isinstance(vid, str) or not vid:
        # Photo/carousel posts have no audio; give a clear, user-facing reason.
        # Short links (vm.tiktok.com/...) must be resolved first.
        final_url = url
        try:
            import requests

            resp = requests.head(url, allow_redirects=True, timeout=10)
            final_url = str(resp.url)
        except Exception:  # noqa: BLE001 - best effort only
            pass
        if "/photo/" in final_url:
            raise JobAborted("Dette er et bildeinnlegg (karusell), ikke en video – ingen lyd å transkribere.")
        raise JobAborted(f"Kunne ikke hente metadata fra {platform} (yt-dlp feilet eller lenken er ikke gyldig)")
    duration = meta.get("duration")
    return Video(
        video_id=vid,
        title=meta.get("title", "Untitled") or "Untitled",
        url=url,
        duration_seconds=int(duration) if isinstance(duration, (int, float)) else 0,
        channel=meta.get("channel") or meta.get("uploader") or "Unknown",
        playlist_order=0,
        status=VideoStatus.DISCOVERED,
        platform=platform,
    )


def _stage_transcribe(video: Video, data_dir: Path) -> None:
    """Fetch a transcript: subtitles first for YouTube, Whisper elsewhere."""
    if video.platform == "youtube":
        from .transcripts import process_transcripts

        process_transcripts([video], force=False)
        if get_transcript(video.video_id, data_dir=data_dir) is not None:
            return
    # TikTok/Instagram/local, or YouTube without subtitles: local Whisper.
    from .whisper_transcripts import process_whisper_transcripts

    process_whisper_transcripts([video], force=False, keep_audio=True)


def process_job(job: dict, data_dir: Optional[Path] = None) -> None:
    """Run the full pipeline for one claimed job. Updates the job as it goes."""
    data_dir = data_dir or Config.DATA_DIR
    opts = job["options"] or {}
    job_id = job["job_id"]
    note = ""

    # --- Stage 1: discover -------------------------------------------------
    video = resolve_video(job["url"], job["platform"])
    save_video(video, data_dir=data_dir)
    # Keep the submitter-provided title (e.g. the original upload file name);
    # remote URLs get their title from metadata resolution.
    update_job(
        job_id,
        video_id=video.video_id,
        title=job.get("title") or video.title,
        status="transcribing" if opts.get("transcribe") else (
            "analyzing" if opts.get("analyze") else "done"
        ),
        progress=15,
        data_dir=data_dir,
    )
    if not opts.get("transcribe") and not opts.get("analyze"):
        update_job(job_id, status="done", progress=100, note=note or "Ingen trinn valgt", data_dir=data_dir)
        return

    # --- Stage 2: transcript ----------------------------------------------
    if opts.get("transcribe"):
        update_job(job_id, status="transcribing", progress=15, data_dir=data_dir)
        _stage_transcribe(video, data_dir)
        transcript = get_transcript(video.video_id, data_dir=data_dir)
        if transcript is None:
            stored = get_video(video.video_id, data_dir=data_dir)
            detail = (stored.error_message if stored else "") or "Whisper fant ingen tale"
            update_job(
                job_id,
                status="done",
                progress=100,
                note=f"Ingen transkript: {detail}",
                data_dir=data_dir,
            )
            console.print(f"  [yellow]{video.video_id}: no transcript, job finished[/yellow]")
            return

    # --- Stage 3: on-screen text (OCR) -- optional, non-fatal ---------------
    if opts.get("ocr"):
        update_job(job_id, status="ocr", progress=45, data_dir=data_dir)
        try:
            from .visual import process_on_screen_text

            result = process_on_screen_text([video], force=False, keep_media=False)
            if result.get("failed"):
                note = f"OCR feilet for {result['failed']} video(er)"
        except Exception as exc:  # noqa: BLE001 - OCR is opt-in and best-effort
            note = f"OCR tilgjengelig: {exc}"

    # --- Stage 4: vision frame descriptions -- optional, non-fatal ----------
    if opts.get("vision"):
        update_job(job_id, status="vision", progress=60, data_dir=data_dir)
        try:
            from .vision import process_frame_descriptions

            result = process_frame_descriptions([video], force=False, keep_media=False)
            if result.get("failed"):
                note = f"Visjon feilet for {result['failed']} video(er)"
        except Exception as exc:  # noqa: BLE001 - vision is opt-in and best-effort
            note = f"Visjon utilgjengelig: {exc}"

    # --- Stage 5: analysis ---------------------------------------------------
    if opts.get("analyze"):
        update_job(job_id, status="analyzing", progress=75, data_dir=data_dir)
        from .orchestrator import run_analysis

        run_analysis(videos=[video], dry_run_limit=1)
        stored = get_video(video.video_id, data_dir=data_dir)
        if stored is not None and stored.status != VideoStatus.ANALYSIS_OK:
            note = (note + "; " if note else "") + "Analysen ble ikke ferdig (se status på videoen)"

    # --- Stage 6: embeddings -- optional, non-fatal --------------------------
    if opts.get("embed"):
        update_job(job_id, status="embedding", progress=90, data_dir=data_dir)
        try:
            from .embeddings import process_embeddings

            result = process_embeddings([video], force=False)
            if result.get("failed"):
                note = (note + "; " if note else "") + "embeddings feilet"
        except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
            note = (note + "; " if note else "") + f"embeddings utilgjengelig: {exc}"

    update_job(job_id, status="done", progress=100, note=note or None, data_dir=data_dir)
    console.print(f"  [green]Job {job_id[:8]} done:[/green] {video.title or video.video_id}")


def worker_loop(
    data_dir: Optional[Path] = None,
    poll_seconds: float = 2.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Poll the queue and process jobs until stop_event is set (or forever)."""
    data_dir = data_dir or Config.DATA_DIR
    init_db(data_dir)
    Config.ensure_dirs()
    console.print("[bold blue]Worker started — waiting for jobs (Ctrl+C to stop)[/bold blue]")
    while stop_event is None or not stop_event.is_set():
        job = claim_next_job(data_dir=data_dir)
        if job is None:
            # Small sleep granularity so Ctrl+C / stop_event stays responsive.
            slept = 0.0
            while slept < poll_seconds and (stop_event is None or not stop_event.is_set()):
                time.sleep(0.2)
                slept += 0.2
            continue
        console.print(f"\n[bold cyan]Job {job['job_id'][:8]} ({job['platform']}):[/bold cyan] {job['title'] or job['url']}")
        try:
            process_job(job, data_dir=data_dir)
        except JobAborted as exc:
            fail_job(job["job_id"], str(exc), data_dir=data_dir)
            console.print(f"  [red]Job aborted:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001 - a job must never kill the worker
            fail_job(job["job_id"], str(exc), data_dir=data_dir)
            console.print(f"  [red]Job failed:[/red] {exc}")
