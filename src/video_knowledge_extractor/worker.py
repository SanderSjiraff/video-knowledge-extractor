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
import os
import re
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
from .jobs import claim_next_job, fail_job, reclaim_orphaned_jobs, update_job
from .models import Video, VideoStatus

console = Console()

HEARTBEAT_INTERVAL = 30.0
# A single job running longer than this is considered stuck (a legitimate
# job — even a long video — finishes well inside this).  The watchdog fails
# the job and exits the worker so the supervisor can restart it.
MAX_JOB_SECONDS = 45 * 60


class _CurrentJob:
    """What the main worker thread is doing right now (watchdog input)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job_id: Optional[str] = None
        self._label = ""
        self._started_at = 0.0

    def start(self, job_id: str, label: str) -> None:
        with self._lock:
            self._job_id, self._label, self._started_at = job_id, label, time.monotonic()

    def finish(self) -> None:
        with self._lock:
            self._job_id, self._label, self._started_at = None, "", 0.0

    def snapshot(self) -> tuple[Optional[str], str, float]:
        with self._lock:
            return self._job_id, self._label, self._started_at


def _heartbeat_path(data_dir: Path) -> Path:
    return data_dir / "run" / "worker.heartbeat"


def _start_heartbeat(data_dir: Path, interval: float = HEARTBEAT_INTERVAL) -> threading.Thread:
    """Start a daemon thread that touches a heartbeat file while the worker lives.

    The supervisor reads the file's mtime and restarts the worker when the
    heartbeat goes stale — that catches a worker that is still alive but hung
    (for example inside a stuck network call or a wedged GPU op).
    """

    def _beat() -> None:
        path = _heartbeat_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                path.write_text(f"{time.time():.0f}\n", encoding="utf-8")
            except OSError:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=_beat, name="worker-heartbeat", daemon=True)
    thread.start()
    return thread


def _start_job_watchdog(
    data_dir: Path,
    current: _CurrentJob,
    max_job_seconds: float = MAX_JOB_SECONDS,
    check_interval: float = 60.0,
) -> threading.Thread:
    """Restart the worker when a single job runs longer than *max_job_seconds*.

    A stuck job (a wedged network call or GPU op) blocks the main loop forever
    while the heartbeat thread keeps beating, so the supervisor cannot see
    it.  Failing the job and exiting immediately makes the supervisor
    restart the worker, whose startup reclaim re-queues the job.
    """

    def _watch() -> None:
        while True:
            time.sleep(check_interval)
            job_id, label, started = current.snapshot()
            if job_id is None or started <= 0:
                continue
            age = time.monotonic() - started
            if age <= max_job_seconds:
                continue
            console.print(
                f"[red]Job {job_id[:8]} ({label}) has run {age / 60:.0f} min - "
                f"watchdog gives up: failing the job and restarting the worker.[/red]"
            )
            try:
                fail_job(
                    job_id,
                    f"Jobben tok for lang tid (over {max_job_seconds / 60:.0f} min) - worker ble restartet",
                    data_dir=data_dir,
                )
            except Exception:  # noqa: BLE001 - the exit below is what matters
                pass
            # Hard exit: the wedged main thread cannot be unblocked.  The
            # supervisor restarts us and the startup reclaim re-queues.
            os._exit(1)

    thread = threading.Thread(target=_watch, name="worker-job-watchdog", daemon=True)
    thread.start()
    return thread


class JobAborted(Exception):
    """Raised to abort a job with a user-readable reason."""


def resolve_video(url: str, platform: str) -> Video:
    """Turn a submitted URL (or local file path) into a Video record.

    YouTube ids come straight from the URL (metadata is enriched
    best-effort).  GitHub repos get a stable ``gh-owner-repo`` id straight
    from the URL.  TikTok/Instagram need one yt-dlp metadata call to get a
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

    if platform == "github":
        from .discovery import parse_github_repo

        parsed = parse_github_repo(url)
        if parsed is None:
            raise JobAborted("Ikke en gyldig GitHub-repolenke (bruk f.eks. https://github.com/eier/repo)")
        owner, repo = parsed
        # Stable, case-insensitive id so the same repo is never duplicated.
        return Video(
            video_id=f"gh-{owner.lower()}-{repo.lower()}",
            title=f"{owner}/{repo}",
            url=f"https://github.com/{owner}/{repo}",
            duration_seconds=0,
            channel="GitHub",
            playlist_order=0,
            status=VideoStatus.DISCOVERED,
            platform="github",
        )

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
            # TikTok photo carousel: no audio, but the slides carry the
            # content — scrape the item data and treat it as a slideshow.
            from .slides import SlideshowError, fetch_slideshow

            try:
                info = fetch_slideshow(final_url)
            except SlideshowError as exc:
                raise JobAborted(f"Bildeinnlegg (karusell), men klarte ikke hente bildene: {exc}")
            m = re.search(r"/photo/(\d+)", final_url)
            photo_id = (m.group(1) if m else None) or info["item_id"] or "unknown"
            return Video(
                video_id=f"photo-{photo_id}",
                title=info["desc"][:200] or "Bildeinnlegg (karusell)",
                url=final_url,
                duration_seconds=0,
                channel=info["author"] or "Unknown",
                playlist_order=0,
                status=VideoStatus.DISCOVERED,
                platform=platform,
            )
        if platform == "tiktok" and "/video/" in final_url:
            # yt-dlp's TikTok extractor is flaky (bot-check responses), but
            # the web page still carries the full item data — scrape it.
            from .slides import SlideshowError, scrape_tiktok_video

            try:
                info = scrape_tiktok_video(final_url)
            except SlideshowError as exc:
                raise JobAborted(f"Kunne ikke hente metadata fra TikTok: {exc}")
            if not info["id"]:
                raise JobAborted("Kunne ikke hente video-ID fra TikTok")
            return Video(
                video_id=info["id"],
                title=info["desc"][:200] or "Untitled",
                url=url,
                duration_seconds=info["duration"],
                channel=info["author"] or "Unknown",
                playlist_order=0,
                status=VideoStatus.DISCOVERED,
                platform="tiktok",
            )
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
    """Fetch a transcript: subtitles first for YouTube, repo text for
    GitHub, Whisper elsewhere."""
    if video.platform == "github":
        from .github_sources import GithubError, process_github_repo

        try:
            process_github_repo(video, data_dir=data_dir)
        except GithubError as exc:
            raise JobAborted(str(exc)) from exc
        return
    if video.platform == "youtube":
        from .transcripts import process_transcripts

        process_transcripts([video], force=False)
        if get_transcript(video.video_id, data_dir=data_dir) is not None:
            return
    # TikTok/Instagram/local, or YouTube without subtitles: local Whisper.
    from .whisper_transcripts import process_whisper_transcripts

    process_whisper_transcripts([video], force=False, keep_audio=True)


def _stage_thumbnail(video: Video, data_dir: Path) -> None:
    """Grab one dashboard thumbnail frame from a local video file, if any.

    No-op when only audio was downloaded (e.g. YouTube m4a) or the post is a
    photo slideshow.  Best-effort: never raises.
    """
    if video.video_id.startswith("photo-"):
        return
    from .visual import extract_thumbnail

    video_exts = {".mp4", ".mov", ".webm", ".mkv"}
    candidates: list[Path] = []
    for d in (Config.VIDEO_DIR, Config.AUDIO_DIR):
        candidates.extend(
            p for p in d.glob(f"{video.video_id}.*") if p.suffix.lower() in video_exts
        )
    local = Path(video.url)
    if local.suffix.lower() in video_exts and local.exists():
        candidates.append(local)
    for p in candidates:
        if extract_thumbnail(p, Config.THUMBS_DIR, video.video_id) is not None:
            console.print(f"  [dim]thumbnail saved for {video.video_id}[/dim]")
            return


def process_job(job: dict, data_dir: Optional[Path] = None) -> None:
    """Run the full pipeline for one claimed job. Updates the job as it goes."""
    data_dir = data_dir or Config.DATA_DIR
    opts = job["options"] or {}
    job_id = job["job_id"]
    note = ""

    # --- Stage 1: discover -------------------------------------------------
    video = resolve_video(job["url"], job["platform"])
    is_slideshow = video.video_id.startswith("photo-")
    save_video(video, data_dir=data_dir)
    # Keep the submitter-provided title (e.g. the original upload file name);
    # remote URLs get their title from metadata resolution.
    update_job(
        job_id,
        video_id=video.video_id,
        title=job.get("title") or video.title,
        kind="slides" if is_slideshow else "video",
        status="transcribing" if (
            opts.get("transcribe") or (is_slideshow and opts.get("analyze"))
        ) else (
            "analyzing" if opts.get("analyze") else "done"
        ),
        progress=15,
        data_dir=data_dir,
    )
    if not opts.get("transcribe") and not opts.get("analyze"):
        update_job(job_id, status="done", progress=100, note=note or "Ingen trinn valgt", data_dir=data_dir)
        return

    # --- Stage 2: transcript (slides for photo posts) ----------------------
    if is_slideshow and (opts.get("transcribe") or opts.get("analyze")):
        update_job(job_id, status="transcribing", progress=15, data_dir=data_dir)
        from .slides import process_slideshow

        # Slideshows: OCR + vision are the whole point (no audio), and the
        # vision endpoint is free (NTNU), so vision always runs for photos.
        transcript, slide_note = process_slideshow(
            video, job["url"], use_vision=True, data_dir=data_dir
        )
        if slide_note:
            note = (note + "; " if note else "") + slide_note
        if not transcript.segments:
            update_job(
                job_id,
                status="done",
                progress=100,
                note=(note + "; " if note else "") + "Fant ingen tekst i bildene – kan ikke analysere",
                data_dir=data_dir,
            )
            console.print(f"  [yellow]{video.video_id}: no slide text, job finished[/yellow]")
            return
    elif opts.get("transcribe"):
        update_job(job_id, status="transcribing", progress=15, data_dir=data_dir)
        _stage_transcribe(video, data_dir)
        transcript = get_transcript(video.video_id, data_dir=data_dir)
        if transcript is None or not transcript.segments:
            # No usable speech (Whisper saves an empty transcript for
            # music-only videos).  The content may live on screen (text
            # overlays, slides, screenshots) — capture it, same dual channel
            # as the slideshow path, so the analysis can still run.
            from .screen_fallback import screen_fallback_transcript

            transcript, fb_note = screen_fallback_transcript(video, data_dir=data_dir)
            if fb_note:
                note = (note + "; " if note else "") + fb_note
            if transcript is None:
                stored = get_video(video.video_id, data_dir=data_dir)
                detail = (stored.error_message if stored else "") or "Whisper fant ingen tale"
                update_job(
                    job_id,
                    status="done",
                    progress=100,
                    note=f"Ingen transkript: {detail}; {fb_note}",
                    data_dir=data_dir,
                )
                console.print(f"  [yellow]{video.video_id}: no transcript, job finished[/yellow]")
                return

    # --- Stage 2b: thumbnail (best-effort, cosmetic only) -------------------
    try:
        _stage_thumbnail(video, data_dir)
    except Exception:  # noqa: BLE001 - a missing frame must not kill the job
        pass

    # --- Stage 3: on-screen text (OCR) -- optional, non-fatal ---------------
    # Slideshows already capture on-screen text on the slides stage above.
    if not is_slideshow and opts.get("ocr"):
        update_job(job_id, status="ocr", progress=45, data_dir=data_dir)
        try:
            from .visual import process_on_screen_text

            result = process_on_screen_text([video], force=False, keep_media=False)
            if result.get("failed"):
                note = f"OCR feilet for {result['failed']} video(er)"
        except Exception as exc:  # noqa: BLE001 - OCR is opt-in and best-effort
            note = f"OCR tilgjengelig: {exc}"

    # --- Stage 4: vision frame descriptions -- optional, non-fatal ----------
    # Slideshows already run vision on the slides stage above.
    if not is_slideshow and opts.get("vision"):
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

    # --- Stage 7: refresh Markdown export (Obsidian) -- best-effort ----------
    # Only for analyzed jobs; export is idempotent (deterministic file names)
    # so it rewrites the full note set plus INDEKS.md / KUNNSKAP.md.
    if opts.get("analyze"):
        try:
            from .notes import export_markdown

            written = export_markdown(data_dir=data_dir)
            console.print(f"  [dim]Markdown-eksport oppdatert ({len(written)} filer)[/dim]")
        except Exception as exc:  # noqa: BLE001 - export is best-effort
            note = (note + "; " if note else "") + f"Markdown-eksport feilet: {exc}"

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
    current = _CurrentJob()
    _start_heartbeat(data_dir)
    _start_job_watchdog(data_dir, current)
    reclaimed = reclaim_orphaned_jobs(data_dir)
    if reclaimed:
        console.print(
            f"[yellow]Set {len(reclaimed)} fastlåst(e) jobb(er) tilbake i kø "
            f"(eier-workeren var ikke lenger i live).[/yellow]"
        )
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
        current.start(job["job_id"], job["platform"])
        try:
            process_job(job, data_dir=data_dir)
        except JobAborted as exc:
            fail_job(job["job_id"], str(exc), data_dir=data_dir)
            console.print(f"  [red]Job aborted:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001 - a job must never kill the worker
            fail_job(job["job_id"], str(exc), data_dir=data_dir)
            console.print(f"  [red]Job failed:[/red] {exc}")
        finally:
            current.finish()
