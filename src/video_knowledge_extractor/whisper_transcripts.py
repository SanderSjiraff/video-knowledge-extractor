"""Phase 2b: Local Whisper (GPU/CPU) transcript extraction.

This is a fallback path for when YouTube's subtitle API is IP-blocked
(HTTP 429 on the ``timedtext`` endpoint). Instead of fetching subtitles it
downloads the video's audio with yt-dlp and transcribes it locally with
``faster-whisper`` (CTranslate2, CUDA when available).

Results are written to the same ``transcripts`` table used by the subtitle
path (so the rest of the pipeline is unaffected) plus human-readable
``.txt`` and ``.srt`` files under ``data/transcripts/``.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

from .config import Config
from .db import get_transcript, save_transcript, update_video_status
from .models import Transcript, TranscriptSegment, Video, VideoStatus


console = Console()

# User-Agent for direct TikTok mp4 downloads (some CDNs reject the default
# python-requests UA).
_MP4_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _register_nvidia_dll_dirs() -> None:
    """Add nvidia pip-package DLL dirs to the search path.

    On machines without a system CUDA install, ctranslate2 cannot find
    cuBLAS/cuDNN/NVRTC. The ``nvidia-*`` pip packages ship the DLLs under
    ``site-packages/nvidia/<pkg>/bin``; registering those dirs lets the CUDA
    runtime load. No-op on non-Windows (where libs resolve normally).
    """
    if not hasattr(os, "add_dll_directory"):
        return

    import ctranslate2

    candidates = []
    # 1. site-packages/nvidia of the running interpreter (next to ctranslate2)
    try:
        candidates.append(Path(ctranslate2.__file__).resolve().parents[1] / "nvidia")
    except Exception:
        pass
    # 2. <repo>/.venv/Lib/site-packages/nvidia (this project's layout).
    #    The repo root is two levels up from this file; also try one level up
    #    so nested checkouts (repo inside a folder) still resolve.
    root = Path(__file__).resolve().parents[2]
    for base in (root, root.parent):
        candidates.append(base / ".venv" / "Lib" / "site-packages" / "nvidia")

    for base in candidates:
        if base and base.is_dir():
            for pkg in base.iterdir():
                bin_dir = pkg / "bin"
                if bin_dir.is_dir():
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except Exception:
                        pass


def _pick_device() -> tuple[str, str]:
    """Choose (device, compute_type). Prefer CUDA float16, fall back to CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


_model_cache: dict = {}


def _get_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        console.print(f"  [dim]Loading Whisper model '{model_size}' on {device}/{compute_type}...[/dim]")
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model_cache[key]


def _download_audio(video: Video, audio_dir: Path) -> Path:
    """Download best-audio for a video with yt-dlp. Returns the local path.

    For local files (platform 'local', e.g. uploads sent from the platform
    UI or the Telegram bot) there is nothing to download: faster-whisper
    (PyAV) can decode the file directly, so it is passed through as-is.
    """
    local = Path(video.url)
    if local.exists():
        return local
    audio_dir.mkdir(parents=True, exist_ok=True)
    # TikTok/Instagram: download the full (low-res) video instead of
    # audio-only.  Whisper decodes mp4 fine, and the same file later feeds
    # the thumbnail and the no-speech screen capture — one download, three
    # uses.  YouTube keeps the m4a-only path (ytimg provides thumbnails).
    full_video = video.platform in ("tiktok", "instagram")
    ext = "mp4" if full_video else "m4a"
    preferred = audio_dir / f"{video.video_id}.{ext}"
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred
    existing = list(audio_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]

    import yt_dlp

    ydl_opts = {
        "format": (
            "best[height<=720][ext=mp4]/best[ext=mp4]/best[height<=720]/best"
            if full_video
            else "bestaudio[ext=m4a]/bestaudio/best"
        ),
        "outtmpl": str(preferred),
        "quiet": True,
        "noprogress": True,
        # Hard limits so a flaky network can never hang the worker forever.
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
    }
    if Config.COOKIES_FILE:
        ydl_opts["cookiefile"] = Config.COOKIES_FILE
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video.url])
    except Exception:
        # yt-dlp's TikTok extractor is flaky (bot-check responses); fall
        # back to the direct mp4 address from the scraped web page.
        if video.platform == "tiktok":
            fallback = _download_tiktok_mp4(video, audio_dir)
            if fallback is not None:
                return fallback
        raise

    if preferred.exists():
        return preferred
    existing = list(audio_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]
    raise FileNotFoundError(f"No audio file produced for {video.video_id}")


def _download_tiktok_mp4(video: Video, audio_dir: Path) -> Optional[Path]:
    """Download the full mp4 straight from TikTok's web page data.

    Used when yt-dlp's TikTok extractor fails.  faster-whisper (PyAV)
    decodes mp4 directly, so no separate audio extraction is needed.
    Returns None when the page data has no playable address.
    """
    from .slides import SlideshowError, scrape_tiktok_video

    try:
        info = scrape_tiktok_video(video.url)
    except SlideshowError:
        return None
    play_addr = info.get("play_addr")
    if not play_addr:
        return None

    dest = audio_dir / f"{video.video_id}.mp4"
    try:
        # TikTok's CDN rejects plain python-requests (403); impersonating a
        # browser TLS fingerprint gets through.
        from curl_cffi import requests as cffi_requests

        resp = cffi_requests.get(play_addr, impersonate="chrome", timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except ImportError:
        import requests

        resp = requests.get(
            play_addr,
            headers={"User-Agent": _MP4_UA, "Referer": "https://www.tiktok.com/"},
            timeout=180,
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def _srt_time(x: float) -> str:
    h, rem = divmod(int(x), 3600)
    m, sec = divmod(rem, 60)
    ms = int(round((x - int(x)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def transcribe_video(
    video: Video,
    model_size: str,
    device: str,
    compute_type: str,
    force: bool = False,
    keep_audio: bool = True,
) -> Optional[Transcript]:
    """Download a video's audio and transcribe it locally with Whisper."""
    model = _get_model(model_size, device, compute_type)
    audio_path = _download_audio(video, Config.AUDIO_DIR)

    segments_iter, info = model.transcribe(str(audio_path), vad_filter=True)
    segments = [
        TranscriptSegment(start=seg.start, duration=seg.end - seg.start, text=seg.text.strip())
        for seg in segments_iter
    ]

    transcript = Transcript(
        video_id=video.video_id,
        language=info.language,
        is_generated=True,
        segments=segments,
        fetched_at=datetime.utcnow(),
    )

    # Persist to the repo DB (same table the subtitle path uses).
    save_transcript(transcript, data_dir=Config.DATA_DIR)
    update_video_status(video.video_id, VideoStatus.TRANSCRIPT_OK, data_dir=Config.DATA_DIR)

    # Human-readable outputs.
    Config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    text_path = Config.TRANSCRIPTS_DIR / f"{video.video_id}.txt"
    srt_path = Config.TRANSCRIPTS_DIR / f"{video.video_id}.srt"
    text_path.write_text(transcript.full_text, encoding="utf-8")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{seg.text}\n\n")

    if not keep_audio and audio_path.exists():
        audio_path.unlink(missing_ok=True)

    return transcript


def process_whisper_transcripts(
    videos: list[Video],
    model_size: str = "medium",
    force: bool = False,
    keep_audio: bool = True,
) -> dict[str, int]:
    """Transcribe a list of videos locally with Whisper. Returns success/failed counts."""
    _register_nvidia_dll_dirs()
    device, compute_type = _pick_device()
    console.print(f"  Whisper model: [cyan]{model_size}[/cyan] on [cyan]{device}/{compute_type}[/cyan]")
    _get_model(model_size, device, compute_type)  # warm up / download once

    total = len(videos)
    success = failed = 0
    for i, video in enumerate(videos, 1):
        if video.status == VideoStatus.INACCESSIBLE:
            continue
        if not force and get_transcript(video.video_id, data_dir=Config.DATA_DIR) is not None:
            success += 1
            console.print(f"  [{i}/{total}] [green]cached[/green] {video.video_id} — {video.title}")
            continue

        console.print(f"  [{i}/{total}] {video.video_id} — {video.title}")
        try:
            t = transcribe_video(
                video, model_size, device, compute_type,
                force=force, keep_audio=keep_audio,
            )
            if t is not None and t.segments:
                success += 1
                console.print(f"      [green]OK[/green] {t.word_count} words ({t.language})")
            else:
                failed += 1
                update_video_status(
                    video.video_id, VideoStatus.TRANSCRIPT_UNAVAILABLE,
                    error_message="Whisper produced no segments",
                    data_dir=Config.DATA_DIR,
                )
                console.print(f"      [red]FAILED[/red] — no speech detected")
        except Exception as e:  # noqa: BLE001
            failed += 1
            update_video_status(
                video.video_id, VideoStatus.TRANSCRIPT_UNAVAILABLE,
                error_message=str(e)[:300], data_dir=Config.DATA_DIR,
            )
            console.print(f"      [red]FAILED[/red]: {e}")
        time.sleep(0.3)  # be gentle with network/IO

    return {"success": success, "failed": failed}
