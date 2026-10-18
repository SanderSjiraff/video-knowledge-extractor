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
    # 2. <repo>/.venv/Lib/site-packages/nvidia (this project's layout)
    candidates.append(Path(__file__).resolve().parents[3] / ".venv" / "Lib" / "site-packages" / "nvidia")

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
    """Download best-audio for a video with yt-dlp. Returns the local path."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    preferred = audio_dir / f"{video.video_id}.m4a"
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred
    existing = list(audio_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]

    import yt_dlp

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(preferred),
        "quiet": True,
        "noprogress": True,
    }
    if Config.COOKIES_FILE:
        ydl_opts["cookiefile"] = Config.COOKIES_FILE
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video.url])

    if preferred.exists():
        return preferred
    existing = list(audio_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]
    raise FileNotFoundError(f"No audio file produced for {video.video_id}")


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
