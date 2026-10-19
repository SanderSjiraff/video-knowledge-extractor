"""On-screen text capture: download a video, extract frames, OCR the text.

This is an opt-in channel that complements the audio transcript. Many videos
(coding tutorials, slides, dashboards) carry important information on screen
that never appears in the audio. Capturing it as text lets the analysis model
see it.

Heavy dependencies (ffmpeg, an OCR engine) are imported lazily so the rest of
the pipeline works without them. Configure the engine via ``OCR_ENGINE``
(``easyocr`` or ``paddleocr``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from .config import Config
from .models import Video


console = Console()

_ocr_reader = None  # cached OCR engine (lazy)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ocr_engine_description() -> str:
    return f"{Config.OCR_ENGINE} (pip install {Config.OCR_ENGINE}{' paddlepaddle' if Config.OCR_ENGINE == 'paddleocr' else ''})"


def _download_video(video: Video, video_dir: Path) -> Path:
    """Download the video (best mp4) with yt-dlp. Returns the local path."""
    video_dir.mkdir(parents=True, exist_ok=True)
    preferred = video_dir / f"{video.video_id}.mp4"
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred
    existing = list(video_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]

    import yt_dlp

    ydl_opts = {
        # Many YouTube videos expose only separate video/audio streams, so
        # prefer merged bestvideo+bestaudio (ffmpeg merges them) and fall
        # back to a single progressive file when one exists. 720p is plenty
        # for OCR/vision and keeps downloads small across many videos.
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
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
    existing = list(video_dir.glob(f"{video.video_id}.*"))
    if existing:
        return existing[0]
    raise FileNotFoundError(f"No video file produced for {video.video_id}")


def extract_frames(video_path: Path, interval: float, frames_dir: Path, max_frames: int = 200) -> list[tuple[float, Path]]:
    """Extract one frame every ``interval`` seconds using ffmpeg.

    Returns a list of ``(approx_time_seconds, frame_path)`` in order.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg to enable on-screen text capture "
            "(e.g. 'winget install ffmpeg' or download from ffmpeg.org)."
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%04d.png")
    fps = 1.0 / max(0.5, float(interval))
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(int(max_frames)),
        "-y", pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[-500:]}")

    frames = sorted(frames_dir.glob("frame_*.png"))
    step = float(interval)
    return [(i * step, fp) for i, fp in enumerate(frames[: max(0, int(max_frames))])]


def _ocr_easyocr(frame_path: str) -> str:
    global _ocr_reader
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'easyocr' OCR engine is not installed. Run: pip install easyocr"
        ) from exc
    if _ocr_reader is None:
        import torch

        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    results = _ocr_reader.readtext(frame_path)
    return " ".join(r[1] for r in results if r[1] and r[1].strip())


def _ocr_paddleocr(frame_path: str) -> str:
    global _ocr_reader
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'paddleocr' OCR engine is not installed. Run: pip install paddleocr paddlepaddle"
        ) from exc
    if _ocr_reader is None:
        _ocr_reader = PaddleOCR(use_textline_orientation=False, lang="en")
    result = _ocr_reader.ocr(frame_path, cls=False)
    texts: list[str] = []
    for page in result or []:
        for line in page or []:
            try:
                text = line[1][0]
            except (IndexError, TypeError):
                continue
            if text and text.strip():
                texts.append(text.strip())
    return " ".join(texts)


def ocr_frame(frame_path: str) -> str:
    """Run the configured OCR engine on a single frame; return its text."""
    engine = (Config.OCR_ENGINE or "easyocr").lower()
    if engine == "easyocr":
        return _ocr_easyocr(frame_path)
    if engine == "paddleocr":
        return _ocr_paddleocr(frame_path)
    raise ValueError(f"Unknown OCR_ENGINE {Config.OCR_ENGINE!r} (use 'easyocr' or 'paddleocr')")


def capture_on_screen_text(
    video: Video,
    interval: Optional[float] = None,
    max_frames: Optional[int] = None,
    keep_media: bool = False,
) -> list[dict]:
    """Download a video, extract frames, OCR them, and return deduped text.

    Returns a list of ``{"time_seconds": float, "text": str}`` in time order.
    Repeated (static) on-screen text is captured once.
    """
    step = float(interval if interval is not None else Config.VISUAL_FRAME_INTERVAL)
    cap = int(max_frames if max_frames is not None else Config.VISUAL_MAX_FRAMES)

    video_path = _download_video(video, Config.VIDEO_DIR)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="yke_frames_") as tmp:
        frames = extract_frames(video_path, step, Path(tmp) / "frames", max_frames=cap)
        items: list[dict] = []
        seen: set[str] = set()
        for t, fp in frames:
            text = (ocr_frame(str(fp)) or "").strip()
            key = " ".join(text.lower().split())
            if not text or key in seen:
                continue
            seen.add(key)
            items.append({"time_seconds": t, "text": text})

    if not keep_media and video_path.exists():
        video_path.unlink(missing_ok=True)
    return items


def process_on_screen_text(videos: list[Video], force: bool = False, keep_media: bool = False) -> dict[str, int]:
    """Capture on-screen text for a list of videos. Returns success/failed counts."""
    from .db import get_screen_text, save_screen_text

    if not ffmpeg_available():
        console.print("[red]ffmpeg not found — install it to capture on-screen text.[/red]")
        return {"success": 0, "failed": len(videos)}

    total = len(videos)
    success = failed = 0
    for i, video in enumerate(videos, 1):
        if video.status is None or str(video.status) == "inaccessible":
            continue
        if not force and get_screen_text(video.video_id, data_dir=Config.DATA_DIR):
            success += 1
            console.print(f"  [{i}/{total}] [green]cached[/green] {video.video_id}")
            continue
        console.print(f"  [{i}/{total}] {video.video_id} — {video.title}")
        try:
            items = capture_on_screen_text(video, keep_media=keep_media)
            save_screen_text(video.video_id, items, data_dir=Config.DATA_DIR)
            success += 1
            console.print(f"      [green]OK[/green] {len(items)} unique on-screen text blocks")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            console.print(f"      [red]FAILED[/red]: {exc}")
    return {"success": success, "failed": failed}
