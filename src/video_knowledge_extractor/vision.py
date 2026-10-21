"""Vision-model descriptions of on-screen visuals (code, diagrams, UI).

This complements the OCR channel. Instead of only reading text, it samples a
few representative frames and asks a multimodal model to describe what is on
screen — code layout, diagrams, charts, UI. That captures the *meaning* of
visuals OCR cannot parse, but it costs vision tokens, so it is:

- opt-in (run the 'describe-frames' command explicitly), and
- capped at ``VISION_MAX_FRAMES`` evenly-spaced frames per video.

Providers
---------
``ollama`` (default): a local vision model (``llava``, ``qwen2.5vl``,
``llama3.2-vision`` ...). Free and keeps everything on your machine.

``api``: an OpenAI-compatible Chat Completions endpoint that accepts image
content parts (``image_url``). Set ``VISION_API_URL`` (or a vision-capable
``LLM_API_URL``) and ``VISION_MODEL``.

Heavy network calls are the only external dependency (``requests``); frame
extraction reuses :mod:`visual` (ffmpeg).
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import requests
from rich.console import Console

from .config import Config
from .models import Video
from .visual import _download_video, extract_frames, ffmpeg_available


console = Console()

DEFAULT_PROMPT = (
    "Du ser et bilde (ett frame) fra en video. Beskriv kort og konkret hva som "
    "vises på skjermen: kode (inkluder viktige linjer/kommandoer), diagrammer, "
    "tall, UI-tekst eller layout. Svar kun med beskrivelsen, uten innledninger."
)


def vision_configured() -> bool:
    """Whether a vision model is available for the configured provider."""
    provider = (Config.VISION_PROVIDER or "ollama").lower()
    if provider == "ollama":
        return bool(Config.OLLAMA_URL)
    return bool(Config.VISION_API_URL or Config.LLM_API_URL) and bool(Config.VISION_MODEL)


def _b64_frame(frame_path: Path) -> str:
    return base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")


def _image_mime(frame_path: Path) -> str:
    """MIME type from the file extension (frames are usually PNG, slides JPEG)."""
    return "image/jpeg" if Path(frame_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"


def _vision_ollama(frame_path: Path, prompt: str) -> str:
    if not Config.OLLAMA_URL:
        raise RuntimeError(
            "VISION_PROVIDER=ollama needs OLLAMA_URL "
            "(e.g. http://localhost:11434/api/generate)."
        )
    payload = {
        "model": Config.VISION_MODEL or "llava",
        "prompt": prompt,
        "images": [_b64_frame(frame_path)],
        "stream": False,
    }
    resp = requests.post(Config.OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def _vision_api(frame_path: Path, prompt: str) -> str:
    url = Config.VISION_API_URL or Config.LLM_API_URL
    if not url:
        raise RuntimeError(
            "VISION_PROVIDER=api needs VISION_API_URL (or a vision-capable LLM_API_URL)."
        )
    headers = {"Content-Type": "application/json"}
    if Config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {Config.LLM_API_KEY}"
    payload = {
        "model": Config.VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{_image_mime(frame_path)};base64,{_b64_frame(frame_path)}"},
                    },
                ],
            }
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return json.dumps(data, ensure_ascii=False)


def describe_frame(frame_path: Path, prompt: str = DEFAULT_PROMPT) -> str:
    """Ask the configured vision model to describe a single frame."""
    provider = (Config.VISION_PROVIDER or "ollama").lower()
    if provider == "ollama":
        return _vision_ollama(frame_path, prompt)
    if provider == "api":
        return _vision_api(frame_path, prompt)
    raise ValueError(
        f"Unknown VISION_PROVIDER {Config.VISION_PROVIDER!r} (use 'ollama' or 'api')"
    )


def describe_frames(video: Video, max_frames: int | None = None, keep_media: bool = False) -> list[dict]:
    """Sample up to ``max_frames`` evenly-spaced frames and describe each.

    Returns a list of ``{"time_seconds": float, "description": str}`` in time
    order. Frames the model cannot describe are skipped.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to describe on-screen visuals.")
    if not vision_configured():
        raise RuntimeError(
            "No vision model configured. Set VISION_MODEL (and VISION_PROVIDER "
            "to 'ollama' or 'api')."
        )
    cap = int(max_frames if max_frames is not None else Config.VISION_MAX_FRAMES)
    duration = video.duration_seconds or 0
    interval = max(1.0, duration / cap) if duration > 0 else Config.VISUAL_FRAME_INTERVAL

    video_path = _download_video(video, Config.VIDEO_DIR)
    out: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="yke_vision_") as tmp:
        frames = extract_frames(video_path, interval, Path(tmp) / "frames", max_frames=cap)
        for t, fp in frames:
            try:
                desc = describe_frame(fp).strip()
            except Exception:  # noqa: BLE001 - one bad frame should not abort the video
                desc = ""
            if desc:
                out.append({"time_seconds": t, "description": desc})
    if not keep_media and video_path.exists():
        video_path.unlink(missing_ok=True)
    return out


def process_frame_descriptions(videos: list[Video], force: bool = False, keep_media: bool = False) -> dict[str, int]:
    """Describe on-screen visuals for a list of videos. Returns success/failed."""
    from .db import get_frame_descriptions, save_frame_descriptions

    if not vision_configured():
        console.print(
            "[red]No vision model configured — set VISION_MODEL and VISION_PROVIDER.[/red]"
        )
        return {"success": 0, "failed": len(videos)}
    if not ffmpeg_available():
        console.print("[red]ffmpeg not found — install it to describe on-screen visuals.[/red]")
        return {"success": 0, "failed": len(videos)}

    total = len(videos)
    success = failed = 0
    for i, video in enumerate(videos, 1):
        if video.status is None or str(video.status) == "inaccessible":
            continue
        if not force and get_frame_descriptions(video.video_id, data_dir=Config.DATA_DIR):
            success += 1
            console.print(f"  [{i}/{total}] [green]cached[/green] {video.video_id}")
            continue
        console.print(f"  [{i}/{total}] {video.video_id} — {video.title}")
        try:
            items = describe_frames(video, keep_media=keep_media)
            save_frame_descriptions(video.video_id, items, data_dir=Config.DATA_DIR)
            success += 1
            console.print(f"      [green]OK[/green] {len(items)} frame description(s)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            console.print(f"      [red]FAILED[/red]: {exc}")
    return {"success": success, "failed": failed}
