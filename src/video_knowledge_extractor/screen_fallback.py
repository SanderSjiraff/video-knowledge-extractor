"""Fallback for videos with no speech: capture on-screen info as a transcript.

Music-only videos (TikTok/Reels) often carry their content as text overlays,
slides, or screenshots on screen instead of audio.  When Whisper detects no
speech, this module downloads the video, extracts frames, reads the on-screen
text with the vision model (and OCR when available — the same dual channel as
the slideshow path), and synthesizes a transcript so the normal
analysis/search/embedding pipeline runs unchanged.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

from .config import Config
from .models import Transcript, TranscriptSegment, Video

console = Console()

FRAME_PROMPT = (
    "Du ser et bilde (frame) fra en video uten tale. Les og gjengi all synlig "
    "tekst ordrett, og beskriv kort hva bildet ellers viser (diagrammer, "
    "skjermbilder, UI, tabeller, bilder). Svar kun med teksten/beskrivelsen, "
    "uten innledninger."
)

# Roughly one frame per 5 seconds, up to 12 frames per video.
_FALLBACK_INTERVAL = 5.0
_FALLBACK_MAX_FRAMES = 12


def _guess_language(text: str) -> str:
    return "no" if re.search(r"[æøå]", text.lower()) else "en"


def _ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False


def _get_video_file(video: Video) -> Optional[Path]:
    """Find or download a full video file (with frames) for the video."""
    # 1) Already on disk (video or full mp4 saved by the audio fallback)?
    for d in (Config.VIDEO_DIR, Config.AUDIO_DIR):
        for p in sorted(d.glob(f"{video.video_id}.*")):
            if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} and p.stat().st_size > 0:
                return p
    # 2) TikTok: direct mp4 from the web page data (no yt-dlp needed).
    if video.platform == "tiktok":
        from .whisper_transcripts import _download_tiktok_mp4

        return _download_tiktok_mp4(video, Config.VIDEO_DIR)
    # 3) Other platforms: yt-dlp best quality <=720p.
    from .visual import _download_video

    return _download_video(video, Config.VIDEO_DIR)


def screen_fallback_transcript(
    video: Video,
    data_dir: Optional[Path] = None,
) -> tuple[Optional[Transcript], str]:
    """Capture on-screen info for a video with no speech.

    Returns ``(transcript, note)``; ``transcript`` is None when nothing
    usable was found on screen.  The captured text is persisted to the
    screen_text and frame_descriptions tables (like the slideshow path)
    before the transcript is synthesized and saved.
    """
    from .db import save_frame_descriptions, save_screen_text, save_transcript
    from .visual import extract_frames, ffmpeg_available, ocr_frame
    from .vision import describe_frame, vision_configured

    data_dir = data_dir or Config.DATA_DIR
    have_vision = vision_configured()
    have_ocr = _ocr_available()
    if not ffmpeg_available():
        return None, "ingen ffmpeg — kan ikke fange skjerminnhold"
    if not have_vision and not have_ocr:
        return None, "verken visjonsmodell eller OCR tilgjengelig — kan ikke fange skjerminnhold"

    try:
        video_path = _get_video_file(video)
    except Exception as exc:  # noqa: BLE001 - a download hiccup must not kill the job
        return None, f"klarte ikke laste ned videoen for skjermfangst: {exc}"
    if video_path is None:
        return None, "klarte ikke laste ned videoen for skjermfangst"

    with tempfile.TemporaryDirectory(prefix="vke_screencap_") as tmp:
        frames = extract_frames(
            video_path, _FALLBACK_INTERVAL, Path(tmp) / "frames", max_frames=_FALLBACK_MAX_FRAMES
        )
        per_frame: list[tuple[float, str, str]] = []
        seen_ocr: set[str] = set()
        seen_desc: set[str] = set()
        for t, fp in frames:
            # OCR text (deduped across frames — static overlays repeat).
            ocr = ""
            if have_ocr:
                try:
                    ocr = (ocr_frame(str(fp)) or "").strip()
                except Exception:  # noqa: BLE001 - OCR is best-effort
                    ocr = ""
            key = " ".join(ocr.lower().split())
            if key and key in seen_ocr:
                ocr = ""
            if key:
                seen_ocr.add(key)
            # Vision description (free NTNU endpoint) — reads text verbatim.
            desc = ""
            if have_vision:
                try:
                    desc = (describe_frame(fp, prompt=FRAME_PROMPT) or "").strip()
                except Exception:  # noqa: BLE001 - one bad frame must not abort the rest
                    desc = ""
            dkey = " ".join(desc.lower().split())
            if dkey and dkey in seen_desc:
                desc = ""
            if dkey:
                seen_desc.add(dkey)
            if ocr or desc:
                per_frame.append((float(t), ocr, desc))

    if not per_frame:
        return None, "Fant ingen tekst eller innhold på skjermen"

    ocr_items = [
        {"time_seconds": t, "text": ocr} for t, ocr, _ in per_frame if ocr
    ]
    desc_items = [
        {"time_seconds": t, "description": d} for t, _, d in per_frame if d
    ]
    if ocr_items:
        save_screen_text(video.video_id, ocr_items, data_dir=data_dir)
    if desc_items:
        save_frame_descriptions(video.video_id, desc_items, data_dir=data_dir)

    segments: list[TranscriptSegment] = []
    all_text: list[str] = []
    for t, ocr, desc in per_frame:
        parts = []
        if ocr:
            parts.append(f"Tekst på skjermen: {ocr}")
        if desc:
            parts.append(f"Bildet viser: {desc}")
        segments.append(TranscriptSegment(start=t, duration=0.0, text=" ".join(parts)))
        all_text.extend(parts)

    transcript = Transcript(
        video_id=video.video_id,
        language=_guess_language(" ".join(all_text)),
        is_generated=True,
        segments=segments,
    )
    save_transcript(transcript, data_dir=data_dir)
    console.print(f"  [green]screen capture:[/green] {len(segments)} ramme(r) med innhold")
    return transcript, "ingen tale — innhold fanget fra skjermen"
