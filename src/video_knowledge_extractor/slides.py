"""TikTok slideshow (photo carousel) support.

Photo posts have no audio; the content lives in the slide images (text,
screenshots, diagrams).  TikTok's mobile web page embeds the full item data
in a ``__UNIVERSAL_DATA_FOR_REHYDRATION__`` JSON blob, which this module
scrapes, then downloads every slide and runs the configured OCR engine
(and, when enabled, the vision model) on each image.

The captured text is stored as a synthesized transcript so the normal
analysis/search/embedding pipeline runs unchanged, and the per-slide OCR
text / vision descriptions reuse the existing screen-text and
frame-description channels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console

from .config import Config
from .models import Transcript, TranscriptSegment, Video

console = Console()

_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)

# Video pages (unlike photo pages) only embed the full itemStruct when the
# request looks like a desktop browser.
_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

SLIDE_PROMPT = (
    "Du ser et bilde fra et TikTok-slideshow (bildeserie). Les og gjengi all "
    "synlig tekst ordrett, og beskriv kort hva bildet ellers viser (UI, "
    "diagram, skisse, layout, bilder). Svar kun med teksten/beskrivelsen, "
    "uten innledninger."
)


class SlideshowError(Exception):
    """A photo post could not be resolved or its images could not be fetched."""


def resolve_photo_url(url: str) -> str:
    """Follow redirects (vm.tiktok.com short links) and return the final URL."""
    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=10, headers={"User-Agent": _MOBILE_UA}
        )
        return str(resp.url)
    except Exception:  # noqa: BLE001 - fall back to the URL as given
        return url


def is_slideshow_url(url: str) -> bool:
    """Best-effort check whether a URL points at a TikTok photo post."""
    return "/photo/" in resolve_photo_url(url)


def parse_slideshow_html(html: str, final_url: str) -> dict[str, Any]:
    """Extract the photo post's item data from TikTok's mobile page HTML.

    Returns ``{"item_id", "desc", "author", "images": [url, ...]}``.
    """
    m = re.search(
        r'<script[^>]*id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
        html,
        re.S,
    )
    if not m:
        raise SlideshowError(
            "Fant ikke innebygd JSON i TikTok-siden (sidesstrukturen har endret seg?)"
        )
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise SlideshowError(f"Kunne ikke parse TikTok-JSON: {exc}") from exc

    scope = data.get("__DEFAULT_SCOPE__") or {}
    item: dict[str, Any] = {}
    for val in scope.values():
        if not isinstance(val, dict):
            continue
        candidate = (val.get("itemInfo") or {}).get("itemStruct")
        if isinstance(candidate, dict) and candidate:
            item = candidate
            break
    if not item:
        raise SlideshowError(
            "Fant ikke itemStruct i TikTok-siden (innlegget finnes kanskje ikke lenger)"
        )

    images: list[str] = []
    for img in (item.get("imagePost") or {}).get("images") or []:
        url_list = (img.get("imageURL") or {}).get("urlList") or []
        if url_list:
            images.append(url_list[0])
    if not images:
        raise SlideshowError("Fant ingen bilder i innlegget")

    m_id = re.search(r"/photo/(\d+)", final_url)
    return {
        "item_id": (m_id.group(1) if m_id else None) or str(item.get("id") or ""),
        "desc": item.get("desc") or "",
        "author": (item.get("author") or {}).get("uniqueId") or "",
        "images": images,
    }


def fetch_slideshow(url: str) -> dict[str, Any]:
    """Scrape a TikTok photo post (canonical URL or short link) for its item data."""
    final_url = resolve_photo_url(url)
    if "/photo/" not in final_url:
        raise SlideshowError("Lenken peker ikke på et TikTok-bildeinnlegg")
    resp = requests.get(
        final_url,
        headers={"User-Agent": _MOBILE_UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_slideshow_html(resp.text, final_url)


def parse_video_html(html: str) -> dict[str, Any]:
    """Extract a video post's item data from TikTok's web page HTML.

    Returns ``{"id", "desc", "author", "duration", "play_addr"}`` where
    ``play_addr`` is a direct mp4 URL from the page data.
    """
    m = re.search(
        r'<script[^>]*id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
        html,
        re.S,
    )
    if not m:
        raise SlideshowError(
            "Fant ikke innebygd JSON i TikTok-siden (sidesstrukturen har endret seg?)"
        )
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise SlideshowError(f"Kunne ikke parse TikTok-JSON: {exc}") from exc

    scope = data.get("__DEFAULT_SCOPE__") or {}
    item: dict[str, Any] = {}
    for val in scope.values():
        if not isinstance(val, dict):
            continue
        candidate = (val.get("itemInfo") or {}).get("itemStruct")
        if isinstance(candidate, dict) and candidate:
            item = candidate
            break
    if not item:
        raise SlideshowError(
            "Fant ikke itemStruct i TikTok-siden (innlegget finnes kanskje ikke lenger)"
        )

    video = item.get("video") or {}
    play_addr = str(video.get("playAddr") or "")
    if not play_addr:
        # Highest-quality variant from the bitrate list, if playAddr is empty.
        for br in reversed(video.get("bitrateInfo") or []):
            urls = (br.get("PlayAddr") or {}).get("UrlList") or []
            if urls:
                play_addr = str(urls[0])
                break
    try:
        duration = int(video.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    return {
        "id": str(item.get("id") or ""),
        "desc": item.get("desc") or "",
        "author": (item.get("author") or {}).get("uniqueId") or "",
        "duration": duration,
        "play_addr": play_addr,
    }


def scrape_tiktok_video(url: str) -> dict[str, Any]:
    """Scrape a TikTok video post (canonical URL or short link) for its item data.

    Fallback for when yt-dlp's TikTok extractor fails (bot-check responses):
    the web page still embeds the full item data, including a direct mp4
    play address that can be downloaded without yt-dlp.
    """
    final_url = resolve_photo_url(url)
    if "/video/" not in final_url:
        raise SlideshowError("Lenken peker ikke på et TikTok-videoinnlegg")
    resp = requests.get(
        final_url,
        headers={"User-Agent": _DESKTOP_UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_video_html(resp.text)


def download_slides(image_urls: list[str], slides_dir: Path) -> list[Path]:
    """Download every slide image into ``slides_dir``; returns paths in order."""
    slides_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, u in enumerate(image_urls, 1):
        dest = slides_dir / f"slide-{i:02d}.jpg"
        resp = requests.get(
            u,
            headers={"User-Agent": _MOBILE_UA, "Referer": "https://www.tiktok.com/"},
            timeout=60,
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        paths.append(dest)
    return paths


def _guess_language(text: str) -> str:
    return "no" if re.search(r"[æøå]", text.lower()) else "en"


def process_slideshow(
    video: Video,
    url: str,
    use_vision: bool = True,
    data_dir: Optional[Path] = None,
) -> tuple[Transcript, str]:
    """Run the slideshow pipeline for one photo post.

    Downloads the slide images to ``data/slides/<video_id>/``, OCRs every
    slide, optionally adds vision descriptions, stores both channels, and
    synthesizes a transcript from the captured text.  Returns
    ``(transcript, note)``; ``note`` is empty unless something was degraded.
    """
    from .db import save_frame_descriptions, save_screen_text, save_transcript
    from .visual import ocr_frame
    from .vision import describe_frame, vision_configured

    data_dir = data_dir or Config.DATA_DIR
    info = fetch_slideshow(url)
    slides_dir = data_dir / "slides" / video.video_id
    paths = download_slides(info["images"], slides_dir)

    ocr_texts: list[str] = []
    ocr_ok = True
    note = ""
    for p in paths:
        try:
            ocr_texts.append((ocr_frame(str(p)) or "").strip())
        except Exception as exc:  # noqa: BLE001 - OCR is best-effort
            if ocr_ok:
                note = f"OCR utilgjengelig: {exc}"
                ocr_ok = False
            ocr_texts.append("")

    descs: list[str] = [""] * len(paths)
    if use_vision:
        if vision_configured():
            for i, p in enumerate(paths):
                try:
                    descs[i] = (describe_frame(p, prompt=SLIDE_PROMPT) or "").strip()
                except Exception:  # noqa: BLE001 - one bad slide must not abort the rest
                    descs[i] = ""
        else:
            note = (note + "; " if note else "") + "visjonsmodell ikke konfigurert – kun OCR"

    screen_items = [
        {"time_seconds": float(i), "text": t} for i, t in enumerate(ocr_texts) if t
    ]
    if screen_items:
        save_screen_text(video.video_id, screen_items, data_dir=data_dir)
    desc_items = [
        {"time_seconds": float(i), "description": d} for i, d in enumerate(descs) if d
    ]
    if desc_items:
        save_frame_descriptions(video.video_id, desc_items, data_dir=data_dir)

    segments: list[TranscriptSegment] = []
    for i, (t, d) in enumerate(zip(ocr_texts, descs), 1):
        parts = []
        if t:
            parts.append(f"Slide {i}: {t}")
        if d:
            parts.append(f"Slide {i} (beskrivelse): {d}")
        if parts:
            segments.append(
                TranscriptSegment(start=float(i - 1), duration=0.0, text=" ".join(parts))
            )

    all_text = " ".join(t for t in ocr_texts if t)
    transcript = Transcript(
        video_id=video.video_id,
        language=_guess_language(all_text),
        is_generated=True,
        segments=segments,
    )
    save_transcript(transcript, data_dir=data_dir)

    console.print(
        f"  [green]slides:[/green] {len(paths)} bilder, "
        f"{len(screen_items)} OCR-blokker, {len(desc_items)} visjonsbeskrivelser"
    )
    return transcript, note
