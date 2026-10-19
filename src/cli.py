"""CLI for YouTube Knowledge Extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

# Ensure src directory is on path so package imports work
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 I/O so emoji in video titles don't crash the Windows console
# (which defaults to cp1252). Harmless on other platforms.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from youtube_knowledge_extractor.config import Config
from youtube_knowledge_extractor.db import init_db
from youtube_knowledge_extractor.notes import export_markdown
from youtube_knowledge_extractor.orchestrator import (
    run_analysis,
    run_cost_estimate,
    run_dashboard,
    run_discovery,
    run_synthesis,
    run_transcripts,
    run_whisper_transcripts,
    print_status,
)


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """YouTube Knowledge Extractor — analyze playlists into searchable knowledge."""
    init_db(data_dir=Config.DATA_DIR)
    Config.ensure_dirs()


@cli.command()
@click.argument("source")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max videos to discover")
def discover(source: str, limit: int | None) -> None:
    """Discover videos from a playlist URL or text file."""
    run_discovery(source, limit=limit)


@cli.command()
@click.option("--force", is_flag=True, help="Re-fetch all transcripts")
@click.option("--cookies", type=click.Path(exists=True, dir_okay=False, readable=True),
              envvar="YOUTUBE_COOKIES_FILE", help="Netscape cookies file for yt-dlp")
def transcripts(force: bool, cookies: str | None) -> None:
    """Extract transcripts for discovered videos."""
    Config.COOKIES_FILE = cookies
    run_transcripts(force=force)


@cli.command()
@click.argument("source", required=False)
@click.option("--model", default="medium",
              help="Whisper model size: tiny, base, small, medium, large-v3 (default: medium)")
@click.option("--force", is_flag=True, help="Re-transcribe even if a transcript exists")
@click.option("--no-keep-audio", is_flag=True, help="Delete downloaded audio after transcribing")
def transcribe(source: str | None, model: str, force: bool, no_keep_audio: bool) -> None:
    """Transcribe videos locally with Whisper (audio download + GPU).

    Use this when YouTube's subtitle API is blocked. Works on videos already
    discovered in the DB, or discover first from SOURCE (playlist URL or a
    text file with one URL per line).
    """
    videos = None
    if source:
        videos = run_discovery(source)
    run_whisper_transcripts(
        videos=videos,
        model_size=model,
        force=force,
        keep_audio=not no_keep_audio,
    )


@cli.command("screen-text")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max videos to process")
@click.option("--force", is_flag=True, help="Re-capture even if on-screen text already exists")
@click.option("--frame-interval", type=float, default=None,
              help="Seconds between extracted frames (default from VISUAL_FRAME_INTERVAL)")
@click.option("--keep-media", is_flag=True, help="Keep the downloaded video after capture")
def screen_text(limit: int | None, force: bool, frame_interval: float | None, keep_media: bool) -> None:
    """Capture on-screen text (OCR) for discovered videos.

    Downloads each video, extracts one frame every N seconds and OCRs the
    text so the analysis can see code, numbers, slides, etc. that never
    appear in the audio. Requires ffmpeg plus an OCR engine
    (``easyocr`` or ``paddleocr``); see OCR_ENGINE in the README.
    """
    from youtube_knowledge_extractor.db import get_all_videos
    from youtube_knowledge_extractor.visual import process_on_screen_text

    if frame_interval is not None:
        Config.VISUAL_FRAME_INTERVAL = float(frame_interval)

    videos = get_all_videos(data_dir=Config.DATA_DIR)
    videos = [
        v for v in videos
        if v.status is not None and str(v.status) != "inaccessible" and v.url
    ]
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        console.print("[yellow]No videos to process. Run 'discover' first.[/yellow]")
        return

    console.print(
        f"[cyan]Capturing on-screen text for {len(videos)} video(s) "
        f"(engine: {Config.OCR_ENGINE})...[/cyan]"
    )
    result = process_on_screen_text(videos, force=force, keep_media=keep_media)
    console.print(
        f"[green]On-screen text done:[/green] {result.get('success', 0)} captured, "
        f"{result.get('failed', 0)} failed."
    )


@cli.command()
def cost() -> None:
    """Estimate cost for AI analysis."""
    try:
        run_cost_estimate()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Maximum number of videos in this analysis batch")
@click.option("--force", is_flag=True, help="Re-analyze videos")
def analyze(limit: int | None, force: bool) -> None:
    """Run AI analysis on videos with transcripts.

    Use --limit to set the maximum number of videos in this batch.
    """
    if limit is None:
        limit = Config.DRY_RUN_LIMIT
        console.print(f"[yellow]Using the default batch limit of {limit} videos.[/yellow]")
        console.print("[yellow]Add --limit N to change the batch size.[/yellow]")

    try:
        run_analysis(dry_run_limit=limit, force=force)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--force", is_flag=True, help="Regenerate even if a cached synthesis exists")
def synthesize(force: bool) -> None:
    """Create a combined playlist synthesis (overlap, contradictions, unique knowledge)."""
    try:
        result = run_synthesis(force=force)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if result:
        console.print(f"[green]Saved synthesis for {len(result.get('video_ids', []))} videos.[/green]")


@cli.command()
def export() -> None:
    """Export notes as Markdown with source deep-links (Obsidian/OpenCode-friendly)."""
    files = export_markdown(data_dir=Config.DATA_DIR)
    out_dir = Config.EXPORTS_DIR / "markdown"
    console.print(f"[green]Exported {len(files)} Markdown files to {out_dir}[/green]")
    for f in files:
        console.print(f"  {f.name}")


@cli.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    run_dashboard()


@cli.command()
def status() -> None:
    """Show current processing status."""
    print_status()


if __name__ == "__main__":
    cli()
