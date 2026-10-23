"""CLI for Video Knowledge Extractor."""

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

from video_knowledge_extractor.config import Config
from video_knowledge_extractor.db import init_db
from video_knowledge_extractor.notes import export_markdown
from video_knowledge_extractor.orchestrator import (
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
    """Video Knowledge Extractor — analyze playlists into searchable knowledge."""
    init_db(data_dir=Config.DATA_DIR)
    Config.ensure_dirs()


@cli.command()
@click.argument("source")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max videos to discover")
def discover(source: str, limit: int | None) -> None:
    """Discover videos from a playlist URL or text file."""
    run_discovery(source, limit=limit)


@cli.command("sync-playlist")
@click.argument("playlist_url")
@click.option("--limit", type=click.IntRange(min=1), default=None,
              help="Only consider the first N videos of the playlist")
@click.option("--dry-run", is_flag=True,
              help="Show what would be queued without touching the library")
@click.option("--retry-failed", is_flag=True,
              help="Also re-queue known videos whose earlier processing failed")
def sync_playlist(playlist_url: str, limit: int | None, dry_run: bool, retry_failed: bool) -> None:
    """Synchronize a YouTube playlist with the library.

    Fetches the playlist and queues every video that is not in the library
    yet (and not already in the queue). The worker then runs the full
    pipeline on each queued job. Videos already analyzed are never queued
    again, so re-running this is safe — e.g. from a Windows scheduled task:

        python src/cli.py sync-playlist "https://www.youtube.com/playlist?list=..."
    """
    from rich.table import Table

    from video_knowledge_extractor.playlist_sync import sync_playlist as _sync_playlist

    try:
        result = _sync_playlist(
            playlist_url,
            data_dir=Config.DATA_DIR,
            limit=limit,
            dry_run=dry_run,
            retry_failed=retry_failed,
        )
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title=f"Playlist sync — {len(result.fetched)} video(er) i spillelisten")
    table.add_column("Handling", style="cyan", no_wrap=True)
    table.add_column("Video")
    for v in result.new:
        action = "[green]legges i kø[/green]" if not dry_run else "[yellow]ville legge i kø[/yellow]"
        table.add_row(action, f"{v.title or v.video_id} ({v.duration_formatted})")
    for v in result.retried:
        action = "[magenta]prøves på nytt[/magenta]" if not dry_run else "[yellow]ville prøve på nytt[/yellow]"
        table.add_row(action, f"{v.title or v.video_id} ({v.duration_formatted})")
    for v in result.in_queue:
        table.add_row("[dim]allerede i kø[/dim]", v.title or v.video_id)
    if len(result.new) or len(result.retried) or len(result.in_queue):
        console.print(table)
    else:
        console.print("[cyan]Ingen nye videoer i spillelisten.[/cyan]")

    if not result.fetched:
        console.print("[yellow]Spillelisten ga ingen videoer (privat/slettet?).[/yellow]")
        return

    verb = "Ville kø" if dry_run else "Lagt i kø"
    console.print(
        f"[green]{verb} {len(result.new)} ny(e) video(er)[/green] "
        f"(+ {len(result.retried)} forsøk på nytt), "
        f"{len(result.in_queue)} allerede i kø, {len(result.known)} i biblioteket."
    )
    if not dry_run and result.jobs:
        console.print("[cyan]Workeren behandler de innkøede videoene automatisk.[/cyan]")


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
    from video_knowledge_extractor.db import get_all_videos
    from video_knowledge_extractor.visual import process_on_screen_text

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


@cli.command("describe-frames")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max videos to process")
@click.option("--force", is_flag=True, help="Re-describe even if descriptions already exist")
@click.option("--max-frames", type=click.IntRange(min=1), default=None,
              help="Max frames to describe per video (default from VISION_MAX_FRAMES)")
@click.option("--keep-media", is_flag=True, help="Keep the downloaded video after capture")
def describe_frames(limit: int | None, force: bool, max_frames: int | None, keep_media: bool) -> None:
    """Describe on-screen visuals with a vision model (code, diagrams, UI).

    Samples a few evenly-spaced frames from each video and asks a multimodal
    model to describe what is on screen. Complements `screen-text` (OCR reads
    text; this understands layout/diagrams). Requires ffmpeg plus a vision
    model (local Ollama via VISION_MODEL, or a vision API). See README.
    """
    from video_knowledge_extractor.db import get_all_videos
    from video_knowledge_extractor.vision import process_frame_descriptions

    if max_frames is not None:
        Config.VISION_MAX_FRAMES = int(max_frames)

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
        f"[cyan]Describing on-screen visuals for {len(videos)} video(s) "
        f"(provider: {Config.VISION_PROVIDER}, model: {Config.VISION_MODEL})...[/cyan]"
    )
    result = process_frame_descriptions(videos, force=force, keep_media=keep_media)
    console.print(
        f"[green]Frame descriptions done:[/green] {result.get('success', 0)} captured, "
        f"{result.get('failed', 0)} failed."
    )


@cli.command()
@click.option("--force", is_flag=True, help="Re-embed even if embeddings already exist")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max videos to process")
def embed(force: bool, limit: int | None) -> None:
    """Build semantic-search embeddings for transcribed videos.

    Embeds chunks of each transcript with a small embedding model so you can
    later do meaning-based retrieval (see 'search-semantic'). Uses a separate
    embedding model (local Ollama by default), so it does not use the analysis
    endpoint.
    """
    from video_knowledge_extractor.db import get_all_videos
    from video_knowledge_extractor.embeddings import process_embeddings

    videos = get_all_videos(data_dir=Config.DATA_DIR)
    videos = [v for v in videos if v.status is not None and str(v.status) != "inaccessible"]
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        console.print("[yellow]No videos to process. Run 'discover' first.[/yellow]")
        return
    console.print(
        f"[cyan]Building embeddings for {len(videos)} video(s) "
        f"(provider: {Config.EMBED_PROVIDER}, model: {Config.EMBED_MODEL})...[/cyan]"
    )
    result = process_embeddings(videos, force=force)
    console.print(
        f"[green]Embeddings done:[/green] {result.get('success', 0)} embedded, "
        f"{result.get('failed', 0)} failed."
    )


@cli.command("search-semantic")
@click.argument("query")
@click.option("--top", "top_k", type=click.IntRange(min=1), default=5, help="Number of results")
def search_semantic(query: str, top_k: int) -> None:
    """Semantic (embedding) search over transcript chunks by meaning."""
    from video_knowledge_extractor.db import get_all_videos
    from video_knowledge_extractor.embeddings import embeddings_configured, search_embeddings

    if not embeddings_configured():
        raise click.ClickException("No embedding model configured (set EMBED_MODEL / EMBED_PROVIDER).")
    results = search_embeddings(query, data_dir=Config.DATA_DIR, top_k=top_k)
    if not results:
        console.print("[yellow]No results — have you run 'embed' yet?[/yellow]")
        return
    video_by_id = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
    for r in results:
        v = video_by_id.get(r["video_id"])
        title = v.title if v else r["video_id"]
        snippet = r["snippet"]
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        console.print(f"[green]{r['score']:.3f}[/green] — [bold]{title}[/bold] (chunk {r['chunk_index']})")
        console.print(f"    {snippet}")


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
def worker() -> None:
    """Run the background job worker (processes the send queue).

    Keep this running (in its own terminal or via 'platform') whenever you
    want submitted videos to be processed.
    """
    from video_knowledge_extractor.worker import worker_loop

    worker_loop()


@cli.command()
def api() -> None:
    """Run the platform API server (FastAPI, http://127.0.0.1:8000)."""
    import uvicorn

    from video_knowledge_extractor.api import create_app

    console.print(f"[bold blue]API server on http://{Config.API_HOST}:{Config.API_PORT}[/bold blue]")
    uvicorn.run(create_app(), host=Config.API_HOST, port=Config.API_PORT, log_level="warning")


@cli.command()
def platform() -> None:
    """Launch the full platform: API server + worker + dashboard (one command)."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    console.print("[bold blue]Starting VKE platform (API + worker + dashboard)...[/bold blue]")
    console.print("[yellow]Ctrl+C stops everything.[/yellow]")
    children = [
        subprocess.Popen([sys.executable, "src/cli.py", "api"], cwd=root),
        subprocess.Popen([sys.executable, "src/cli.py", "worker"], cwd=root),
    ]
    try:
        dashboard_path = root / "src" / "video_knowledge_extractor" / "dashboard.py"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            cwd=root,
            check=False,
        )
    finally:
        for proc in children:
            proc.terminate()
        for proc in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


@cli.command()
def bot() -> None:
    """Run the Telegram send-channel bot (long polling, no public URL needed).

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_IDS in .env.
    """
    from video_knowledge_extractor.telegram_bot import run_bot

    run_bot()


@cli.command()
def supervisor() -> None:
    """Keep api + worker + bot + dashboard running (self-restarting).

    Normally started by launcher.pyw / start-platform.bat as a detached
    process, so no terminal windows are needed. Restarts any service that
    stops; service logs land in data/logs/.
    """
    from video_knowledge_extractor.supervisor import supervise

    supervise()


@cli.command()
def status() -> None:
    """Show current processing status."""
    print_status()


if __name__ == "__main__":
    cli()
