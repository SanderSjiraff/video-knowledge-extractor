"""CLI for YouTube Knowledge Extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

# Ensure src directory is on path so package imports work
sys.path.insert(0, str(Path(__file__).parent))

from youtube_knowledge_extractor.config import Config
from youtube_knowledge_extractor.db import init_db
from youtube_knowledge_extractor.orchestrator import (
    run_analysis,
    run_cost_estimate,
    run_dashboard,
    run_discovery,
    run_transcripts,
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
def transcripts(force: bool) -> None:
    """Extract transcripts for discovered videos."""
    run_transcripts(force=force)


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
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    run_dashboard()


@cli.command()
def status() -> None:
    """Show current processing status."""
    print_status()


if __name__ == "__main__":
    cli()
