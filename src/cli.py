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
@click.option("--limit", type=int, default=None, help="Max videos to discover")
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
    run_cost_estimate()


@cli.command()
@click.option("--limit", type=int, default=None, help="Limit to N videos (dry-run)")
@click.option("--force", is_flag=True, help="Re-analyze videos")
def analyze(limit: int | None, force: bool) -> None:
    """Run AI analysis on videos with transcripts.

    Use --limit to run a small dry-run before full processing.
    """
    if limit is None:
        limit = Config.DRY_RUN_LIMIT
        console.print(f"[yellow]Using default dry-run limit of {limit} videos.[/yellow]")
        console.print("[yellow]Add --limit N to process more, or --limit 9999 for all.[/yellow]")

    run_analysis(dry_run_limit=limit, force=force)


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
