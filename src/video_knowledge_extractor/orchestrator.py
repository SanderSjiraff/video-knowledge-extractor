"""Main orchestrator coordinating all phases."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from video_knowledge_extractor.config import Config
from video_knowledge_extractor.db import (
    get_all_analyses,
    get_all_videos,
    get_analyzed_video_ids,
    get_progress_stats,
    get_transcript,
    init_db,
    load_synthesis,
    save_synthesis,
)
from video_knowledge_extractor.discovery import discover_playlist
from video_knowledge_extractor.transcripts import fetch_transcript, process_transcripts
from video_knowledge_extractor.whisper_transcripts import process_whisper_transcripts
from video_knowledge_extractor.analysis import analyze_video, estimate_analysis_cost
from video_knowledge_extractor.cross_video import (
    estimate_synthesis_cost,
    synthesize_playlist,
)
from video_knowledge_extractor.models import Video, VideoAnalysis, VideoStatus


console = Console()


def run_discovery(source: str, limit: int | None = None) -> list[Video]:
    """Run Phase 1: Discover videos."""
    console.print("[bold blue]Phase 1: Discovery[/bold blue]")
    init_db(data_dir=Config.DATA_DIR)
    Config.ensure_dirs()

    videos = discover_playlist(source, limit=limit, data_dir=Config.DATA_DIR)

    accessible = [v for v in videos if v.status != VideoStatus.INACCESSIBLE]
    total_duration = sum(v.duration_seconds for v in accessible)
    m, s = divmod(total_duration, 60)
    h, m = divmod(m, 60)

    console.print(f"[green]Discovered {len(videos)} videos ({len(accessible)} accessible)[/green]")
    console.print(f"[green]Total estimated duration: {h}h {m}m {s}s[/green]")
    return videos


def run_transcripts(videos: list[Video] | None = None, force: bool = False) -> dict[str, int]:
    """Run Phase 2: Extract transcripts."""
    console.print("\n[bold blue]Phase 2: Transcript Extraction[/bold blue]")
    if videos is None:
        videos = get_all_videos(data_dir=Config.DATA_DIR)
        videos = [v for v in videos if v.status not in (VideoStatus.INACCESSIBLE,)]

    results = process_transcripts(videos, force=force)
    console.print(
        f"[green]Transcripts: {results['success']} OK, {results['failed']} failed[/green]"
    )
    return results


def run_whisper_transcripts(
    videos: list[Video] | None = None,
    model_size: str = "medium",
    force: bool = False,
    keep_audio: bool = True,
) -> dict[str, int]:
    """Run Phase 2b: local Whisper transcription (no YouTube subtitle API needed)."""
    console.print("\n[bold blue]Phase 2b: Local Whisper Transcription (audio + GPU)[/bold blue]")
    if videos is None:
        videos = get_all_videos(data_dir=Config.DATA_DIR)
        videos = [v for v in videos if v.status != VideoStatus.INACCESSIBLE]

    results = process_whisper_transcripts(
        videos, model_size=model_size, force=force, keep_audio=keep_audio
    )
    console.print(
        f"[green]Whisper transcripts: {results['success']} OK, {results['failed']} failed[/green]"
    )
    return results


def run_cost_estimate(videos: list[Video] | None = None) -> dict[str, Any]:
    """Estimate total cost for analyzing all videos with transcripts."""
    console.print("\n[bold blue]Cost Estimation[/bold blue]")
    Config.validate()
    if videos is None:
        videos = get_all_videos(data_dir=Config.DATA_DIR)

    from video_knowledge_extractor.db import get_transcript

    total_cost = 0.0
    total_in = 0
    total_out = 0
    count = 0
    model = Config.get_llm_model()

    for v in videos:
        if v.status == VideoStatus.INACCESSIBLE:
            continue
        t = get_transcript(v.video_id, data_dir=Config.DATA_DIR)
        if t is None:
            continue
        est = estimate_analysis_cost(t, model=model)
        total_cost += est["estimated_cost_usd"]
        total_in += est["tokens_in"]
        total_out += est["tokens_out"]
        count += 1

    console.print(f"[cyan]Videos ready for analysis: {count}[/cyan]")
    console.print(f"[cyan]Total estimated tokens: {total_in:,} in / {total_out:,} out[/cyan]")
    console.print(f"[cyan]Total estimated cost: ${total_cost:.4f} USD[/cyan]")
    console.print(f"[cyan]Spending limit: ${Config.SPENDING_LIMIT_USD:.2f} USD[/cyan]")

    return {
        "videos_ready": count,
        "tokens_in": total_in,
        "tokens_out": total_out,
        "estimated_cost_usd": total_cost,
        "spending_limit_usd": Config.SPENDING_LIMIT_USD,
        "model": model,
    }


def run_synthesis(force: bool = False) -> dict[str, Any] | None:
    """Run Phase 4: synthesize a combined view across all analyzed videos."""
    console.print("\n[bold blue]Phase 4: Playlist Synthesis[/bold blue]")
    Config.validate()
    analyses = get_all_analyses(data_dir=Config.DATA_DIR)
    if len(analyses) < 2:
        console.print("[yellow]Need at least two analyzed videos to synthesize. Run 'analyze' first.[/yellow]")
        return None

    current_ids = sorted(a.video_id for a in analyses)
    cached = load_synthesis(data_dir=Config.DATA_DIR)
    if not force and cached and sorted(cached.get("video_ids", [])) == current_ids:
        console.print("[cyan]Using cached synthesis (pass --force to regenerate).[/cyan]")
        return cached

    videos = get_all_videos(data_dir=Config.DATA_DIR)
    video_meta = {
        v.video_id: {"title": v.title, "channel": v.channel, "url": v.url}
        for v in videos
    }

    est = estimate_synthesis_cost(analyses, video_meta)
    if est["estimated_cost_usd"] > Config.SPENDING_LIMIT_USD:
        console.print(
            f"[red]ERROR: Estimated synthesis cost ${est['estimated_cost_usd']:.4f} exceeds "
            f"spending limit ${Config.SPENDING_LIMIT_USD:.2f}. Aborting.[/red]"
        )
        return None
    if est["estimated_cost_usd"] > 0 and not Config.USE_OLLAMA and not Config.LLM_API_KEY:
        console.print("[red]ERROR: No API key configured. Set LLM_API_KEY or enable Ollama.[/red]")
        return None

    console.print(
        f"[cyan]Synthesizing {len(analyses)} videos with model: {Config.get_llm_model()} "
        f"(~${est['estimated_cost_usd']:.4f} est.)[/cyan]"
    )
    raw = synthesize_playlist(analyses, video_meta)
    synthesis = {
        **raw,
        "video_ids": current_ids,
        "llm_model": Config.get_llm_model(),
        "created_at": datetime.utcnow().isoformat(),
    }
    save_synthesis(synthesis, data_dir=Config.DATA_DIR)
    console.print(f"[green]Synthesis OK for {len(analyses)} videos[/green]")
    return synthesis


def run_analysis(
    videos: list[Video] | None = None,
    dry_run_limit: int | None = None,
    force: bool = False,
) -> list[VideoAnalysis]:
    """Run Phase 3: AI analysis."""
    console.print("\n[bold blue]Phase 3: AI Analysis[/bold blue]")
    Config.validate()
    if dry_run_limit is not None and dry_run_limit < 1:
        raise ValueError("limit must be at least 1")
    if videos is None:
        videos = get_all_videos(data_dir=Config.DATA_DIR)

    # Only analyze videos with transcripts
    eligible = [v for v in videos if v.status != VideoStatus.INACCESSIBLE
                and get_transcript(v.video_id, data_dir=Config.DATA_DIR) is not None]

    if not force:
        analyzed_ids = get_analyzed_video_ids(data_dir=Config.DATA_DIR)
        eligible = [v for v in eligible if v.video_id not in analyzed_ids]

    if dry_run_limit is not None:
        eligible = eligible[:dry_run_limit]
        console.print(f"[yellow]Batch limit: analyzing at most {dry_run_limit} videos[/yellow]")

    if not eligible:
        console.print("[yellow]No videos with transcripts ready for analysis.[/yellow]")
        return []

    # Cost check
    cost_info = run_cost_estimate(eligible)
    if cost_info["estimated_cost_usd"] > Config.SPENDING_LIMIT_USD:
        console.print(
            f"[red]ERROR: Estimated cost ${cost_info['estimated_cost_usd']:.4f} exceeds spending limit "
            f"${Config.SPENDING_LIMIT_USD:.2f}. Aborting.[/red]"
        )
        return []

    if cost_info["estimated_cost_usd"] > 0 and not Config.USE_OLLAMA and not Config.LLM_API_KEY:
        console.print(
            "[red]ERROR: No API key configured. Set LLM_API_KEY or enable Ollama.[/red]"
        )
        return []

    console.print(
        f"[cyan]Starting analysis of {len(eligible)} videos with model: {Config.get_llm_model()}[/cyan]"
    )

    analyses: list[VideoAnalysis] = []
    for v in eligible:
        analysis = analyze_video(v, force=force)
        if analysis:
            analyses.append(analysis)
            # Save to DB
            from video_knowledge_extractor.db import get_connection
            import json as json_mod
            with get_connection(Config.DATA_DIR) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO analyses
                    (video_id, summary_no, insights_json, techniques_tools_json, actionable_steps_json,
                     relevance_score, depth_score, practical_score, redundancy_score,
                     recommendation, recommendation_justification, key_sections_json,
                     estimated_key_watch_minutes, low_confidence, confidence_reason,
                     analyzed_at, llm_model, estimated_tokens_in, estimated_tokens_out)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis.video_id,
                        analysis.summary_no,
                        json_mod.dumps([
                            {"insight": i.insight, "explanation": i.explanation,
                             "timestamp": i.timestamp, "confidence": i.confidence,
                             "quote": i.quote, "verified": i.verified,
                             "verification_reason": i.verification_reason}
                            for i in analysis.insights
                        ]),
                        json_mod.dumps(analysis.techniques_tools),
                        json_mod.dumps(analysis.actionable_steps),
                        analysis.relevance_score,
                        analysis.depth_score,
                        analysis.practical_score,
                        analysis.redundancy_score,
                        analysis.recommendation,
                        analysis.recommendation_justification,
                        json_mod.dumps(analysis.key_sections),
                        analysis.estimated_key_watch_minutes,
                        1 if analysis.low_confidence else 0,
                        analysis.confidence_reason,
                        analysis.analyzed_at.isoformat(),
                        analysis.llm_model,
                        analysis.estimated_tokens_in,
                        analysis.estimated_tokens_out,
                    ),
                )
                conn.execute(
                    "UPDATE videos SET status = ?, updated_at = ?, error_message = NULL WHERE video_id = ?",
                    (VideoStatus.ANALYSIS_OK.value, analysis.analyzed_at.isoformat(), analysis.video_id),
                )
                conn.commit()

    console.print(f"[green]Analyzed {len(analyses)} / {len(eligible)} videos[/green]")
    return analyses


def run_dashboard() -> None:
    """Launch Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "dashboard.py"
    console.print("[bold blue]Launching Streamlit dashboard...[/bold blue]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


def print_status() -> None:
    """Print current processing status."""
    stats = get_progress_stats(data_dir=Config.DATA_DIR)
    table = Table(title="Processing Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total videos", str(stats["total_videos"]))
    table.add_row("Discovered", str(stats["discovered"]))
    table.add_row("Transcripts OK", str(stats["transcripts_ok"]))
    table.add_row("Transcripts failed", str(stats["transcripts_failed"]))
    table.add_row("Analyses OK", str(stats["analyses_ok"]))
    table.add_row("Analyses failed", str(stats["analyses_failed"]))
    table.add_row("Inaccessible", str(stats["inaccessible"]))
    table.add_row("Total duration", stats["total_duration_formatted"])
    console.print(table)
