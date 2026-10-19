"""Markdown note export with clickable source deep-links.

Produces human-readable, Obsidian/OpenCode-friendly notes: one file per
analyzed video plus an index that also folds in the playlist synthesis.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .cross_video import rank_videos_by_value
from .db import get_all_analyses, get_all_videos, load_synthesis
from .models import Video, VideoAnalysis


def _ts_to_seconds(ts: Optional[str]) -> Optional[int]:
    """Parse 'MM:SS' or 'HH:MM:SS' into seconds (or None)."""
    try:
        nums = [int(x) for x in (ts or "").split(":")]
    except ValueError:
        return None
    if not 1 <= len(nums) <= 3:
        return None
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def deep_link(video_id: str, timestamp: Optional[str]) -> str:
    """A YouTube deep link; jumps to the timestamp when one is given."""
    base = f"https://www.youtube.com/watch?v={video_id}"
    secs = _ts_to_seconds(timestamp)
    return f"{base}&t={secs}s" if secs is not None else base


def _confidence_mark(verified: bool, confidence: str, has_quote: bool) -> str:
    if verified:
        return "✅"
    if has_quote or confidence == "low":
        return "⚠️"
    return ""


def _safe_filename(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^0-9a-zA-ZæøåÆØÅ]+", "-", text).strip("-")
    return slug[:max_len].rstrip("-") or "notat"


def note_filename(video: Video) -> str:
    """Deterministic, unique, readable note filename (title + video id)."""
    return f"{_safe_filename(video.title)}.{video.video_id}.md"


def video_to_markdown(video: Video, analysis: VideoAnalysis) -> str:
    """Render one video's analysis as a Markdown note with source deep-links."""
    lines: list[str] = []
    lines.append(f"# {video.title or video.video_id}")
    lines.append("")
    lines.append(f"- **Kanal:** {video.channel or '—'}")
    lines.append(f"- **Varighet:** {video.duration_formatted}")
    if analysis.recommendation:
        lines.append(
            f"- **Anbefaling:** {analysis.recommendation} — {analysis.recommendation_justification}"
        )
    lines.append(
        f"- **Skår:** relevans {analysis.relevance_score}/5 · dybde {analysis.depth_score}/5 · "
        f"praktisk {analysis.practical_score}/5 · redundans {analysis.redundancy_score}/5"
    )
    lines.append(f"- **Video:** [Se på YouTube]({video.url})")
    if analysis.low_confidence:
        lines.append(f"- **Lav konfidens:** {analysis.confidence_reason}")
    lines.append("")

    lines.append("## Sammendrag")
    lines.append("")
    lines.append(analysis.summary_no or "_Ingen._")
    lines.append("")

    lines.append("## Innsikter")
    lines.append("")
    if analysis.insights:
        for n, ins in enumerate(analysis.insights, 1):
            mark = _confidence_mark(ins.verified, ins.confidence, bool(ins.quote))
            ts_link = ""
            if ins.timestamp:
                ts_link = f" [⏱ {ins.timestamp}]({deep_link(video.video_id, ins.timestamp)})"
            head = f"{n}. **{ins.insight}**{ts_link}"
            if mark:
                head += f" {mark}"
            lines.append(head)
            if ins.explanation:
                lines.append(f"   {ins.explanation}")
            if ins.quote:
                lines.append(f"   > {ins.quote}")
            if ins.verification_reason:
                lines.append(f"   _{ins.verification_reason}_")
    else:
        lines.append("_Ingen innsikter._")
    lines.append("")

    lines.append("## Teknikker og verktøy")
    lines.append("")
    lines.extend([f"- {t}" for t in analysis.techniques_tools] or ["_Ingen._"])
    lines.append("")

    lines.append("## Handlingsbare steg")
    lines.append("")
    lines.extend([f"{n}. {s}" for n, s in enumerate(analysis.actionable_steps, 1)] or ["_Ingen._"])

    if analysis.key_sections:
        lines.append("")
        lines.append("## Nøkkelseksjoner")
        lines.append("")
        for sec in analysis.key_sections:
            start, end = sec.get("start", ""), sec.get("end", "")
            link = f" [hoppe til]({deep_link(video.video_id, start)})" if start else ""
            lines.append(f"- **{start}–{end}**{link}: {sec.get('description', '')}")

    lines.append("")
    lines.append(f"_Generert {datetime.utcnow():%Y-%m-%d %H:%M} av YouTube Knowledge Extractor._")
    return "\n".join(lines) + "\n"


def playlist_overview_markdown(
    videos: list[Video],
    analyses: list[VideoAnalysis],
    synthesis: Optional[dict] = None,
) -> str:
    """Render the index note: synthesis (if any) + ranked video list + note links."""
    video_by_id = {v.video_id: v for v in videos}
    by_id = {a.video_id: a for a in analyses}
    ranked = rank_videos_by_value(analyses)

    lines: list[str] = []
    lines.append("# YouTube Knowledge Extractor — samling")
    lines.append("")
    lines.append(
        f"_Generert {datetime.utcnow():%Y-%m-%d %H:%M} · {len(analyses)} analyserte videoer._"
    )
    lines.append("")

    if synthesis:
        lines.append("## Syntese")
        lines.append("")
        if synthesis.get("theme_no"):
            lines.append(synthesis["theme_no"])
            lines.append("")
        cons = synthesis.get("contradictions", [])
        if cons:
            lines.append("**Motsetninger:**")
            for c in cons:
                lines.append(f"- {c.get('topic')} ({', '.join(c.get('videos', []))}): {c.get('note')}")
            lines.append("")
        uniq = synthesis.get("unique_knowledge", [])
        if uniq:
            lines.append("**Unik kunnskap:**")
            for u in uniq:
                vid = u.get("video_id", "")
                v = video_by_id.get(vid)
                title = v.title if v else vid
                lines.append(f"- [{title}]({deep_link(vid, None)}): {u.get('knowledge')}")
            lines.append("")
        shared = synthesis.get("shared_topics", [])
        if shared:
            lines.append("**Felles tema:** " + ", ".join(shared))
            lines.append("")

    lines.append("## Videolisten")
    lines.append("")
    lines.append("| # | Tittel | Kanal | Anbefaling | Verdi |")
    lines.append("|---|--------|-------|------------|-------|")
    for rank, (vid, score) in enumerate(ranked, 1):
        v = video_by_id.get(vid)
        a = by_id[vid]
        title = v.title if v else vid
        channel = v.channel if v else "—"
        url = v.url if v else deep_link(vid, None)
        lines.append(f"| {rank} | [{title}]({url}) | {channel} | {a.recommendation} | {score} |")
    lines.append("")

    lines.append("## Notater")
    lines.append("")
    for rank, (vid, score) in enumerate(ranked, 1):
        v = video_by_id.get(vid)
        title = v.title if v else vid
        fname = note_filename(v) if v else f"notat.{vid}.md"
        lines.append(f"- [{title}]({fname})")
    lines.append("")
    return "\n".join(lines) + "\n"


def export_markdown(data_dir: Path, out_dir: Optional[Path] = None) -> list[Path]:
    """Write one Markdown note per analyzed video plus an index. Returns paths."""
    out_dir = Path(out_dir) if out_dir else Path(data_dir) / "exports" / "markdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = get_all_videos(data_dir=data_dir)
    analyses = get_all_analyses(data_dir=data_dir)
    video_by_id = {v.video_id: v for v in videos}

    written: list[Path] = []
    for a in analyses:
        v = video_by_id.get(a.video_id)
        if v is None:
            v = Video(a.video_id, a.video_id, deep_link(a.video_id, None), 0, "", 0)
        path = out_dir / note_filename(v)
        path.write_text(video_to_markdown(v, a), encoding="utf-8")
        written.append(path)

    synthesis = load_synthesis(data_dir=data_dir)
    index = out_dir / "INDEKS.md"
    index.write_text(playlist_overview_markdown(videos, analyses, synthesis), encoding="utf-8")
    written.append(index)
    return written
