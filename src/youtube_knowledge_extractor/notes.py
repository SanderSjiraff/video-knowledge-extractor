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


def _yaml(value: str) -> str:
    """Double-quote a scalar for safe YAML frontmatter."""
    return '"' + (value or "").replace('"', '\\"') + '"'


def _frontmatter(video: Video, analysis: VideoAnalysis) -> str:
    """YAML frontmatter for Obsidian / programmatic parsing."""
    lines = ["---"]
    lines.append(f"video_id: {video.video_id}")
    lines.append(f"platform: {video.platform or 'youtube'}")
    lines.append(f"title: {_yaml(video.title)}")
    lines.append(f"url: {_yaml(video.url)}")
    if video.channel:
        lines.append(f"channel: {_yaml(video.channel)}")
    lines.append(f"duration: {video.duration_formatted}")
    if analysis.recommendation:
        lines.append(f"recommendation: {analysis.recommendation}")
    lines.append(f"relevance: {analysis.relevance_score}")
    lines.append(f"depth: {analysis.depth_score}")
    lines.append(f"practical: {analysis.practical_score}")
    lines.append(f"redundancy: {analysis.redundancy_score}")
    lines.append(f"generated: {datetime.utcnow():%Y-%m-%d}")
    tags = list(dict.fromkeys(analysis.techniques_tools))  # dedupe, keep order
    if tags:
        lines.append("tags:")
        for t in tags:
            lines.append(f"  - {_yaml(t)}")
    lines.append("---")
    return "\n".join(lines)


def video_to_markdown(video: Video, analysis: VideoAnalysis) -> str:
    """Render one video's analysis as a Markdown note with source deep-links."""
    lines: list[str] = []
    lines.append(_frontmatter(video, analysis))
    lines.append("")
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


def knowledge_concentrate_markdown(
    videos: list[Video],
    analyses: list[VideoAnalysis],
    synthesis: Optional[dict] = None,
) -> str:
    """A compact, deduplicated knowledge document optimized for LLM chat context.

    Unlike INDEKS (which is a human-readable table), this strips structure down
    to the highest-value, non-repeating knowledge so it loads efficiently as a
    system prompt / RAG context.
    """
    video_by_id = {v.video_id: v for v in videos}
    lines: list[str] = []
    lines.append("# Kunnskapskonsentrat")
    lines.append("")
    lines.append(
        f"_Kompakt, deduplisert kunnskap fra {len(analyses)} analyserte videoer. "
        f"Lagt {datetime.utcnow():%Y-%m-%d} for inntasting som kontekst._"
    )
    lines.append("")

    if synthesis:
        if synthesis.get("theme_no"):
            lines.append("## Samlet tema")
            lines.append("")
            lines.append(synthesis["theme_no"])
            lines.append("")
        shared = synthesis.get("shared_topics", [])
        if shared:
            lines.append("## Felles tema")
            lines.append("")
            lines.extend(f"- {t}" for t in shared)
            lines.append("")
        cons = synthesis.get("contradictions", [])
        if cons:
            lines.append("## Motsetninger")
            lines.append("")
            for c in cons:
                lines.append(f"- {c.get('topic')} ({', '.join(c.get('videos', []))}): {c.get('note')}")
            lines.append("")

    # Top insights across all videos, deduplicated, verified-first.
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    pool: list[tuple] = []
    for a in analyses:
        v = video_by_id.get(a.video_id)
        title = v.title if v else a.video_id
        for ins in a.insights:
            pool.append((ins, title, a.relevance_score))
    pool.sort(key=lambda x: (not x[0].verified, conf_rank.get(x[0].confidence, 1), -x[2]))
    lines.append("## Top-innsikter (samlet, deduplisert)")
    lines.append("")
    seen: set[str] = set()
    count = 0
    for ins, title, _rel in pool:
        key = " ".join(ins.insight.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        mark = "✅" if ins.verified else ("⚠️" if (ins.quote or ins.confidence == "low") else "")
        ts = f" ({ins.timestamp})" if ins.timestamp else ""
        lines.append(f"- **{title}**{ts}: {ins.insight}{(' ' + mark) if mark else ''}")
        count += 1
        if count >= 40:
            lines.append("- … (flere innsikter i de enkelte notater)")
            break
    lines.append("")

    techniques = list(dict.fromkeys(t for a in analyses for t in a.techniques_tools))
    if techniques:
        lines.append("## Teknikker og verktøy (samlet)")
        lines.append("")
        lines.extend(f"- {t}" for t in techniques)
        lines.append("")

    steps = list(dict.fromkeys(s for a in analyses for s in a.actionable_steps))
    if steps:
        lines.append("## Handlingsbare steg (samlet)")
        lines.append("")
        lines.extend(f"{i}. {s}" for i, s in enumerate(steps, 1))
        lines.append("")

    if synthesis:
        uniq = synthesis.get("unique_knowledge", [])
        if uniq:
            lines.append("## Unike bidrag per video")
            lines.append("")
            for u in uniq:
                vid = u.get("video_id", "")
                v = video_by_id.get(vid)
                title = v.title if v else vid
                lines.append(f"- **{title}**: {u.get('knowledge')}")
            lines.append("")
        order = synthesis.get("watch_order", [])
        if order:
            lines.append("## Anbefalt sesjonsrekkefølge")
            lines.append("")
            for i, w in enumerate(order, 1):
                vid = w.get("video_id", "")
                v = video_by_id.get(vid)
                title = v.title if v else vid
                lines.append(f"{i}. **{title}**: {w.get('reason')}")
            lines.append("")

    lines.append("---")
    lines.append("_For sitater og kildelenker: se notatene i samme mappe (én .md per video)._")
    lines.append("")
    return "\n".join(lines)


def export_markdown(data_dir: Path, out_dir: Optional[Path] = None) -> list[Path]:
    """Write one Markdown note per analyzed video plus an index and concentrate.

    Returns the list of written paths.
    """
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
    concentrate = out_dir / "KUNNSKAP.md"
    concentrate.write_text(
        knowledge_concentrate_markdown(videos, analyses, synthesis), encoding="utf-8"
    )
    written.append(concentrate)
    return written
