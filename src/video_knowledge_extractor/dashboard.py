"""Phase 5: Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_knowledge_extractor.config import Config
from video_knowledge_extractor.db import (
    get_all_videos,
    get_progress_stats,
    get_transcript,
    get_connection,
    init_db,
    search_transcripts,
    load_synthesis,
)
from video_knowledge_extractor.models import VideoStatus
from video_knowledge_extractor.notes import export_markdown
from video_knowledge_extractor.orchestrator import run_synthesis


st.set_page_config(
    page_title="Video Knowledge Extractor",
    page_icon="🧠",
    layout="wide",
)
Config.ensure_dirs()
init_db(data_dir=Config.DATA_DIR)


def _load_analysis(video_id: str):
    """Load analysis from DB."""
    with get_connection(Config.DATA_DIR) as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            return None
        import json
        return {
            "video_id": row["video_id"],
            "summary_no": row["summary_no"],
            "insights": json.loads(row["insights_json"]),
            "techniques_tools": json.loads(row["techniques_tools_json"]),
            "actionable_steps": json.loads(row["actionable_steps_json"]),
            "relevance_score": row["relevance_score"],
            "depth_score": row["depth_score"],
            "practical_score": row["practical_score"],
            "redundancy_score": row["redundancy_score"],
            "recommendation": row["recommendation"],
            "recommendation_justification": row["recommendation_justification"],
            "key_sections": json.loads(row["key_sections_json"]),
            "estimated_key_watch_minutes": row["estimated_key_watch_minutes"],
            "low_confidence": bool(row["low_confidence"]),
            "confidence_reason": row["confidence_reason"],
        }


def _all_analyses():
    with get_connection(Config.DATA_DIR) as conn:
        rows = conn.execute("SELECT * FROM analyses").fetchall()
        import json
        return [
            {
                "video_id": r["video_id"],
                "summary_no": r["summary_no"],
                "insights": json.loads(r["insights_json"]),
                "techniques_tools": json.loads(r["techniques_tools_json"]),
                "actionable_steps": json.loads(r["actionable_steps_json"]),
                "relevance_score": r["relevance_score"],
                "depth_score": r["depth_score"],
                "practical_score": r["practical_score"],
                "redundancy_score": r["redundancy_score"],
                "recommendation": r["recommendation"],
                "recommendation_justification": r["recommendation_justification"],
                "key_sections": json.loads(r["key_sections_json"]),
                "estimated_key_watch_minutes": r["estimated_key_watch_minutes"],
                "low_confidence": bool(r["low_confidence"]),
                "confidence_reason": r["confidence_reason"],
            }
            for r in rows
        ]


def _ts_to_seconds(ts: str) -> int | None:
    """Parse 'MM:SS' or 'HH:MM:SS' into seconds for a YouTube deep link."""
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


def render_overview():
    st.title("🧠 Video Knowledge Extractor")
    st.markdown("---")

    stats = get_progress_stats(data_dir=Config.DATA_DIR)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Totale videoer", stats["total_videos"])
    col2.metric("Transkribert OK", stats["transcripts_ok"])
    col3.metric("Analysert OK", stats["analyses_ok"])
    col4.metric("Total varighet", stats["total_duration_formatted"])

    st.markdown("---")

    videos = get_all_videos(data_dir=Config.DATA_DIR)
    if not videos:
        st.info("Ingen videoer oppdaget ennå. Kjør discovery fra CLI.")
        return

    st.subheader("Videoer")
    search = st.text_input("Søk etter tittel eller kanal", "")
    status_filter = st.multiselect(
        "Filtrer etter status",
        options=list(VideoStatus),
        default=[VideoStatus.ANALYSIS_OK, VideoStatus.TRANSCRIPT_OK, VideoStatus.DISCOVERED],
    )

    filtered = [
        v for v in videos
        if (not search or search.lower() in v.title.lower() or search.lower() in v.channel.lower())
        and v.status in status_filter
    ]

    for v in filtered:
        with st.expander(f"{v.playlist_order}. {v.title} ({v.duration_formatted}) — {v.channel}"):
            st.write(f"**Status:** {v.status.value}")
            st.write(f"**URL:** [{v.url}]({v.url})")
            analysis = _load_analysis(v.video_id)
            if analysis:
                st.markdown("#### Sammendrag")
                st.write(analysis["summary_no"])
                st.markdown("#### Innsikter")
                st.caption("✅ sitat verifisert i transkriptet · ⚠️ sitat ikke funnet · • ingen sitat. Tidsstempel er klikkbart.")
                for ins in analysis["insights"]:
                    mark = "✅" if ins.get("verified") else ("⚠️" if ins.get("quote") else "•")
                    ts = ins.get("timestamp") or ""
                    ts_link = ""
                    secs = _ts_to_seconds(ts)
                    if secs is not None:
                        deep_link = f"https://www.youtube.com/watch?v={v.video_id}&t={secs}s"
                        ts_link = f" [⏱ {ts}]({deep_link})"
                    st.write(f"- {mark} **{ins['insight']}**{ts_link}: {ins['explanation']}")
                    reason = ins.get("verification_reason") or ""
                    if reason:
                        st.caption(f"   {mark} {reason}")
                st.markdown("#### Teknikker og verktøy")
                st.write(", ".join(analysis["techniques_tools"]) or "Ingen")
                st.markdown("#### Handlingsbare steg")
                for step in analysis["actionable_steps"]:
                    st.write(f"- {step}")
                cols = st.columns(4)
                cols[0].metric("Relevans", analysis["relevance_score"])
                cols[1].metric("Dybde", analysis["depth_score"])
                cols[2].metric("Praktisk nytte", analysis["practical_score"])
                cols[3].metric("Redundans", analysis["redundancy_score"])
                rec = analysis["recommendation"]
                color = {"watch": "green", "skim": "orange", "skip": "red"}.get(rec, "gray")
                st.markdown(
                    f"**Anbefaling:** :{color}[{rec.upper()}] — {analysis['recommendation_justification']}"
                )
                st.write(f"**Estimert nøkkel-tid:** {analysis['estimated_key_watch_minutes']:.1f} min")
                if analysis["low_confidence"]:
                    st.warning(f"Lav konfidens: {analysis['confidence_reason']}")
            else:
                st.info("Ingen analyse tilgjengelig ennå.")


def render_ranking():
    st.title("🏆 Ranking")
    analyses = _all_analyses()
    if not analyses:
        st.info("Ingen analyser ennå.")
        return

    videos = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
    scored = [
        (a, (a["practical_score"] * a["relevance_score"]) / (a["redundancy_score"] + 1))
        for a in analyses
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (a, score) in enumerate(scored, 1):
        v = videos.get(a["video_id"])
        title = v.title if v else a["video_id"]
        st.write(f"{rank}. **{title}** — verdi: {score:.1f} (praktisk {a['practical_score']}, relevans {a['relevance_score']})")


def render_search():
    st.title("🔎 Søk i transkripter")
    st.caption("Nøkkelordsøk (SQLite FTS5) og, hvis du har kjørt `embed`, semantisk søk etter betydning.")
    query = st.text_input("Søksterm", "", label_visibility="collapsed")
    if not query.strip():
        st.info("Skriv et søksterm for å søke i transkriptene.")
        return

    # --- Keyword search (FTS5) ---
    st.markdown("### Nøkkelordsøk")
    hits = search_transcripts(query, limit=50, data_dir=Config.DATA_DIR)
    videos = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
    if not hits:
        st.warning(f"Ingen treff for «{query}».")
    else:
        st.write(f"{len(hits)} treff:")
        for hit in hits:
            v = videos.get(hit["video_id"])
            title = v.title if v else hit["video_id"]
            channel = v.channel if v else ""
            deep_link = f"https://www.youtube.com/watch?v={hit['video_id']}"
            st.markdown(f"**{title}** — {channel}\n\n{hit['snippet']}\n\n[Åpne video]({deep_link})")

    # --- Semantic search (embeddings), when configured ---
    from .embeddings import embeddings_configured, search_embeddings

    if embeddings_configured():
        st.markdown("### Semantisk søk (betydning)")
        try:
            semantic = search_embeddings(query, data_dir=Config.DATA_DIR, top_k=10)
        except Exception:  # noqa: BLE001 - a missing model should not break the page
            st.error("Semantisk søk feilet — kjør `embed` og sørg for at embeddingsmodellen kjører.")
            return
        if not semantic:
            st.info("Ingen embeddings ennå. Kjør `python src/cli.py embed` først.")
        else:
            for r in semantic:
                v = videos.get(r["video_id"])
                title = v.title if v else r["video_id"]
                st.markdown(
                    f"**{title}** (chunk {r['chunk_index']}, likhet {r['score']:.3f})\n\n{r['snippet']}"
                )


def render_synthesis():
    st.title("🧩 Syntese av samlingen")
    st.caption("Én samlet analyse: felles tema, motsetninger og unik kunnskap fra flere videoer.")

    if st.button("🔄 Generer / oppdater syntese"):
        with st.spinner("Genererer syntese (modellkall)..."):
            try:
                run_synthesis(force=True)
            except Exception as exc:
                st.error(f"Syntesen feilet: {exc}")
        st.rerun()

    synth = load_synthesis(data_dir=Config.DATA_DIR)
    if not synth:
        st.info("Ingen syntese ennå. Klikk «Generer / oppdater» (krever minst to analyserte videoer).")
        return

    analyzed_now = sorted(a["video_id"] for a in _all_analyses())
    covered = sorted(synth.get("video_ids", []))
    if analyzed_now != covered:
        st.warning("Syntesen er basert på et annet sett av videoer enn nå. Klikk «Generer / oppdater» for å friske opp.")

    st.markdown("### Samlet tema")
    st.write(synth.get("theme_no", ""))

    st.markdown("### Felles tema (overlapp)")
    shared = synth.get("shared_topics", [])
    if shared:
        for t in shared:
            st.write(f"- {t}")
    else:
        st.write("—")

    st.markdown("### Motsetninger")
    cons = synth.get("contradictions", [])
    if cons:
        for c in cons:
            st.write(f"- **{c.get('topic')}** ({', '.join(c.get('videos', []))}): {c.get('note')}")
    else:
        st.write("Ingen tydelige motsetninger.")

    st.markdown("### Unik kunnskap per video")
    uniq = synth.get("unique_knowledge", [])
    if uniq:
        for u in uniq:
            st.write(f"- **{u.get('video_id')}**: {u.get('knowledge')}")
    else:
        st.write("—")

    st.markdown("### Mangler (gaps)")
    gaps = synth.get("gaps", [])
    if gaps:
        for g in gaps:
            st.write(f"- {g}")
    else:
        st.write("—")

    st.markdown("### Anbefalt sesjonsrekkefølge")
    order = synth.get("watch_order", [])
    if order:
        for w in order:
            st.write(f"- **{w.get('video_id')}**: {w.get('reason')}")
    else:
        st.write("—")

    if synth.get("recommendation"):
        st.markdown("### Anbefaling")
        st.write(synth["recommendation"])

    st.caption(f"Generert: {synth.get('created_at', '')} · modell: {synth.get('llm_model', '')} · {len(covered)} videoer")


def render_topics():
    st.title("📚 Temaer")
    st.info("Temaklyngevisning er ikke implementert ennå.")


def render_exports():
    st.title("📤 Eksport")
    videos = get_all_videos(data_dir=Config.DATA_DIR)
    analyses = _all_analyses()
    if st.button("Eksportér JSON"):
        data = {
            "videos": [
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "url": v.url,
                    "channel": v.channel,
                    "duration_seconds": v.duration_seconds,
                    "status": v.status.value,
                }
                for v in videos
            ],
            "analyses": analyses,
        }
        path = Config.EXPORTS_DIR / "export.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success(f"Eksportert til {path}")
        st.download_button("Last ned JSON", path.read_text(encoding="utf-8"), "knowledge.json")

    if st.button("Eksportér Markdown"):
        md_files = export_markdown(data_dir=Config.DATA_DIR)
        st.success(
            f"Eksportert {len(md_files)} Markdown-filer til {Config.EXPORTS_DIR / 'markdown'} "
            "(Obsidian/OpenCode-vennlig, med kildelenker til YouTube)."
        )


PAGES = {
    "Oversikt": render_overview,
    "Søk": render_search,
    "Syntese": render_synthesis,
    "Ranking": render_ranking,
    "Temaer": render_topics,
    "Eksport": render_exports,
}


def main():
    page = st.sidebar.radio("Navigasjon", list(PAGES.keys()))
    PAGES[page]()


if __name__ == "__main__":
    main()
