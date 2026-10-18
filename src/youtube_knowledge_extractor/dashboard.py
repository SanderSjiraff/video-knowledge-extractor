"""Phase 5: Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_knowledge_extractor.config import Config
from youtube_knowledge_extractor.db import (
    get_all_videos,
    get_progress_stats,
    get_transcript,
    get_connection,
    init_db,
)
from youtube_knowledge_extractor.models import VideoStatus


st.set_page_config(
    page_title="YouTube Knowledge Extractor",
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


def render_overview():
    st.title("🧠 YouTube Knowledge Extractor")
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
                for ins in analysis["insights"]:
                    ts = f" ({ins.get('timestamp', '')})" if ins.get("timestamp") else ""
                    st.write(f"- **{ins['insight']}**{ts}: {ins['explanation']}")
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


PAGES = {
    "Oversikt": render_overview,
    "Ranking": render_ranking,
    "Temaer": render_topics,
    "Eksport": render_exports,
}


def main():
    page = st.sidebar.radio("Navigasjon", list(PAGES.keys()))
    PAGES[page]()


if __name__ == "__main__":
    main()
