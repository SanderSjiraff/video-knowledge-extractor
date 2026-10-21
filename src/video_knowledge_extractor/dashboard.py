"""Platform dashboard (Streamlit).

Top-tabbed UI with the approved dark/violet design:
    📥 Send      - submit URLs / files, pick pipeline stages, live queue
    📚 Bibliotek - video cards, filters, detail view
    🔎 Søk       - keyword (FTS5) + semantic search
    🧩 Syntese   - combined collection synthesis
    🏆 Ranking   - value ranking of analyzed videos
    📤 Eksport   - JSON / Markdown export

Submissions go straight into the SQLite job queue (see jobs.py); a worker
process (`python src/cli.py worker`, or part of `platform`) picks them up.
"""

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
    get_screen_text,
    get_transcript,
    get_frame_descriptions,
    get_connection,
    init_db,
    search_transcripts,
    load_synthesis,
)
from video_knowledge_extractor.models import VideoStatus
from video_knowledge_extractor.notes import export_markdown
from video_knowledge_extractor.orchestrator import run_synthesis
from video_knowledge_extractor.discovery import detect_platform
from video_knowledge_extractor.jobs import enqueue_job, list_jobs

st.set_page_config(page_title="VKE – Video Knowledge Extractor", page_icon="🧠", layout="wide")
Config.ensure_dirs()
init_db(data_dir=Config.DATA_DIR)

# ---------------------------------------------------------------------------
# Styling (dark theme comes from .streamlit/config.toml; these are the custom
# components: badges, stage pills, chips, cards).
# ---------------------------------------------------------------------------
STYLES = """
<style>
.vke-header { display:flex; align-items:center; justify-content:space-between; padding: 4px 2px 14px; }
.vke-logo { font-size: 19px; font-weight: 800; letter-spacing: .3px; display:flex; gap:10px; align-items:center; }
.vke-logo-badge { background: linear-gradient(135deg,#8b5cf6,#c084fc); border-radius:10px;
  width:34px; height:34px; display:inline-flex; align-items:center; justify-content:center; font-size:17px; }
.vke-status { display:inline-flex; align-items:center; gap:8px; background:#1a1e2e;
  border:1px solid #262b3d; border-radius:999px; padding:6px 14px; font-size:12.5px; color:#8b93a7; }
.vke-dot { width:8px; height:8px; border-radius:50%; background:#34d399; display:inline-block; }

.plat { font-size:10.5px; font-weight:700; padding:3px 9px; border-radius:999px; letter-spacing:.5px; }
.plat-yt { background:rgba(255,0,51,.15); color:#ff5c7a; border:1px solid rgba(255,0,51,.4); }
.plat-tt { background:rgba(37,244,238,.12); color:#25f4ee; border:1px solid rgba(37,244,238,.35); }
.plat-ig { background:linear-gradient(45deg,rgba(98,40,215,.25),rgba(238,42,123,.25)); color:#f472b6;
  border:1px solid rgba(238,42,123,.45); }
.plat-local { background:rgba(139,92,246,.14); color:#a78bfa; border:1px solid rgba(139,92,246,.4); }

.vke-job { background:#1a1e2e; border:1px solid #262b3d; border-radius:13px; padding:12px 16px; margin-bottom:10px; }
.vke-job-title { font-weight:600; font-size:14.5px; margin-bottom:9px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.vke-stages { display:flex; gap:8px; flex-wrap:wrap; }
.stage { display:inline-flex; align-items:center; gap:6px; font-size:12px; padding:5px 12px;
  border-radius:999px; background:#1f2437; color:#8b93a7; border:1px solid #262b3d; }
.stage .sdot { width:7px; height:7px; border-radius:50%; background:#3a415a; display:inline-block; }
.stage.done { color:#a7f3d0; background:rgba(52,211,153,.08); border-color:rgba(52,211,153,.3); }
.stage.done .sdot { background:#34d399; }
.stage.run { color:#c4b5fd; background:rgba(139,92,246,.14); border-color:rgba(139,92,246,.45); font-weight:600; }
.stage.run .sdot { background:#a78bfa; box-shadow:0 0 8px #a78bfa; }
.stage.skip { opacity:.4; }
.vke-job-note { color:#fbbf24; font-size:12.5px; margin-top:8px; }
.vke-job-err { color:#f87171; font-size:12.5px; margin-top:8px; }

.rec { font-size:11px; font-weight:800; padding:3px 11px; border-radius:999px; letter-spacing:.7px; }
.rec-watch { background:rgba(52,211,153,.14); color:#34d399; border:1px solid rgba(52,211,153,.4); }
.rec-skim { background:rgba(251,191,36,.12); color:#fbbf24; border:1px solid rgba(251,191,36,.4); }
.rec-skip { background:rgba(248,113,113,.12); color:#f87171; border:1px solid rgba(248,113,113,.4); }

.vke-card { background:#1a1e2e; border:1px solid #262b3d; border-radius:14px; overflow:hidden; margin-bottom:4px; }
.vke-thumb { height:108px; position:relative; display:flex; align-items:center; justify-content:center; }
.thumb-yt { background:linear-gradient(135deg,#312e81,#7c3aed); }
.thumb-tt { background:linear-gradient(135deg,#134e4a,#0ea5e9); }
.thumb-ig { background:linear-gradient(135deg,#7c2d12,#f59e0b); }
.thumb-local { background:linear-gradient(135deg,#14532d,#22c55e); }
.vke-play { width:40px; height:40px; border-radius:50%; background:rgba(13,15,23,.55);
  border:1.5px solid rgba(255,255,255,.35); display:flex; align-items:center; justify-content:center;
  font-size:13px; color:white; padding-left:3px; }
.vke-dur { position:absolute; bottom:9px; right:10px; font-size:11px; font-weight:600;
  background:rgba(13,15,23,.7); padding:3px 8px; border-radius:7px; }
.vke-card .vke-plat { position:absolute; top:9px; left:10px; }
.vke-card-body { padding:12px 14px 13px; }
.vke-card-title { font-size:14px; font-weight:600; line-height:1.35; margin-bottom:3px; min-height:2.7em;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.vke-card-chan { font-size:12px; color:#8b93a7; margin-bottom:9px; }
.vke-scores { display:flex; gap:14px; font-size:11px; color:#8b93a7; margin-bottom:9px; }
.dots i { width:6px; height:6px; border-radius:50%; background:#2c3350; display:inline-block; margin-right:3px; }
.dots i.on { background:#a78bfa; }
.vke-card-foot { display:flex; align-items:center; gap:10px; font-size:11.5px; color:#8b93a7; }
.vke-empty { color:#8b93a7; font-size:13px; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Small HTML builders
# ---------------------------------------------------------------------------
_PLAT_LABEL = {"youtube": ("YOUTUBE", "plat-yt"), "tiktok": ("TIKTOK", "plat-tt"),
               "instagram": ("INSTAGRAM", "plat-ig"), "local": ("FIL", "plat-local")}


def _plat_badge(platform: str) -> str:
    label, cls = _PLAT_LABEL.get(platform, ("VIDEO", "plat-local"))
    return f'<span class="plat {cls}">{label}</span>'


def _score_dots(n: int) -> str:
    dots = "".join(f'<i class="{"on" if i < n else ""}"></i>' for i in range(5))
    return f'<span class="dots">{dots}</span>'


def _rec_chip(rec: str) -> str:
    mapping = {"watch": ("SE", "rec-watch"), "skim": ("SKIM", "rec-skim"), "skip": ("SLEP", "rec-skip")}
    label, cls = mapping.get(rec or "", ("?", ""))
    return f'<span class="rec {cls}">{label}</span>'


def _job_stages_html(job: dict) -> str:
    """Render the 4 stage pills for a job, based on its current status."""
    opts = job.get("options") or {}
    status = job.get("status", "queued")
    has_transcribe = opts.get("transcribe", True)
    has_embed = opts.get("embed", True)
    is_slides = job.get("kind") == "slides"

    stage_order = ["Oppdaget", "Bilder" if is_slides else "Transkribert", "Analysert", "Indeksert"]
    # Map status to how many stages are done, and what (if anything) is running.
    stage_index = {
        "queued": (-1, None),
        "discovering": (-1, 0),
        "transcribing": (0, 1),
        "ocr": (1, "ocr"),
        "vision": (1, "vision"),
        "analyzing": (1 if not has_transcribe else 2, 2),
        "embedding": (2 if not has_embed else 3 if not has_transcribe else 3, 3),
        "done": (99, None),
        "failed": (-1, None),
    }
    done_before, running = stage_index.get(status, (-1, None))

    labels_run = {
        0: "Oppdager…",
        1: (
            f"Behandler bilder… {job.get('progress', 0)}%"
            if is_slides
            else f"Transkriberer… {job.get('progress', 0)}%"
        ),
        "ocr": "OCR på skjerm…",
        "vision": "Beskriver bilder…",
        2: "Analyserer…",
        3: "Indekserer…",
    }

    pills = []
    for i, label in enumerate(stage_order):
        if i in (1,) and not has_transcribe:
            pills.append(f'<span class="stage skip"><span class="sdot"></span>{label}</span>')
            continue
        if i == 3 and not has_embed:
            pills.append(f'<span class="stage skip"><span class="sdot"></span>{label}</span>')
            continue
        if status == "failed" and running is None and i == 0:
            pills.append(f'<span class="stage run"><span class="sdot"></span>Feilet</span>')
            break
        if status == "failed" and i > 0:
            pills.append(f'<span class="stage skip"><span class="sdot"></span>{label}</span>')
            continue
        if status == "done":
            pills.append(f'<span class="stage done"><span class="sdot"></span>{label}</span>')
        elif i <= done_before:
            pills.append(f'<span class="stage done"><span class="sdot"></span>{label}</span>')
        elif running == i:
            pills.append(f'<span class="stage run"><span class="sdot"></span>{labels_run[i]}</span>')
        else:
            pills.append(f'<span class="stage"><span class="sdot"></span>{label}</span>')
    if running in ("ocr", "vision"):
        # Insert the special sub-stage pill after the transcribed pill.
        label = "OCR på skjerm…" if running == "ocr" else "Beskriver bilder…"
        pill = f'<span class="stage run"><span class="sdot"></span>{label}</span>'
        insert_at = 2 if has_transcribe else 1
        pills.insert(insert_at, pill)
    return f'<div class="vke-stages">{"".join(pills)}</div>'


def _job_card_html(job: dict) -> str:
    title = job.get("title") or job.get("url") or "Video"
    if len(title) > 90:
        title = title[:87] + "…"
    extra = ""
    if job.get("status") == "done" and job.get("note"):
        extra = f'<div class="vke-job-note">ℹ {job["note"]}</div>'
    if job.get("status") == "failed":
        extra = f'<div class="vke-job-err">✖ {job.get("error_message") or "Ukjent feil"}</div>'
    return (
        f'<div class="vke-job">'
        f'<div class="vke-job-title">{_plat_badge(job.get("platform", "local"))} {title}</div>'
        f'{_job_stages_html(job)}'
        f'{extra}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Data helpers (from the old overview, shared by library + detail)
# ---------------------------------------------------------------------------
def _load_analysis(video_id: str):
    with get_connection(Config.DATA_DIR) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE video_id = ?", (video_id,)).fetchone()
    if row is None:
        return None
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


def _render_video_detail(video, analysis: dict | None) -> None:
    st.markdown(
        f'{_plat_badge(video.platform)} **{video.title or video.video_id}**   '
        f'<span style="color:#8b93a7">{video.channel} · {video.duration_formatted} · '
        f'{video.status.value}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"[Åpne {video.platform}]({video.url})")

    if not analysis:
        st.info("Ingen analyse tilgjengelig ennå.")
        return

    st.markdown("#### Sammendrag")
    st.write(analysis["summary_no"])
    st.markdown("#### Innsikter")
    st.caption("✅ sitat verifisert i transkriptet · ⚠️ sitat ikke funnet · • ingen sitat. Tidsstempel er klikkbart (YouTube).")
    for ins in analysis["insights"]:
        mark = "✅" if ins.get("verified") else ("⚠️" if ins.get("quote") else "•")
        ts = ins.get("timestamp") or ""
        ts_link = ""
        if ts and video.platform == "youtube":
            secs = _ts_to_seconds(ts)
            if secs is not None:
                deep_link = f"https://www.youtube.com/watch?v={video.video_id}&t={secs}s"
                ts_link = f" [⏱ {ts}]({deep_link})"
        elif ts:
            ts_link = f" ⏱ {ts}"
        st.write(f"- {mark} **{ins['insight']}**{ts_link}: {ins['explanation']}")
        reason = ins.get("verification_reason") or ""
        if reason:
            st.caption(f"   {mark} {reason}")

    st.markdown("#### Teknikker og verktøy")
    st.write(", ".join(analysis["techniques_tools"]) or "Ingen")
    st.markdown("#### Handlingsbare steg")
    for step in analysis["actionable_steps"]:
        st.write(f"- {step}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevans", analysis["relevance_score"])
    c2.metric("Dybde", analysis["depth_score"])
    c3.metric("Praktisk nytte", analysis["practical_score"])
    c4.metric("Redundans", analysis["redundancy_score"])
    rec = analysis["recommendation"]
    color = {"watch": "green", "skim": "orange", "skip": "red"}.get(rec, "gray")
    st.markdown(
        f"**Anbefaling:** :{color}[{rec.upper()}] — {analysis['recommendation_justification']}"
    )
    st.write(f"**Estimert nøkkel-tid:** {analysis['estimated_key_watch_minutes']:.1f} min")
    if analysis["low_confidence"]:
        st.warning(f"Lav konfidens: {analysis['confidence_reason']}")

    with st.expander("Transkript"):
        t = get_transcript(video.video_id, data_dir=Config.DATA_DIR)
        if t:
            for seg in t.segments:
                st.markdown(f"`{seg.timestamp}` — {seg.text}")
        else:
            st.caption("Ingen transkript lagret.")

    screen = get_screen_text(video.video_id, data_dir=Config.DATA_DIR)
    if screen:
        with st.expander(f"Tekst på skjermen (OCR, {len(screen)} treff)"):
            for item in screen:
                st.markdown(f"`{int(item['time_seconds']) // 60}:{int(item['time_seconds']) % 60:02d}` — {item['text']}")

    frames = get_frame_descriptions(video.video_id, data_dir=Config.DATA_DIR)
    if frames:
        with st.expander(f"Beskrivelse av skjermen ({len(frames)} frames)"):
            for item in frames:
                st.markdown(f"`{int(item['time_seconds']) // 60}:{int(item['time_seconds']) % 60:02d}` — {item['description']}")


# ---------------------------------------------------------------------------
# Page: Send
# ---------------------------------------------------------------------------
@st.fragment(run_every=2)
def _live_queue():
    jobs = list_jobs(limit=8, data_dir=Config.DATA_DIR)
    if not jobs:
        st.markdown('<span class="vke-empty">Køen er tom. Send noen videoer over ↑</span>', unsafe_allow_html=True)
        return
    done_count = sum(1 for j in jobs if j["status"] == "done")
    st.markdown(f"**Kø** &nbsp;<span style='color:#8b93a7; font-size:13px'>{len(jobs)} siste — live-oppdatering</span>")
    for job in jobs:
        st.markdown(_job_card_html(job), unsafe_allow_html=True)


def render_send() -> None:
    st.markdown("### 📥 Send nye videoer")
    st.caption("Lime inn lenker fra YouTube, TikTok eller Instagram (én per linje) – eller dra inn lokale videofiler. De transkriberes og analyseres automatisk av worker-en.")

    urls_text = st.text_area(
        "Lenker",
        height=110,
        placeholder="https://www.tiktok.com/@bruker/video/123…\nhttps://www.youtube.com/watch?v=…",
    )
    files = st.file_uploader(
        "Eller dra inn videofiler",
        type=["mp4", "mov", "mkv", "webm", "m4v", "avi", "m4a", "mp3", "wav", "ogg"],
        accept_multiple_files=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    opt_transcribe = c1.toggle("🎧 Transkriber (Whisper)", value=True)
    opt_ocr = c2.toggle("📝 Tekst på skjerm (OCR)", value=True)
    opt_vision = c3.toggle("👁 Visjonsbeskrivelse", value=False)
    opt_embed = c4.toggle("🔢 Semantisk søk", value=True)

    urls = [ln.strip() for ln in urls_text.splitlines() if ln.strip()]
    n = len(urls) + len(files or [])
    cost_text = "0 kr (lokal Ollama)" if Config.USE_OLLAMA else "API-kostnad (se `cost`)"
    st.markdown(f"⏱ {n} video(er) i kø &nbsp;·&nbsp; 💰 {cost_text} &nbsp;·&nbsp; worker-en kjører når den er i gang")

    if st.button("🚀 Send videoer", type="primary", use_container_width=True, disabled=n == 0):
        bad = []
        added = 0
        options = {"transcribe": opt_transcribe, "ocr": opt_ocr, "vision": opt_vision, "embed": opt_embed}
        for url in urls:
            platform = detect_platform(url)
            if platform is None:
                bad.append(url)
                continue
            enqueue_job("web", url, platform=platform, options=options, data_dir=Config.DATA_DIR)
            added += 1
        for upload in files or []:
            import uuid as _uuid

            suffix = Path(upload.name).suffix.lower() or ".mp4"
            dest = Config.DATA_DIR / "uploads" / f"{_uuid.uuid4().hex}{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(upload.getvalue())
            enqueue_job("web", str(dest), platform="local", title=Path(upload.name).stem,
                        options=options, data_dir=Config.DATA_DIR)
            added += 1
        if bad:
            st.warning("Ukjente lenker hoppet over: " + ", ".join(bad))
        if added:
            st.success(f"{added} video(er) sendt til køen.")
        st.rerun()

    st.markdown("---")
    _live_queue()


# ---------------------------------------------------------------------------
# Page: Bibliotek
# ---------------------------------------------------------------------------
def _card_html(video, analysis: dict | None) -> str:
    thumb_cls = {"youtube": "thumb-yt", "tiktok": "thumb-tt", "instagram": "thumb-ig"}.get(
        video.platform, "thumb-local"
    )
    if analysis:
        foot = (
            f'{_rec_chip(analysis["recommendation"])} '
            f'<span>{analysis["estimated_key_watch_minutes"]:.0f} nøkkelmin</span>'
        )
        scores = (
            f'<span>Rel {_score_dots(analysis["relevance_score"])}</span>'
            f'<span>Dyb {_score_dots(analysis["depth_score"])}</span>'
        )
    elif video.status.value in ("discovered", "transcript_pending"):
        foot = "<span>Venter i pipeline…</span>"
        scores = ""
    elif video.status.value == "transcript_unavailable":
        foot = "<span>Ingen transkript</span>"
        scores = ""
    else:
        foot = "<span>Analyseres…</span>"
        scores = ""
    return (
        f'<div class="vke-card">'
        f'<div class="vke-thumb {thumb_cls}">{_plat_badge(video.platform).replace("plat ", "plat vke-plat ")}'
        f'<span class="vke-play">▶</span>'
        f'<span class="vke-dur">{video.duration_formatted}</span></div>'
        f'<div class="vke-card-body">'
        f'<div class="vke-card-title">{video.title or video.video_id}</div>'
        f'<div class="vke-card-chan">{video.channel or "—"} · {video.platform}</div>'
        f'<div class="vke-scores">{scores}</div>'
        f'<div class="vke-card-foot">{foot}</div>'
        f'</div></div>'
    )


def render_library() -> None:
    st.markdown("### 📚 Bibliotek")
    stats = get_progress_stats(data_dir=Config.DATA_DIR)
    st.caption(f'{stats["total_videos"]} videoer · {stats["transcripts_ok"]} transkribert · {stats["analyses_ok"]} analysert · {stats["total_duration_formatted"]}')

    videos = get_all_videos(data_dir=Config.DATA_DIR)
    analyses = {a["video_id"]: a for a in _all_analyses()}

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.6])
    plat_f = f1.selectbox("Plattform", ["Alle", "YouTube", "TikTok", "Instagram", "Lokale"])
    status_f = f2.selectbox(
        "Status",
        ["Alle"] + [s.value for s in VideoStatus],
        format_func=lambda s: s if s == "Alle" else s,
    )
    sort_f = f3.selectbox("Sorter", ["Nyeste", "Eldste", "Tittel A–Å", "Skår (høy→lav)"])
    min_score = f4.selectbox("Mind. relevansskår", ["Alle", "2/5", "3/5", "4/5", "5/5"])

    plat_map = {"YouTube": "youtube", "TikTok": "tiktok", "Instagram": "instagram", "Lokale": "local"}
    filtered = [
        v for v in videos
        if (plat_f == "Alle" or plat_map.get(plat_f) == v.platform)
        and (status_f == "Alle" or v.status.value == status_f)
    ]
    if min_score != "Alle":
        threshold = int(min_score[0])
        filtered = [
            v for v in filtered
            if analyses.get(v.video_id, {}).get("relevance_score", 0) >= threshold
        ]
    if sort_f == "Nyeste":
        filtered.sort(key=lambda v: v.created_at, reverse=True)
    elif sort_f == "Eldste":
        filtered.sort(key=lambda v: v.created_at)
    elif sort_f == "Tittel A–Å":
        filtered.sort(key=lambda v: v.title.lower())
    elif sort_f == "Skår (høy→lav)":
        filtered.sort(
            key=lambda v: analyses.get(v.video_id, {}).get("relevance_score", 0),
            reverse=True,
        )

    if not filtered:
        st.info("Ingen videoer treffet filteret. Start med «Send»-fanen.")

    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for col, v in zip(cols, filtered[i : i + 3]):
            with col:
                st.markdown(_card_html(v, analyses.get(v.video_id)), unsafe_allow_html=True)
                if st.button("Åpne", key=f"open_{v.video_id}", use_container_width=True):
                    st.session_state["selected_video"] = v.video_id
                    st.rerun()

    selected_id = st.session_state.get("selected_video")
    if selected_id:
        selected = next((v for v in videos if v.video_id == selected_id), None)
        if selected:
            st.markdown("---")
            st.markdown("#### Detaljer")
            _render_video_detail(selected, analyses.get(selected.video_id))
            if st.button("✕ Lukk detaljer"):
                st.session_state.pop("selected_video", None)
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Søk
# ---------------------------------------------------------------------------
def render_search() -> None:
    st.markdown("### 🔎 Søk i transkripter")
    st.caption("Nøkkelordsøk (SQLite FTS5) og, hvis du har kjørt `embed`, semantisk søk etter betydning.")
    query = st.text_input("Søksterm", "", label_visibility="collapsed")
    if not query.strip():
        st.info("Skriv et søksterm for å søke i transkriptene.")
        return

    st.markdown("#### Nøkkelordsøk")
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

    from video_knowledge_extractor.embeddings import embeddings_configured, search_embeddings

    if embeddings_configured():
        st.markdown("#### Semantisk søk (betydning)")
        try:
            semantic = search_embeddings(query, data_dir=Config.DATA_DIR, top_k=10)
        except Exception:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# Page: Syntese
# ---------------------------------------------------------------------------
def render_synthesis() -> None:
    st.markdown("### 🧩 Syntese av samlingen")
    st.caption("Én samlet analyse: felles tema, motsetninger og unik kunnskap fra flere videoer.")

    if st.button("🔄 Generer / oppdater syntese"):
        with st.spinner("Genererer syntese (modellkall)..."):
            try:
                run_synthesis(force=True)
            except Exception as exc:  # noqa: BLE001
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

    st.markdown("#### Samlet tema")
    st.write(synth.get("theme_no", ""))
    st.markdown("#### Felles tema (overlapp)")
    for t in synth.get("shared_topics", []) or ["—"]:
        st.write(f"- {t}")
    st.markdown("#### Motsetninger")
    cons = synth.get("contradictions", [])
    if cons:
        for c in cons:
            st.write(f"- **{c.get('topic')}** ({', '.join(c.get('videos', []))}): {c.get('note')}")
    else:
        st.write("Ingen tydelige motsetninger.")
    st.markdown("#### Unik kunnskap per video")
    uniq = synth.get("unique_knowledge", [])
    if uniq:
        for u in uniq:
            st.write(f"- **{u.get('video_id')}**: {u.get('knowledge')}")
    else:
        st.write("—")
    st.markdown("#### Mangler (gaps)")
    gaps = synth.get("gaps", [])
    if gaps:
        for g in gaps:
            st.write(f"- {g}")
    else:
        st.write("—")
    st.markdown("#### Anbefalt sesjonsrekkefølge")
    order = synth.get("watch_order", [])
    if order:
        for w in order:
            st.write(f"- **{w.get('video_id')}**: {w.get('reason')}")
    else:
        st.write("—")
    if synth.get("recommendation"):
        st.markdown("#### Anbefaling")
        st.write(synth["recommendation"])
    st.caption(f"Generert: {synth.get('created_at', '')} · modell: {synth.get('llm_model', '')} · {len(covered)} videoer")


# ---------------------------------------------------------------------------
# Page: Ranking
# ---------------------------------------------------------------------------
def render_ranking() -> None:
    st.markdown("### 🏆 Ranking")
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
        st.write(
            f"{rank}. **{title}** — verdi: {score:.1f} "
            f"(praktisk {a['practical_score']}, relevans {a['relevance_score']})"
        )


# ---------------------------------------------------------------------------
# Page: Eksport
# ---------------------------------------------------------------------------
def render_exports() -> None:
    st.markdown("### 📤 Eksport")
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
                    "platform": v.platform,
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
            "(Obsidian/OpenCode-vennlig, med kildelenker)."
        )


# ---------------------------------------------------------------------------
def main() -> None:
    header_mode = "Lokal · Ollama" if Config.USE_OLLAMA else f"API · {Config.LLM_MODEL}"
    st.markdown(
        '<div class="vke-header">'
        '<div class="vke-logo"><span class="vke-logo-badge">🧠</span>VKE</div>'
        f'<div class="vke-status"><span class="vke-dot"></span>{header_mode}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    tab_send, tab_lib, tab_search, tab_synth, tab_rank, tab_export = st.tabs(
        ["📥 Send", "📚 Bibliotek", "🔎 Søk", "🧩 Syntese", "🏆 Ranking", "📤 Eksport"]
    )
    with tab_send:
        render_send()
    with tab_lib:
        render_library()
    with tab_search:
        render_search()
    with tab_synth:
        render_synthesis()
    with tab_rank:
        render_ranking()
    with tab_export:
        render_exports()


main()
