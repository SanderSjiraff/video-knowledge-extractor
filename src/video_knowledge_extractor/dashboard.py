"""
VKE – Video Knowledge Extractor
Streamlit frontend
"""

from __future__ import annotations

import base64
import json
import math
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_knowledge_extractor.config import Config
from video_knowledge_extractor.db import (
    get_all_videos,
    get_connection,
    get_progress_stats,
    get_transcript,
    init_db,
    load_synthesis,
    search_transcripts,
)
from video_knowledge_extractor.discovery import detect_platform
from video_knowledge_extractor.jobs import (
    cancel_job,
    count_finished_jobs,
    enqueue_job,
    list_active_jobs,
    list_jobs,
    pin_job,
    retry_job,
    unpin_job,
)
from video_knowledge_extractor.models import VideoStatus
from video_knowledge_extractor.notes import export_markdown
from video_knowledge_extractor.orchestrator import run_synthesis

st.set_page_config(
    page_title="VKE – Video Knowledge Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
Config.ensure_dirs()
init_db(data_dir=Config.DATA_DIR)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────


def _ensure(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ensure("page", "oversikt")
_ensure("lib_page", 1)
_ensure("lib_tab", "alle")
if st.session_state.page in ("syntese", "ranking"):
    st.session_state.page = "innsikt"
elif st.session_state.page == "sok":
    st.session_state.page = "bibliotek"

PAGES = [
    ("oversikt", "Oversikt"),
    ("bibliotek", "Bibliotek"),
    ("ko", "Kø"),
    ("innsikt", "Innsikt"),
    ("eksport", "Eksport"),
]

STAGE_LABELS = {
    "queued": "I kø",
    "discovering": "Henter video",
    "transcribing": "Transkriberer",
    "ocr": "OCR",
    "vision": "Visjoner",
    "analyzing": "Analyserer",
    "embedding": "Indekserer",
    "done": "Ferdig",
    "failed": "Feilet",
    "cancelled": "Avbrutt",
}

# ─────────────────────────────────────────────────────────────────
# ICONS (inline SVG)
# ─────────────────────────────────────────────────────────────────

NAV_ICONS = {
    "oversikt": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "bibliotek": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "ko": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "innsikt": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "eksport": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
}


def _mask_svg(inner):
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        f"{inner}</svg>"
    )
    return urllib.parse.quote(svg, safe="")


def _mask_rule(key):
    e = _mask_svg(NAV_ICONS[key])
    return f"""
[data-testid="stSidebar"] .st-key-nav_{key} button::before {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    flex-shrink: 0;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,{e}") center / contain no-repeat;
    mask: url("data:image/svg+xml,{e}") center / contain no-repeat;
}}
[data-testid="stSidebar"] .st-key-nav_{key} button[kind="primary"]::before {{
    background-color: #388bfd;
}}
"""


NAV_ICONS_CSS = "".join(_mask_rule(k) for k, _ in PAGES)

_SEARCH_ICON_MASK = _mask_svg('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.8-3.8"/>')
_UPLOAD_ICON_MASK = _mask_svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>')


def _plat_icon(p):
    if p == "youtube":
        return (
            '<svg width="16" height="16" viewBox="0 0 16 16" style="border-radius:4px;margin-right:7px;flex-shrink:0;">'
            '<rect width="16" height="16" rx="4" fill="#ff0000"/>'
            '<path d="M6.3 4.6 11.4 8l-5.1 3.4z" fill="#fff"/></svg>'
        )
    if p == "tiktok":
        return (
            '<svg width="16" height="16" viewBox="0 0 16 16" style="border-radius:4px;margin-right:7px;flex-shrink:0;">'
            '<rect width="16" height="16" rx="4" fill="#010101"/>'
            '<circle cx="6.9" cy="10.7" r="2" fill="#fff"/>'
            '<path d="M8.7 10.7V3.4l3.5 1.1v1.8l-1.9-.6" fill="none" stroke="#fff" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
    if p == "instagram":
        return (
            '<svg width="16" height="16" viewBox="0 0 16 16" style="border-radius:4px;margin-right:7px;flex-shrink:0;">'
            '<defs><linearGradient id="ig" x1="0" y1="1" x2="1" y2="0">'
            '<stop offset="0" stop-color="#f9ce34"/><stop offset=".5" stop-color="#ee2a7b"/><stop offset="1" stop-color="#6228d7"/>'
            "</linearGradient></defs>"
            '<rect width="16" height="16" rx="4" fill="url(#ig)"/>'
            '<rect x="3.6" y="3.6" width="8.8" height="8.8" rx="2.6" fill="none" stroke="#fff" stroke-width="1.3"/>'
            '<circle cx="8" cy="8" r="2" fill="none" stroke="#fff" stroke-width="1.3"/>'
            '<circle cx="11.1" cy="4.9" r=".9" fill="#fff"/></svg>'
        )
    if p == "github":
        return (
            '<svg width="16" height="16" viewBox="0 0 16 16" style="border-radius:4px;margin-right:7px;flex-shrink:0;">'
            '<rect width="16" height="16" rx="4" fill="#24292f"/>'
            '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" fill="#fff"/>'
            "</svg>"
        )
    return (
        '<svg width="16" height="16" viewBox="0 0 16 16" style="border-radius:4px;margin-right:7px;flex-shrink:0;">'
        '<rect width="16" height="16" rx="4" fill="#30363d"/>'
        '<path d="M8 3.8 11.8 8 8 12.2 4.2 8z" fill="#8b949e"/></svg>'
    )


# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
/* === FOUNDATION === */
html, body, .stApp {{
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}}
[data-testid="stVerticalBlock"] {{ gap: 10px !important; }}
[data-testid="stHorizontalBlock"] {{ gap: 10px !important; }}

/* === SIDEBAR === */
[data-testid="stSidebar"] {{
    background: #10151c !important;
    border-right: 1px solid #1f2733 !important;
    width: 272px !important;
    min-width: 272px !important;
    max-width: 272px !important;
}}
[data-testid="stSidebar"] .block-container {{ padding: 22px 14px 16px 14px !important; }}
[data-testid="stSidebar"] .stButton {{ width: 100% !important; margin-bottom: 4px !important; }}
[data-testid="stSidebar"] .stButton > button {{
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    height: 42px !important;
    padding-left: 13px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #8b949e !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(139,148,158,.08) !important;
    color: #c9d1d9 !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: rgba(56,139,253,.12) !important;
    color: #f0f6fc !important;
    border-left: 3px solid #388bfd !important;
    border-radius: 0 8px 8px 0 !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stButton > button .stMarkdownContainer,
[data-testid="stSidebar"] .stButton > button .stMarkdownContainer p {{ margin: 0 !important; line-height: 1.25 !important; }}
{NAV_ICONS_CSS}

[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tabmark) {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}

/* === TYPOGRAPHY === */
h1 {{
    color: #f0f6fc !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    margin: 4px 0 2px 0 !important;
    line-height: 1.1 !important;
}}
.lead {{ color: #8b949e; font-size: 15px; margin: 0 0 16px 0; }}
.sub-lead {{ color: #6e7681; font-size: 13px; margin: 0 0 14px 0; }}
.sec-title {{ font-size: 18px; font-weight: 700; color: #f0f6fc; }}
.secrow {{ display: flex; align-items: center; gap: 12px; padding-top: 6px; }}
.count-pill {{
    background: #21262d; color: #c9d1d9; font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 12px;
}}
.panel-title {{ font-size: 17px; font-weight: 700; color: #f0f6fc; margin-bottom: 2px; }}
.panel-desc {{ font-size: 13px; color: #8b949e; margin-bottom: 14px; }}


/* === STATS BAR === */
.stats-bar {{
    display: flex; align-items: center; gap: 24px;
    background: #131a22; border: 1px solid #26303b; border-radius: 10px;
    padding: 13px 20px;
}}
.stats-bar .stat {{ font-size: 14px; color: #8b949e; }}
.stats-bar .stat b {{ color: #f0f6fc; font-weight: 700; }}
.stats-bar .sep {{ width: 1px; height: 20px; background: #30363d; }}

/* === BUTTONS === */
.stButton > button[kind="primary"] {{
    background: #1f6feb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    height: 40px !important;
}}
.stButton > button[kind="primary"]:hover {{ background: #388bfd !important; }}
.stButton > button[kind="secondary"] {{
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    height: 40px !important;
}}
.stButton > button[kind="secondary"]:hover {{ background: #30363d !important; }}
.stButton > button[kind="tertiary"] {{
    color: #388bfd !important;
    background: transparent !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 0 6px !important;
    font-size: 13px !important;
}}

/* === TABS (borderless wrapper) === */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tabmark) .stButton {{ width: 100% !important; margin: 0 !important; }}
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tabmark) .stButton > button {{
    height: 40px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: #8b949e !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0 4px !important;
}}
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tabmark) .stButton > button:hover {{
    background: transparent !important;
    color: #c9d1d9 !important;
}}
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tabmark) .stButton > button[kind="primary"] {{
    color: #f0f6fc !important;
    background: transparent !important;
    border-bottom: 2px solid #388bfd !important;
    font-weight: 600 !important;
}}

/* === INPUTS === */
[data-baseweb="input"] {{
    position: relative;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    height: 40px !important;
}}
[data-baseweb="input"]::before {{
    content: "";
    position: absolute;
    left: 13px; top: 50%; transform: translateY(-50%);
    width: 15px; height: 15px;
    background: #6e7681;
    z-index: 1;
    -webkit-mask: url("data:image/svg+xml,{_SEARCH_ICON_MASK}") center / contain no-repeat;
    mask: url("data:image/svg+xml,{_SEARCH_ICON_MASK}") center / contain no-repeat;
}}
[data-baseweb="input"] input {{
    padding-left: 38px !important;
    color: #c9d1d9 !important;
    font-size: 14px !important;
}}
[data-baseweb="input"] input::placeholder {{ color: #565e67; }}
[data-baseweb="textarea"] {{
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}}
[data-baseweb="textarea"] textarea {{ font-size: 14px; }}
[data-baseweb="select"] > div {{
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    height: 40px !important;
    color: #c9d1d9 !important;
    font-size: 14px !important;
}}

/* === FILE UPLOADER (button-like, aligned with Analyser button) === */
[data-testid="stFileUploader"] > section {{
    position: relative;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: 40px !important;
    min-height: 40px !important;
    box-sizing: border-box !important;
    padding: 0 12px !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    cursor: pointer;
}}
[data-testid="stFileUploader"] > section::before {{
    content: "Last opp filer";
    font-size: 14px; color: #c9d1d9; font-weight: 500;
}}
[data-testid="stFileUploader"] > section::after {{
    content: "";
    width: 15px; height: 15px;
    margin-left: 8px;
    flex-shrink: 0;
    background: #8b949e;
    -webkit-mask: url("data:image/svg+xml,{_UPLOAD_ICON_MASK}") center / contain no-repeat;
    mask: url("data:image/svg+xml,{_UPLOAD_ICON_MASK}") center / contain no-repeat;
}}
[data-testid="stFileUploader"] > section > * {{
    position: absolute !important;
    inset: 0 !important;
    opacity: 0 !important;
    cursor: pointer;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{ display: none !important; }}

/* === EXPANDER === */
[data-testid="stExpander"] {{
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{ font-size: 13px; color: #c9d1d9 !important; }}

/* === TABLE === */
.tbl-header {{
    display: flex; align-items: center;
    font-size: 13px; color: #8b949e; font-weight: 500;
    padding: 4px 0;
    min-height: 60px;
}}
.cell {{
    display: flex; align-items: center;
    padding: 4px 0;
    min-height: 60px;
}}
.tbline {{ border-bottom: 1px solid #1d2530; }}

/* Table rows: uniform height + one full-width line per row (no gaps) */
.tblbox {{ display: none; }}
[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tblbox) {{ gap: 0 !important; }}
[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tblbox) > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{
    border-bottom: 1px solid #1d2530;
}}
[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tblbox) > [data-testid="stLayoutWrapper"]:last-of-type > [data-testid="stHorizontalBlock"] {{
    border-bottom: none;
}}
.vthumb {{
    width: 46px; height: 32px; border-radius: 6px; object-fit: cover;
    margin-right: 12px; background: #21262d; flex-shrink: 0;
}}
.vthumb-ph {{
    width: 46px; height: 32px; border-radius: 6px; margin-right: 12px;
    background: #21262d; display: flex; align-items: center; justify-content: center;
    color: #565e67; font-size: 9px; flex-shrink: 0;
}}
.vtext {{ min-width: 0; }}
.vtitle {{
    font-size: 14px; color: #e6edf3; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.cbx {{
    width: 16px; height: 16px;
    border: 1.5px solid #3d444d; border-radius: 4px;
    display: inline-block; flex-shrink: 0;
}}
.vchan {{ font-size: 12px; color: #8b949e; margin-top: 1px; }}
.plat {{ display: flex; align-items: center; font-size: 13px; color: #c9d1d9; }}
.dots {{ color: #484f58; font-size: 15px; letter-spacing: 2px; }}
.pageinfo {{ color: #8b949e; font-size: 13px; padding-top: 8px; }}
[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.tblbox) .stButton > button {{
    height: 34px !important;
    padding: 0 10px !important;
    font-size: 13px !important;
}}

/* === BADGES === */
.badge-status {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
}}
.status-done {{ color: #3fb950; background: rgba(35,197,94,.13); }}
.status-run  {{ color: #58a6ff; background: rgba(56,139,253,.12); }}
.status-wait {{ color: #8b949e; background: rgba(139,148,158,.12); }}
.status-err  {{ color: #f85149; background: rgba(242,72,65,.12); }}
.status-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.badge-rating {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 6px;
}}
.rat-watch {{ color: #3fb950; background: rgba(35,197,94,.1); }}
.rat-skim  {{ color: #d29922; background: rgba(210,153,34,.12); }}
.rat-skip  {{ color: #f85149; background: rgba(242,72,65,.1); }}

/* === QUEUE === */
.q-empty {{ text-align: center; padding: 34px 16px 26px 16px; }}
.qe-title {{ font-size: 16px; font-weight: 700; color: #e6edf3; margin-top: 14px; }}
.qe-desc {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
.qe-foot {{
    text-align: right; color: #388bfd; font-size: 13px; font-weight: 500;
    padding-top: 14px; border-top: 1px solid #26303b; margin-top: 6px;
}}
.q-card {{
    background: rgba(0,0,0,.18);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
}}
.q-card .title {{
    font-weight: 700; font-size: 13px; color: #e6edf3;
    display: flex; gap: 8px; align-items: center;
}}
.qrow {{
    display: flex; align-items: center; gap: 12px;
    padding: 10px 4px;
    border-bottom: 1px solid #1d2530;
}}
.qrow:last-child {{ border-bottom: none; }}
.qrow .qtitle {{
    flex: 1; min-width: 0;
    font-size: 14px; color: #e6edf3; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.qrow .qsub {{ font-size: 12px; color: #8b949e; margin-top: 1px; }}
.qrow .qpos {{
    width: 16px; flex-shrink: 0; text-align: center;
    font-size: 12px; font-weight: 600; color: #8b949e;
}}
.qprog {{
    width: 90px; height: 4px; flex-shrink: 0;
    background: #21262d; border-radius: 2px; overflow: hidden;
}}
.qprog > div {{ height: 100%; background: #388bfd; border-radius: 2px; }}
.oqrow {{
    display: flex; align-items: center; gap: 8px;
    padding: 6px 2px;
    border-bottom: 1px solid #1d2530;
}}
.oqrow:last-child {{ border-bottom: none; }}
.oqrow .oqpos {{
    width: 14px; flex-shrink: 0; text-align: center;
    font-size: 12px; font-weight: 600; color: #8b949e;
}}
.oqrow .oqtitle {{
    flex: 1; min-width: 0;
    font-size: 13px; color: #e6edf3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.oqrow .oqprog {{
    width: 46px; height: 3px; flex-shrink: 0;
    background: #21262d; border-radius: 2px; overflow: hidden;
}}
.oqrow .oqprog > div {{ height: 100%; background: #388bfd; }}
.tchip {{
    display: inline-block; max-width: 230px;
    background: rgba(56,139,253,.10); border: 1px solid rgba(56,139,253,.35);
    color: #c9d1d9; font-size: 12px; padding: 3px 10px; border-radius: 999px;
    margin: 0 6px 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.tchip-c {{
    display: inline-block; max-width: 180px;
    background: rgba(139,148,158,.10); border: 1px solid rgba(139,148,158,.30);
    color: #8b949e; font-size: 11px; padding: 2px 9px; border-radius: 999px;
    margin: 0 6px 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.eta-box {{
    display: flex; align-items: center; gap: 10px;
    background: rgba(56,139,253,.08); border: 1px solid rgba(56,139,253,.40);
    border-radius: 10px; padding: 8px 14px; margin-bottom: 12px;
}}
.eta-ico {{ font-size: 17px; line-height: 1.2; }}
.eta-lbl {{ color: #8b949e; font-size: 11px; }}
.eta-val {{ color: #e6edf3; font-size: 14px; font-weight: 700; }}
.tmthumb .vthumb, .tmthumb .vthumb-ph {{ margin-right: 0; }}
.qlast {{ font-size: 12px; color: #8b949e; flex-shrink: 0; width: 92px; text-align: right; }}
</style>
""",
    unsafe_allow_html=True,
)

# Additional styles for the reference layout; native widgets retain keyboard support.
# The script also removes any saved "sidebar hidden" state (set when the old
# collapse button was clicked), so the sidebar always starts expanded.
st.markdown(
    "<style>" + Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8") + "</style>"
    "<script>(function(){try{"
    "Object.keys(localStorage)"
    ".filter(function(k){return k.indexOf('stSidebarCollapsed-')===0;})"
    ".forEach(function(k){localStorage.removeItem(k);});"
    "}catch(e){}})();</script>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────


def _plat_label(p):
    return {"youtube": "YouTube", "tiktok": "TikTok", "instagram": "Instagram", "github": "GitHub", "local": "Fil"}.get(p, "Video")


def _status_badge(s):
    if s == "analysis_ok":
        return '<span class="badge-status status-done"><span class="status-dot" style="background:#3fb950;"></span>Analysert</span>'
    if s == "transcript_ok":
        return '<span class="badge-status status-done"><span class="status-dot" style="background:#3fb950;"></span>Klar</span>'
    if s in ("analysis_pending", "transcript_pending", "discovering", "transcribing", "analyzing", "embedding"):
        return '<span class="badge-status status-run"><span class="status-dot" style="background:#58a6ff;"></span>Behandler</span>'
    if s == "analysis_failed":
        return '<span class="badge-status status-err"><span class="status-dot" style="background:#f85149;"></span>Feilet</span>'
    return f'<span class="badge-status status-wait"><span class="status-dot" style="background:#8b949e;"></span>{s}</span>'


def _rec_badge(rec):
    label, cls = {"watch": ("★ Se", "rat-watch"), "skim": ("★ Skum", "rat-skim"), "skip": ("★ Slep", "rat-skip")}.get(
        rec or "", ("—", "")
    )
    return f'<span class="badge-rating {cls}">{label}</span>'


def _fmt_dur(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h} t {m} min"
    return f"{m} min {s}s"


def _fmt_date(dt):
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return str(dt)
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    if delta.days == 0:
        return f"I dag, {dt.strftime('%H:%M')}"
    if delta.days == 1:
        return f"I går, {dt.strftime('%H:%M')}"
    if delta.days < 7:
        return f"{delta.days} dager siden"
    return dt.strftime("%d. %b %Y")


def _thumb_html(v, ph_text="VIDEO"):
    # Local thumbnail (extracted by the worker) beats the remote placeholder.
    t = Config.THUMBS_DIR / f"{v.video_id}.jpg"
    if t.exists() and t.stat().st_size > 0:
        b64 = base64.b64encode(t.read_bytes()).decode("ascii")
        return f'<img class="vthumb" src="data:image/jpeg;base64,{b64}" alt="">'
    if v.platform == "youtube":
        return (
            f'<img class="vthumb" src="https://i.ytimg.com/vi/{v.video_id}/mqdefault.jpg" '
            f"onerror='this.outerHTML=\"<div class=vthumb-ph>{ph_text}</div>\"'>"
        )
    return f'<div class="vthumb-ph">{ph_text}</div>'


def _video_cell(v, with_channel=True):
    chan = f'<div class="vchan">{v.channel or "—"}</div>' if with_channel else ""
    return (
        '<div class="cell">'
        + _thumb_html(v)
        + f'<div class="vtext"><div class="vtitle">{v.title or v.video_id}</div>{chan}</div>'
        + '</div>'
    )


def _plat_cell(v):
    return f'<div class="cell"><span class="plat">{_plat_icon(v.platform)}{_plat_label(v.platform)}</span></div>'


def _dur_cell(v):
    dur = "—" if (v.platform or "youtube") == "github" else v.duration_formatted
    return f'<div class="cell" style="color:#8b949e;font-size:13px;">{dur}</div>'


def _status_cell(v):
    return f'<div class="cell">{_status_badge(v.status.value)}</div>'


def _load_analyses():
    with get_connection(Config.DATA_DIR) as conn:
        rows = conn.execute("SELECT * FROM analyses").fetchall()
    out = {}
    for r in rows:
        out[r["video_id"]] = {
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
    return out


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
def _sidebar():
    st.sidebar.markdown(
        """
        <div style="padding: 4px 0 20px; margin-bottom: 12px;">
            <div style="font-size: 32px; font-weight: 800; color: #f0f6fc; letter-spacing: -0.5px;">VKE</div>
            <div style="font-size: 13px; color: #aebfd5; margin-top: 0; white-space: nowrap;">Video Knowledge Extractor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for key, label in PAGES:
        active = st.session_state.page == key
        if st.sidebar.button(
            label,
            key=f"nav_{key}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = key
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────
def _overview():
    st.markdown("<h1>Oversikt</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Gjør videoene dine til kunnskap du finner igjen.</p>", unsafe_allow_html=True)

    stats = get_progress_stats(data_dir=Config.DATA_DIR)
    st.markdown(
        f'''<div class="stats-bar">
            <span class="stat"><b>{stats["total_videos"]}</b> videoer</span><span class="sep"></span>
            <span class="stat"><b>{stats["analyses_ok"]}</b> analysert</span><span class="sep"></span>
            <span class="stat"><b>{_fmt_dur(stats["total_duration_seconds"])}</b> innhold</span>
        </div>''',
        unsafe_allow_html=True,
    )

    with st.container(horizontal=True, key="overview_panels", gap="medium"):
        with st.container(border=True, key="add_panel", height="stretch"):
            st.markdown('<div class="panel-title">Legg til videoer</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-desc">Lim inn lenker fra YouTube, TikTok eller Instagram.</div>',
                unsafe_allow_html=True,
            )
            urls = st.text_area(
                "URLer",
                placeholder="Én lenke per linje (YouTube, TikTok, Instagram eller GitHub-repo)",
                label_visibility="collapsed",
                height=102,
                key="ov_u",
            )
            b1, spacer, b2 = st.columns([1.4, 0.4, 1.7])
            with b1:
                files = st.file_uploader(
                    "Last opp filer",
                    type=["mp4", "mov", "mkv", "m4a", "mp3"],
                    accept_multiple_files=True,
                    key="ov_f",
                    label_visibility="collapsed",
                )
            with b2:
                url_list = [l.strip() for l in (urls or "").splitlines() if l.strip()]
                if st.button("Analyser videoer", type="primary", use_container_width=True):
                    import uuid as _u

                    bad, added = [], 0
                    opts = {"transcribe": True, "ocr": False, "vision": False, "embed": True}
                    for url in url_list:
                        plat = detect_platform(url)
                        if plat is None:
                            bad.append(url)
                            continue
                        enqueue_job("web", url, platform=plat, options=opts, data_dir=Config.DATA_DIR)
                        added += 1
                    for up in files or []:
                        dest = Config.DATA_DIR / "uploads" / f"{_u.uuid4().hex}{Path(up.name).suffix}"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(up.getvalue())
                        enqueue_job(
                            "web",
                            str(dest),
                            platform="local",
                            title=Path(up.name).stem,
                            options=opts,
                            data_dir=Config.DATA_DIR,
                        )
                        added += 1
                    if bad:
                        st.warning("Ukjente lenker: " + ", ".join(bad))
                    if added:
                        st.success(f"{added} video(er) i køen")
                        st.rerun()
            with st.expander("Analysevalg"):
                c1, c2, _ = st.columns(3)
                with c1:
                    st.checkbox("Transkriber", value=True, key="ov_t")
                with c2:
                    st.checkbox("OCR (tekst på skjerm)", value=False, key="ov_r")

        with st.container(border=True, key="queue_panel", height="stretch"):
            st.markdown('<div class="panel-title">Behandlingskø</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-desc">Øverst i lista behandles først (først inn – først ut).</div>',
                unsafe_allow_html=True,
            )
            @st.fragment(run_every="20s")
            def _ov_live():
                active = list_active_jobs(data_dir=Config.DATA_DIR)
                eta_html = _eta_box_html(active)
                if eta_html:
                    st.markdown(eta_html, unsafe_allow_html=True)
                if not active:
                    st.markdown(
                        """<div class="q-empty">
                            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#6e7681" stroke-width="1.1">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="m8.5 12.5 2.5 2.5 5-5.5" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <div class="qe-title">Alt er ferdig</div>
                            <div class="qe-desc">Ingen videoer venter på analyse.</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if st.button("Se siste behandlinger →", key="q1", type="tertiary", use_container_width=True):
                        st.session_state.page = "ko"
                        st.rerun()
                    return
                n_queued = sum(1 for j in active if j["status"] == "queued")
                n_running = len(active) - n_queued
                st.markdown(
                    f'<div class="qsub" style="padding:2px 2px 4px;">'
                    f'<b style="color:#e6edf3;">{n_queued}</b> venter &middot; '
                    f'<b style="color:#e6edf3;">{n_running}</b> behandles</div>',
                    unsafe_allow_html=True,
                )
                visible = active[:5]
                for i, job in enumerate(visible, start=1):
                    if i > 1:
                        st.divider()
                    st.markdown(_ov_queue_row(job, i), unsafe_allow_html=True)
                if len(active) > len(visible):
                    st.markdown(
                        f'<div class="qsub" style="padding:6px 2px 0;">'
                        f'+ {len(active) - len(visible)} flere i kø</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(_status_line(active), unsafe_allow_html=True)
                if st.button("Se hele køen →", key="q2", type="secondary", use_container_width=True):
                    st.session_state.page = "ko"
                    st.rerun()

            _ov_live()

    # Recent videos
    videos = get_all_videos(data_dir=Config.DATA_DIR)
    recent = sorted(videos, key=lambda v: v.created_at, reverse=True)[:6]
    if not recent:
        return

    with st.container(border=True, key="recent_table"):
        h1, h2 = st.columns([5, 1])
        with h1:
            st.markdown(
                f'<div class="secrow"><span class="sec-title">Sist lagt til</span>'
                f'<span class="count-pill">{len(videos)} videoer</span></div>',
                unsafe_allow_html=True,
            )
        with h2:
            if st.button("Se alle videoer →", key="sa", type="secondary"):
                st.session_state.page = "bibliotek"
                st.rerun()

        st.markdown('<div class="tblbox"></div>', unsafe_allow_html=True)
        ratios = [0.45, 4.4, 1.6, 1.0, 1.5, 1.5, 0.7]
        hc = st.columns(ratios)
        for col, h in zip(
            hc, ['<span class="cbx" style="opacity:.6;"></span>', "Video", "Plattform", "Varighet", "Status", "Lagt til", ""]
        ):
            with col:
                st.markdown(f'<div class="tbl-header">{h}</div>', unsafe_allow_html=True)

        for v in recent:
            rc = st.columns(ratios)
            with rc[0]:
                st.markdown('<div class="cell"><span class="cbx"></span></div>', unsafe_allow_html=True)
            with rc[1]:
                st.markdown(_video_cell(v, with_channel=False), unsafe_allow_html=True)
            with rc[2]:
                st.markdown(_plat_cell(v), unsafe_allow_html=True)
            with rc[3]:
                st.markdown(_dur_cell(v), unsafe_allow_html=True)
            with rc[4]:
                st.markdown(_status_cell(v), unsafe_allow_html=True)
            with rc[5]:
                st.markdown(
                    f'<div class="cell" style="color:#8b949e;font-size:13px;">{_fmt_date(v.created_at)}</div>',
                    unsafe_allow_html=True,
                )
            with rc[6]:
                st.markdown(
                    '<div class="cell" style="justify-content:flex-end;"><span class="dots">···</span></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────
# QUEUE (KØ)
# ─────────────────────────────────────────────────────────────────
def _job_title(j):
    t = (j.get("title") or "").strip()
    if t:
        return t
    url = j.get("url") or ""
    if url.startswith(("http://", "https://")):
        return url[:70] + ("…" if len(url) > 70 else "")
    return Path(url).stem or "Video"


def _job_row(j, show_progress, pos=None, show_date=True):
    label = STAGE_LABELS.get(j["status"], j["status"])
    if j["status"] == "queued":
        badge = '<span class="badge-status status-wait"><span class="status-dot" style="background:#8b949e;"></span>I kø</span>'
    elif j["status"] == "done":
        badge = '<span class="badge-status status-done"><span class="status-dot" style="background:#3fb950;"></span>Ferdig</span>'
    elif j["status"] == "failed":
        badge = '<span class="badge-status status-err"><span class="status-dot" style="background:#f85149;"></span>Feilet</span>'
    elif j["status"] == "cancelled":
        badge = '<span class="badge-status status-wait"><span class="status-dot" style="background:#8b949e;"></span>Avbrutt</span>'
    else:
        badge = f'<span class="badge-status status-run"><span class="status-dot" style="background:#58a6ff;"></span>{label}</span>'
    if show_progress:
        prog = max(0, min(100, int(j.get("progress") or 0)))
        prog_html = f'<div class="qprog"><div style="width:{prog}%"></div></div>'
    else:
        prog_html = '<div class="qprog" style="background:transparent;"></div>'
    sub = _plat_label(j.get("platform") or "youtube")
    if j.get("pinned") and j["status"] == "queued":
        sub += ' · <span style="color:#58a6ff;">📌 prioritert</span>'
    err = (j.get("error_message") or "")[:60]
    if j["status"] == "failed" and err:
        sub += f" · {err}"
    pos_html = f'<span class="qpos">{pos}.</span>' if pos is not None else ""
    date_html = f'<span class="qlast">{_fmt_date(j["created_at"])}</span>' if show_date else ""
    # One-line string on purpose: a whitespace-only line inside the HTML
    # would end the markdown HTML block and break the layout.
    return (
        f'<div class="qrow">{pos_html}{_plat_icon(j.get("platform") or "youtube")}'
        f'<div style="flex:1;min-width:0;">'
        f'<div class="qtitle">{_job_title(j)}</div>'
        f'<div class="qsub">{sub}</div>'
        f'</div>{badge}{prog_html}{date_html}</div>'
    )


def _ov_queue_row(j, pos):
    """Compact single-line queue row for the front page (active jobs only)."""
    if j["status"] == "queued":
        badge = '<span class="badge-status status-wait"><span class="status-dot" style="background:#8b949e;"></span>I kø</span>'
        prog = ""
    else:
        label = STAGE_LABELS.get(j["status"], j["status"])
        badge = f'<span class="badge-status status-run"><span class="status-dot" style="background:#58a6ff;"></span>{label}</span>'
        p = max(0, min(100, int(j.get("progress") or 0)))
        prog = f'<div class="oqprog"><div style="width:{p}%"></div></div>'
    return (
        f'<div class="oqrow"><span class="oqpos">{pos}.</span>'
        f'{_plat_icon(j.get("platform") or "youtube")}'
        f'<span class="oqtitle">{_job_title(j)}</span>'
        f'{badge}{prog}</div>'
    )


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return "ca. 1–2 min"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"ca. {minutes} min"
    hours = int(minutes // 60)
    return f"ca. {hours} t {minutes % 60:02d} min"


def _queue_eta_text(active: list[dict]) -> str:
    """Rough ETA: average done-job duration × number of queued jobs."""
    queued = sum(1 for j in active if j["status"] == "queued")
    if not queued:
        return "—"
    with get_connection(Config.DATA_DIR) as conn:
        rows = conn.execute(
            "SELECT created_at, updated_at FROM jobs WHERE status='done' "
            "ORDER BY updated_at DESC LIMIT 20"
        ).fetchall()
    durs = []
    for c, u in rows:
        try:
            d = (datetime.fromisoformat(u) - datetime.fromisoformat(c)).total_seconds()
        except Exception:
            continue
        if 0 < d < 6 * 3600:
            durs.append(d)
    if not durs:
        return "ukjent"
    return _fmt_eta((sum(durs) / len(durs)) * queued)


def _worker_status_text(active: list[dict]) -> str:
    """Heuristic worker liveness based on job update timestamps."""
    if not active:
        return "ledig (køen er tom)"
    running = [j for j in active if j["status"] != "queued"]
    if not running:
        return "venter på worker (ingen jobb i gang)"
    try:
        last = max(datetime.fromisoformat(j["updated_at"]) for j in running)
        age = (datetime.utcnow() - last).total_seconds()
    except Exception:
        return "aktiv"
    if age > 15 * 60:
        return f"⚠️ muligens død (sist aktiv {int(age // 60)} min siden)"
    return f"aktiv (sist aktiv {max(1, int(age // 60))} min siden)"


def _status_line(active: list[dict]) -> str:
    return (
        f'<div class="qsub" style="padding:4px 2px 0;">'
        f'Worker: {_worker_status_text(active)}</div>'
    )


def _eta_box_html(active: list[dict]) -> str:
    """Compact standalone box with the remaining queue time ('' when idle)."""
    if not active:
        return ""
    n_queued = sum(1 for j in active if j["status"] == "queued")
    val = _queue_eta_text(active) if n_queued else "Nesten ferdig — siste jobb er i gang"
    return (
        f'<div class="eta-box">'
        f'<div class="eta-ico">⏱</div>'
        f'<div><div class="eta-lbl">Gjenstående tid i kø</div>'
        f'<div class="eta-val">{val}</div></div>'
        f'</div>'
    )


def _eta_box():
    """The ETA box as its own auto-refreshing fragment (front page)."""
    @st.fragment(run_every="20s")
    def _live():
        html = _eta_box_html(list_active_jobs(data_dir=Config.DATA_DIR))
        if html:
            st.markdown(html, unsafe_allow_html=True)
    _live()


def _queue():
    st.markdown("<h1>Kø</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Videoer som venter på eller er under analyse.</p>", unsafe_allow_html=True)

    @st.fragment(run_every="20s")
    def _live():
        active = list_active_jobs(data_dir=Config.DATA_DIR)  # FIFO: eldste først
        finished = [
            j for j in list_jobs(limit=100, data_dir=Config.DATA_DIR)
            if j["status"] in ("done", "failed")
        ][:10]

        n_queued = sum(1 for j in active if j["status"] == "queued")
        n_running = len(active) - n_queued
        # Count every terminal job in the DB — the `finished` list above is
        # capped at 10 (it feeds the "Sist behandlet" panel below).
        n_finished = count_finished_jobs(data_dir=Config.DATA_DIR)
        st.markdown(
            f'''<div class="stats-bar">
                <span class="stat"><b>{n_queued}</b> venter i kø</span><span class="sep"></span>
                <span class="stat"><b>{n_running}</b> behandles nå</span><span class="sep"></span>
                <span class="stat"><b>{n_finished}</b> behandlet</span>
            </div>''',
            unsafe_allow_html=True,
        )
        eta_html = _eta_box_html(active)
        if eta_html:
            st.markdown(eta_html, unsafe_allow_html=True)
        st.markdown(_status_line(active), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="panel-title">Aktiv kø</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-desc">Nye videoer behandles nederst opp i rekkefølgen (først inn – først ut). '
                '<b>⬆</b> prioriter · <b>⬇</b> dra ned · <b>✕</b> avbryt · <b>↻</b> prøv igjen.</div>',
                unsafe_allow_html=True,
            )
            if not active:
                st.markdown(
                    """<div class="q-empty">
                        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#6e7681" stroke-width="1.1">
                            <circle cx="12" cy="12" r="10"/>
                            <path d="m8.5 12.5 2.5 2.5 5-5.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <div class="qe-title">Køen er tom</div>
                        <div class="qe-desc">Ingen videoer venter på analyse.</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                for i, j in enumerate(active):
                    if i:
                        st.divider()
                    rc1, rc2 = st.columns([23, 1], gap="small")
                    with rc1:
                        st.markdown(_job_row(j, show_progress=True), unsafe_allow_html=True)
                    with rc2:
                        if j["status"] == "queued":
                            ca, cb = st.columns(2)
                            with ca:
                                if j.get("pinned"):
                                    if st.button("⬇", key=f"pin_{j['job_id']}", type="tertiary", help="Dra ned igjen i køen (FIFO)"):
                                        unpin_job(j["job_id"], data_dir=Config.DATA_DIR)
                                        st.rerun()
                                else:
                                    if st.button("⬆", key=f"pin_{j['job_id']}", type="tertiary", help="Øverst i køen"):
                                        pin_job(j["job_id"], data_dir=Config.DATA_DIR)
                                        st.rerun()
                            with cb:
                                if st.button("✕", key=f"cx_{j['job_id']}", type="tertiary", help="Avbryt jobben"):
                                    cancel_job(j["job_id"], data_dir=Config.DATA_DIR)
                                    st.rerun()

        h1, h2 = st.columns([1, 1])
        with h1:
            st.markdown(
                f'<div class="secrow"><span class="sec-title">Sist behandlet</span>'
                f'<span class="count-pill">{len(finished)} siste</span></div>',
                unsafe_allow_html=True,
            )
        with h2:
            if st.button("Legg til videoer →", key="q_add", type="secondary"):
                st.session_state.page = "oversikt"
                st.rerun()

        if not finished:
            st.markdown('<p class="sub-lead" style="padding-top:10px;">Ingen behandlinger ennå.</p>', unsafe_allow_html=True)
            return

        with st.container(border=True):
            for i, j in enumerate(finished):
                if i:
                    st.divider()
                fc1, fc2 = st.columns([23, 1], gap="small")
                with fc1:
                    st.markdown(_job_row(j, show_progress=False), unsafe_allow_html=True)
                with fc2:
                    if st.button("↻", key=f"rt_{j['job_id']}", type="tertiary", help="Prøv igjen"):
                        retry_job(j["job_id"], data_dir=Config.DATA_DIR)
                        st.rerun()

    _live()


# ─────────────────────────────────────────────────────────────────
# LIBRARY
# ─────────────────────────────────────────────────────────────────
def _page_window(page, total):
    if total <= 6:
        return list(range(1, total + 1))
    start = max(1, min(page - 2, total - 4))
    return list(range(start, start + 5))


def _library():
    videos = get_all_videos(data_dir=Config.DATA_DIR)
    analyses = _load_analyses()
    stats = get_progress_stats(data_dir=Config.DATA_DIR)

    st.markdown("<h1>Bibliotek</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Alle videoene dine, med innsikt og nøkkelpunkter samlet.</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='sub-lead'>{len(videos)} videoer &nbsp;·&nbsp; {_fmt_dur(stats['total_duration_seconds'])}</p>",
        unsafe_allow_html=True,
    )

    # Filters
    f1, f2, f3, f4, f5 = st.columns([4.4, 1.6, 1.5, 1.4, 1.7])
    with f1:
        q = st.text_input(
            "Søk",
            placeholder="Søk etter tittel, kanal eller innhold",
            label_visibility="collapsed",
            key="ls",
        )
    with f2:
        plat_f = st.selectbox(
            "Plattform", ["Alle plattformer", "YouTube", "TikTok", "Instagram", "Lokale"],
            label_visibility="collapsed", key="lp",
        )
    with f3:
        status_f = st.selectbox(
            "Status", ["Alle statuser"] + [s.value for s in VideoStatus],
            label_visibility="collapsed", key="lst",
        )
    with f4:
        rec_f = st.selectbox(
            "Vurdering", ["Relevans", "Se", "Skum", "Slep"],
            label_visibility="collapsed", key="lr",
        )
    with f5:
        sort_f = st.selectbox(
            "Sorter", ["Nyeste først", "Eldste først", "Tittel A-Å", "Relevans (høy-lav)"],
            label_visibility="collapsed", key="lsr",
        )

    pmap = {"YouTube": "youtube", "TikTok": "tiktok", "Instagram": "instagram", "GitHub": "github", "Lokale": "local"}
    filtered = [
        v
        for v in videos
        if (plat_f == "Alle plattformer" or pmap.get(plat_f) == v.platform)
        and (status_f == "Alle statuser" or v.status.value == status_f)
    ]
    if rec_f != "Relevans":
        rec_map = {"Se": "watch", "Skum": "skim", "Slep": "skip"}
        filtered = [v for v in filtered if analyses.get(v.video_id, {}).get("recommendation") == rec_map[rec_f]]
    if q.strip():
        ql = q.strip().lower()
        content_hits = {hit["video_id"] for hit in search_transcripts(q.strip(), limit=len(videos) or 1, data_dir=Config.DATA_DIR)}
        filtered = [v for v in filtered if ql in (v.title or "").lower() or ql in (v.channel or "").lower()
                    or ql in analyses.get(v.video_id, {}).get("summary_no", "").lower() or v.video_id in content_hits]
    if st.session_state.lib_tab == "anbefalt":
        filtered = [v for v in filtered if analyses.get(v.video_id, {}).get("recommendation") == "watch"]
    elif st.session_state.lib_tab == "oppsummert":
        filtered = [v for v in filtered if v.video_id in analyses]
    if sort_f == "Nyeste først":
        filtered.sort(key=lambda v: v.created_at, reverse=True)
    elif sort_f == "Eldste først":
        filtered.sort(key=lambda v: v.created_at)
    elif sort_f == "Tittel A-Å":
        filtered.sort(key=lambda v: (v.title or "").lower())
    elif sort_f == "Relevans (høy-lav)":
        filtered.sort(key=lambda v: analyses.get(v.video_id, {}).get("relevance_score", 0), reverse=True)

    # Tabs (underline style)
    with st.container(border=True, key="library_tabs"):
        st.markdown('<div class="tabmark"></div>', unsafe_allow_html=True)
        tc = st.columns([1.2, 1.4, 1.6, 5.8])
        tab_labels = [
            ("alle", f"Alle videoer   ·   {len(videos)}"),
            ("anbefalt", "Anbefalt å se"),
            ("oppsummert", "Kort oppsummert"),
        ]
        for i, (k, label) in enumerate(tab_labels):
            with tc[i]:
                btn = "primary" if st.session_state.lib_tab == k else "secondary"
                if st.button(label, key=f"lt_{k}", type=btn, use_container_width=True):
                    st.session_state.lib_tab = k
                    st.rerun()

    per_page = 8
    total_pages = max(1, math.ceil(len(filtered) / per_page))
    if st.session_state.lib_page > total_pages:
        st.session_state.lib_page = 1
    page = st.session_state.lib_page
    start = (page - 1) * per_page
    items = filtered[start : start + per_page]

    # Table
    if items:
        with st.container(border=True, key="library_table"):
            st.markdown('<div class="tblbox"></div>', unsafe_allow_html=True)
            ratios = [0.45, 4.0, 1.6, 1.0, 1.4, 1.5, 1.8]
            hc = st.columns(ratios)
            for col, h in zip(
                hc,
                ['<span class="cbx" style="opacity:.6;"></span>', "Video", "Plattform", "Varighet", "Vurdering", "Status", ""],
            ):
                with col:
                    st.markdown(f'<div class="tbl-header">{h}</div>', unsafe_allow_html=True)

            for v in items:
                rc = st.columns(ratios)
                with rc[0]:
                    st.markdown('<div class="cell"><span class="cbx"></span></div>', unsafe_allow_html=True)
                with rc[1]:
                    st.markdown(_video_cell(v, with_channel=True), unsafe_allow_html=True)
                with rc[2]:
                    st.markdown(_plat_cell(v), unsafe_allow_html=True)
                with rc[3]:
                    st.markdown(_dur_cell(v), unsafe_allow_html=True)
                with rc[4]:
                    an = analyses.get(v.video_id)
                    st.markdown(
                        f'<div class="cell">{_rec_badge(an.get("recommendation", "")) if an else "—"}</div>',
                        unsafe_allow_html=True,
                    )
                with rc[5]:
                    st.markdown(_status_cell(v), unsafe_allow_html=True)
                with rc[6]:
                    if v.video_id in analyses:
                        a1, a2 = st.columns([3, 1])
                        with a1:
                            if st.button("Åpne analyse", key=f"lb_{v.video_id}", type="tertiary"):
                                st.session_state[f"d_{v.video_id}"] = True
                        with a2:
                            st.markdown(
                                '<div class="cell" style="justify-content:flex-end;"><span class="dots">···</span></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<div class="cell" style="justify-content:flex-end;"><span class="dots">···</span></div>',
                            unsafe_allow_html=True,
                        )

            # Footer + pagination
            fl, fr = st.columns([2, 1])
            with fl:
                st.markdown(
                    f'<div class="pageinfo">Viser {start+1}–{min(start+per_page, len(filtered))} av {len(filtered)} videoer</div>',
                    unsafe_allow_html=True,
                )
            with fr:
                if total_pages > 1:
                    pages = _page_window(page, total_pages)
                    cols = st.columns([0.8] + [1.0] * len(pages) + [0.8])
                    with cols[0]:
                        if st.button("‹", key="pg_p", type="secondary", disabled=page <= 1):
                            st.session_state.lib_page = page - 1
                            st.rerun()
                    for j, p in enumerate(pages):
                        with cols[1 + j]:
                            if st.button(
                                str(p),
                                key=f"pg_{p}",
                                type="primary" if p == page else "secondary",
                            ):
                                st.session_state.lib_page = p
                                st.rerun()
                    with cols[-1]:
                        if st.button("›", key="pg_n", type="secondary", disabled=page >= total_pages):
                            st.session_state.lib_page = page + 1
                            st.rerun()
    else:
        st.info("Ingen videoer matcher filteret.")

    # Detail view
    for v in items:
        if st.session_state.get(f"d_{v.video_id}"):
            _detail(v, analyses.get(v.video_id))
            if st.button("✕ Lukk analyse", key=f"cl_{v.video_id}"):
                st.session_state.pop(f"d_{v.video_id}", None)
                st.rerun()


# ─────────────────────────────────────────────────────────────────
# SYNTHESIS
# ─────────────────────────────────────────────────────────────────
def _temakart():
    """Topic map: recurring topics from video titles + techniques (local)."""
    from video_knowledge_extractor.cross_video import topic_map
    from video_knowledge_extractor.models import VideoAnalysis

    analyses = _load_analyses()
    if not analyses:
        st.info("Ingen analyser ennå — temakartet bygger på ferdig analyserte videoer.")
        return

    videos = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
    objs = [
        VideoAnalysis(
            video_id=vid,
            summary_no=a["summary_no"],
            techniques_tools=a["techniques_tools"],
            relevance_score=a["relevance_score"],
            practical_score=a["practical_score"],
            redundancy_score=a["redundancy_score"],
        )
        for vid, a in analyses.items()
    ]
    topics = topic_map(
        objs, {vid: (videos[vid].title if vid in videos else vid) for vid in analyses}
    )
    if not topics:
        return

    st.markdown(
        '<div class="secrow"><span class="sec-title">Temakart</span>'
        '<span class="count-pill">lokalt, uten LLM</span></div>',
        unsafe_allow_html=True,
    )
    for ti, (name, vids) in enumerate(topics):
        with st.container(border=True):
            st.markdown(
                f'<div class="panel-title">{name} '
                f'<span class="count-pill">{len(vids)} videoer</span></div>',
                unsafe_allow_html=True,
            )
            # Click a video to open its overview + analysis below.
            for i in range(0, len(vids), 4):
                cols = st.columns(4, gap="small")
                for c, r in zip(cols, vids[i : i + 4]):
                    with c:
                        v = videos.get(r.video_id)
                        if v:
                            st.markdown(
                                f'<div class="tmthumb" style="margin-bottom:4px;">'
                                f"{_thumb_html(v)}</div>",
                                unsafe_allow_html=True,
                            )
                        label = r.title if len(r.title) <= 30 else r.title[:27] + "…"
                        if st.button(
                            label,
                            key=f"tm{ti}_{r.video_id}",
                            type="tertiary",
                            use_container_width=True,
                            help=r.title,
                        ):
                            _insight_analysis(r.video_id)


def _insights():
    st.markdown("<h1>Innsikt</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>De beste videoene og kunnskapen som binder dem sammen.</p>", unsafe_allow_html=True)
    st.subheader("Høyest rangert")
    st.caption("Praktisk verdi og relevans, justert for overlapp med andre videoer.")
    _ranking(limit=5)
    with st.expander("Se hele rangeringen", key="full_ranking", on_change="rerun") as ranking:
        if ranking.open:
            _ranking()
    st.subheader("Samlet kunnskap")
    _synthesis()


def _synthesis():
    _temakart()

    if st.button("Generer / oppdater syntese", type="primary"):
        with st.spinner("Genererer syntese..."):
            try:
                run_synthesis(force=True)
            except Exception as exc:
                st.error(f"Feilet: {exc}")
        st.rerun()
    synth = load_synthesis(data_dir=Config.DATA_DIR)
    if not synth:
        st.info("Ingen syntese ennå. Klikk «Generer / oppdater syntese» for å lage en.")
        return

    st.markdown(
        f'<p style="color:#8b949e; font-size:12px;">Generert: {synth.get("created_at", "")}</p>',
        unsafe_allow_html=True,
    )
    sections = [
        ("Samlet tema", synth.get("theme_no", "—")),
        ("Felles tema", "\n".join(f"- {t}" for t in synth.get("shared_topics", []) or ["—"])),
        (
            "Motsetninger",
            "\n".join(
                f"- **{c.get('topic')}** ({', '.join(c.get('videos', []))}): {c.get('note')}"
                for c in synth.get("contradictions", [])
            )
            or "Ingen",
        ),
        (
            "Unik kunnskap",
            "\n".join(
                f"- **{u.get('video_id', '')}**: {u.get('knowledge', '')}" for u in synth.get("unique_knowledge", [])
            )
            or "—",
        ),
        ("Mangler (gaps)", "\n".join(f"- {g}" for g in synth.get("gaps", [])) or "—"),
        (
            "Anbefalt rekkefølge",
            "\n".join(
                f"- **{w.get('video_id', '')}**: {w.get('reason', '')}" for w in synth.get("watch_order", [])
            )
            or "—",
        ),
    ]
    with st.container(border=True):
        for title, content in sections:
            st.markdown(
                f"<h3 style='font-size:15px; color:#e6edf3; font-weight:700; margin:14px 0 6px 0;'>{title}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:14px; color:#c9d1d9; line-height:1.6;">{content}</div>',
                unsafe_allow_html=True,
            )
    if synth.get("recommendation"):
        st.markdown(
            "<h3 style='font-size:16px; color:#f0f6fc; font-weight:700; margin-top:16px;'>Anbefaling</h3>",
            unsafe_allow_html=True,
        )
        st.write(synth["recommendation"])


# ─────────────────────────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────────────────────────
@st.dialog("Analyse", width="large")
def _insight_analysis(video_id):
    st.caption("Lukk med × eller Esc for å gå tilbake til samme sted i Innsikt.")
    video = next((v for v in get_all_videos(data_dir=Config.DATA_DIR) if v.video_id == video_id), None)
    if video is None:
        st.info("Videoen finnes ikke lenger i biblioteket.")
        return
    _detail(video, _load_analyses().get(video_id))


def _ranking(limit=None):
    analyses = list(_load_analyses().values())
    if not analyses:
        st.info("Ingen analyser ennå.")
        return
    videos = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
    scored = [(a, (a["practical_score"] * a["relevance_score"]) / (a["redundancy_score"] + 1)) for a in analyses]
    scored.sort(key=lambda x: x[1], reverse=True)
    with st.container(border=True, key=f"insight_ranking_{limit}"):
        for rank, (a, score) in enumerate(scored[:limit], 1):
            if rank > 1:
                st.divider()
            v = videos.get(a["video_id"])
            title = v.title if v else a["video_id"]
            number, thumbnail, content = st.columns([0.4, 1, 8], vertical_alignment="center")
            with number:
                st.markdown(f"<span class='rank-number'>{rank}</span>", unsafe_allow_html=True)
            with thumbnail:
                if v:
                    st.markdown(_thumb_html(v), unsafe_allow_html=True)
            with content:
                if st.button(title, key=f"rank_{limit}_{a['video_id']}", type="tertiary", help="Åpne analyse"):
                    _insight_analysis(a["video_id"])
                st.caption(f"Verdi {score:.1f} · Praktisk {a['practical_score']} · Relevans {a['relevance_score']}")


# ─────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────
def _export():
    st.markdown("<h1>Eksport</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Eksporter data til JSON eller Markdown.</p>", unsafe_allow_html=True)
    videos = get_all_videos(data_dir=Config.DATA_DIR)
    analyses = list(_load_analyses().values())
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">JSON</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-desc">Komplett datasett med videoer og analyser.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Eksportér JSON", type="primary", use_container_width=True):
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
                p = Config.EXPORTS_DIR / "export.json"
                p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                st.success("Eksportert!")
                st.download_button(
                    "Last ned", p.read_text(encoding="utf-8"), "vke_export.json", use_container_width=True
                )
    with c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Markdown</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-desc">Obsidian-vennlige notater.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Eksportér Markdown", type="primary", use_container_width=True):
                md_files = export_markdown(data_dir=Config.DATA_DIR)
                st.success(f"{len(md_files)} filer eksportert")


# ─────────────────────────────────────────────────────────────────
# VIDEO DETAIL
# ─────────────────────────────────────────────────────────────────
def _ts_sec(ts):
    try:
        nums = [int(x) for x in (ts or "").split(":")]
    except Exception:
        return None
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def _detail(video, analysis):
    st.markdown(
        f'''<div style="margin-top:8px; padding:16px; background:#131a22; border:1px solid #26303b; border-radius:12px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px; flex-wrap:wrap;">
                {_plat_icon(video.platform)}
                <span style="font-weight:700; font-size:15px; color:#f0f6fc;">{video.title or video.video_id}</span>
            </div>
            <div style="color:#8b949e; font-size:12px;">
                {video.channel or "—"} · {video.duration_formatted} ·
                <a href="{video.url}" target="_blank" style="color:#388bfd;">Åpne i {video.platform}</a>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
    if not analysis:
        st.info("Ingen analyse tilgjengelig.")
        return
    st.markdown(
        "<h3 style='font-size:15px; color:#e6edf3; font-weight:700; margin-top:14px;'>Sammendrag</h3>",
        unsafe_allow_html=True,
    )
    st.write(analysis["summary_no"])
    st.markdown(
        "<h3 style='font-size:15px; color:#e6edf3; font-weight:700;'>Innsikter</h3>",
        unsafe_allow_html=True,
    )
    st.caption("✅ sitat verifisert · ⚠️ ikke funnet · • ingen sitat")
    for ins in analysis["insights"]:
        mark = "✅" if ins.get("verified") else ("⚠️" if ins.get("quote") else "•")
        ts = ins.get("timestamp") or ""
        tl = ""
        if ts and video.platform == "youtube":
            secs = _ts_sec(ts)
            if secs is not None:
                tl = f" [⏱ {ts}](https://www.youtube.com/watch?v={video.video_id}&t={secs}s)"
        elif ts:
            tl = f" ⏱ {ts}"
        st.write(f"- {mark} **{ins['insight']}**{tl}: {ins['explanation']}")
    st.markdown(
        "<h3 style='font-size:15px; color:#e6edf3; font-weight:700;'>Teknikker og verktøy</h3>",
        unsafe_allow_html=True,
    )
    st.write(", ".join(analysis["techniques_tools"]) or "Ingen")
    st.markdown(
        "<h3 style='font-size:15px; color:#e6edf3; font-weight:700;'>Handlingsbare steg</h3>",
        unsafe_allow_html=True,
    )
    for step in analysis["actionable_steps"]:
        st.write(f"- {step}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevans", analysis["relevance_score"])
    c2.metric("Dybde", analysis["depth_score"])
    c3.metric("Praktisk", analysis["practical_score"])
    c4.metric("Redundans", analysis["redundancy_score"])
    rec = analysis["recommendation"]
    rc = {"watch": "#3fb950", "skim": "#d29922", "skip": "#f85149"}.get(rec, "#8b949e")
    st.markdown(
        f'<p><strong>Anbefaling:</strong> <span style="color:{rc}; font-weight:700;">{rec.upper()}</span> — {analysis["recommendation_justification"]}</p>',
        unsafe_allow_html=True,
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
            st.caption("Ingen transkript.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    _sidebar()
    page = st.session_state.page
    if page == "oversikt":
        _overview()
    elif page == "bibliotek":
        _library()
    elif page == "ko":
        _queue()
    elif page == "innsikt":
        _insights()
    elif page == "eksport":
        _export()
    else:
        _overview()


main()
