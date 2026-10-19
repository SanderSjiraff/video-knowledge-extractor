"""Quality tests for plan items #5 (Markdown export) and #6 (quality assurance).

Covers: Markdown export with source deep-links, timestamp edge cases,
long-transcript chunking edge cases, quote grounding / source linking, and
model error handling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youtube_knowledge_extractor import analysis, db, notes
from youtube_knowledge_extractor.models import (
    Insight,
    Transcript,
    TranscriptSegment,
    Video,
    VideoAnalysis,
    VideoStatus,
)


def video(vid="abc123"):
    return Video(vid, "Bygge en AI-agent", f"https://www.youtube.com/watch?v={vid}", 1290, "Kanal", 1, VideoStatus.ANALYSIS_OK)


def _mk_analysis(vid="abc123"):
    return VideoAnalysis(
        video_id=vid,
        summary_no="Oppsummert på norsk.",
        insights=[
            Insight(
                insight="Modellen leveres mandag",
                explanation="Fordi den er klar.",
                timestamp="01:30",
                confidence="high",
                quote="the model ships monday",
                verified=True,
            ),
            Insight(
                insight="Usikkert påstand",
                explanation="Mangler belegg.",
                timestamp="05:00",
                confidence="low",
                quote="ikkje funne i transkriptet",
                verified=False,
                verification_reason="Sitatet ble ikke funnet i transkriptet",
            ),
        ],
        techniques_tools=["transformer", "RAG"],
        actionable_steps=["Lag et poeng"],
        relevance_score=4,
        depth_score=3,
        practical_score=4,
        redundancy_score=1,
        recommendation="watch",
        recommendation_justification="Grunn.",
        key_sections=[{"start": "01:30", "end": "02:00", "description": "lansering"}],
    )


# ---------------------------------------------------------------------------
# #5: Markdown export with source deep-links
# ---------------------------------------------------------------------------

def test_deep_link_with_and_without_timestamp():
    assert notes.deep_link("abc", None) == "https://www.youtube.com/watch?v=abc"
    assert notes.deep_link("abc", "") == "https://www.youtube.com/watch?v=abc"
    assert notes.deep_link("abc", "01:30") == "https://www.youtube.com/watch?v=abc&t=90s"
    assert notes.deep_link("abc", "01:00:00") == "https://www.youtube.com/watch?v=abc&t=3600s"
    # Invalid timestamps must not produce a broken t= parameter.
    assert notes.deep_link("abc", "bad") == "https://www.youtube.com/watch?v=abc"
    assert notes.deep_link("abc", "1:2:3:4") == "https://www.youtube.com/watch?v=abc"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("00:00", 0),
        ("02:00", 120),
        ("01:00:00", 3600),
        (None, None),
        ("", None),
        ("abc", None),
        ("1:2:3:4", None),
    ],
)
def test_ts_to_seconds_edge_cases(raw, expected):
    assert notes._ts_to_seconds(raw) == expected


def test_safe_filename_handles_norwegian_and_special_chars():
    assert notes._safe_filename("Bygge en AI-agent") == "Bygge-en-AI-agent"
    assert notes._safe_filename("Å/øå: er tillatt?") == "Å-øå-er-tillatt"
    assert notes._safe_filename("a" * 200) == "a" * 80
    assert notes._safe_filename("///") == "notat"
    assert notes._safe_filename("") == "notat"


def test_note_filename_is_stable_and_unique_per_video():
    a, b = video("abc123"), video("def456")
    assert notes.note_filename(a) == "Bygge-en-AI-agent.abc123.md"
    # Same title, different id -> different file.
    assert notes.note_filename(a) != notes.note_filename(b)


def test_video_to_markdown_includes_frontmatter():
    v, a = video("abc123"), _mk_analysis("abc123")
    md = notes.video_to_markdown(v, a)
    assert md.startswith("---\n")
    # Frontmatter block must close before the title heading.
    fm = md.split("---", 2)[1]
    assert "video_id: abc123" in fm
    assert "platform: youtube" in fm
    assert "url: \"https://www.youtube.com/watch?v=abc123\"" in fm
    assert "recommendation: watch" in fm
    assert "relevance: 4" in fm
    assert "tags:" in fm and "- \"transformer\"" in fm
    assert "# Bygge en AI-agent" in md


def test_video_to_markdown_includes_source_links():
    v, a = video("abc123"), _mk_analysis("abc123")
    md = notes.video_to_markdown(v, a)
    assert "# Bygge en AI-agent" in md
    assert "[Se på YouTube](https://www.youtube.com/watch?v=abc123)" in md
    assert "relevans 4/5" in md
    assert "**Anbefaling:** watch" in md
    # Insight with 01:30 timestamp becomes a deep link to 90s.
    assert "t=90s" in md
    assert "Modellen leveres mandag" in md
    # Verified insight gets a check mark; unverified one a warning.
    assert "✅" in md
    assert "⚠️" in md
    # Quote and verification note are preserved as source evidence.
    assert "> the model ships monday" in md
    assert "_Sitatet ble ikke funnet i transkriptet_" in md
    assert "- transformer" in md
    assert "1. Lag et poeng" in md
    # Key section links jump to its start timestamp.
    assert "[hoppe til](https://www.youtube.com/watch?v=abc123&t=90s)" in md
    assert "lansering" in md


def test_video_to_markdown_flags_low_confidence_analysis():
    v, a = video("abc123"), _mk_analysis("abc123")
    a.low_confidence = True
    a.confidence_reason = "Ufullstendig transkript"
    md = notes.video_to_markdown(v, a)
    assert "- **Lav konfidens:** Ufullstendig transkript" in md


def test_playlist_overview_contains_synthesis_and_ranked_table():
    v1, v2 = video("abc123"), video("def456")
    a1 = _mk_analysis("abc123")  # score = 4*4/(1+1) = 8.0
    a2 = VideoAnalysis(
        video_id="def456", summary_no="s", relevance_score=1,
        practical_score=1, redundancy_score=1, recommendation="skip",
    )
    synthesis = {
        "theme_no": "Samlingen handler om AI-agenter.",
        "shared_topics": ["transformere"],
        "contradictions": [
            {"topic": "Kostnad", "videos": ["abc123", "def456"], "note": "uoverenstemmende tall"}
        ],
        "unique_knowledge": [{"video_id": "abc123", "knowledge": "eval-routine"}],
    }
    md = notes.playlist_overview_markdown([v1, v2], [a1, a2], synthesis)
    assert "Samlingen handler om AI-agenter." in md
    assert "transformere" in md
    assert "Kostnad (abc123, def456): uoverenstemmende tall" in md
    assert "- [Bygge en AI-agent](https://www.youtube.com/watch?v=abc123): eval-routine" in md
    # Highest-value video is ranked first.
    assert "| 1 | [Bygge en AI-agent](https://www.youtube.com/watch?v=abc123)" in md
    # Index links to the per-video note files.
    assert "Bygge-en-AI-agent.abc123.md" in md
    assert "Bygge-en-AI-agent.def456.md" in md


def _insert_analysis(data_dir: Path, vid: str, summary: str = "s") -> None:
    with db.get_connection(data_dir) as conn:
        conn.execute(
            "INSERT INTO analyses (video_id, summary_no, insights_json, techniques_tools_json, "
            "actionable_steps_json, relevance_score, depth_score, practical_score, redundancy_score, "
            "recommendation, recommendation_justification, key_sections_json, "
            "estimated_key_watch_minutes, low_confidence, confidence_reason, analyzed_at, "
            "llm_model, estimated_tokens_in, estimated_tokens_out) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (vid, summary, "[]", "[]", "[]", 3, 3, 4, 1, "watch", "g", "[]", 1.0, 0, "",
             "2024-01-01T00:00:00", "m", 0, 0),
        )
        conn.commit()


def test_export_markdown_writes_note_per_video_and_index(tmp_path):
    db.init_db(tmp_path)
    db.save_video(video("abc123"), tmp_path)
    db.save_video(video("def456"), tmp_path)
    _insert_analysis(tmp_path, "abc123", "Opp A")
    _insert_analysis(tmp_path, "def456", "Opp B")
    db.save_synthesis({"theme_no": "Tema", "video_ids": ["abc123", "def456"], "created_at": "now"}, tmp_path)

    files = notes.export_markdown(data_dir=tmp_path)
    names = sorted(f.name for f in files)
    assert len(names) == 4  # 2 notes + index + concentrate
    assert "INDEKS.md" in names and "KUNNSKAP.md" in names
    out_dir = tmp_path / "exports" / "markdown"
    note = (out_dir / "Bygge-en-AI-agent.abc123.md").read_text(encoding="utf-8")
    assert "Opp A" in note and note.startswith("---\n")
    index = (out_dir / "INDEKS.md").read_text(encoding="utf-8")
    assert "Tema" in index
    assert "abc123" in index and "def456" in index
    concentrate = (out_dir / "KUNNSKAP.md").read_text(encoding="utf-8")
    assert "# Kunnskapskonsentrat" in concentrate
    assert "Tema" in concentrate


def test_export_markdown_falls_back_when_video_metadata_missing(tmp_path):
    db.init_db(tmp_path)
    _insert_analysis(tmp_path, "ghost999", "Ukjent")
    files = notes.export_markdown(data_dir=tmp_path)
    ghost = next(f for f in files if "ghost999" in f.name)
    assert "Ukjent" in ghost.read_text(encoding="utf-8")


def test_export_markdown_empty_library_writes_overviews_only(tmp_path):
    db.init_db(tmp_path)
    files = notes.export_markdown(data_dir=tmp_path)
    assert sorted(f.name for f in files) == ["INDEKS.md", "KUNNSKAP.md"]
    concentrate = (tmp_path / "exports" / "markdown" / "KUNNSKAP.md").read_text(encoding="utf-8")
    assert "fra 0 analyserte videoer" in concentrate


def test_concentrate_deduplicates_and_orders_verified_first():
    v1, v2 = video("abc123"), video("def456")
    a1 = VideoAnalysis(
        video_id="abc123", summary_no="s", relevance_score=5,
        insights=[
            Insight(insight="Delt innsikt", explanation="e", timestamp="01:00",
                    confidence="high", quote="q", verified=True),
            Insight(insight="Unik i video 1", explanation="e", confidence="low"),
        ],
        techniques_tools=["transformer", "RAG"],
        actionable_steps=["Steg A"],
    )
    a2 = VideoAnalysis(
        video_id="def456", summary_no="s", relevance_score=2,
        insights=[
            # Duplicate of a1's first insight (same text) but unverified.
            Insight(insight="Delt innsikt", explanation="e", confidence="medium"),
            Insight(insight="Unik i video 2", explanation="e", confidence="high",
                    quote="q2", verified=True),
        ],
        techniques_tools=["RAG", "pytorch"],
        actionable_steps=["Steg B"],
    )
    md = notes.knowledge_concentrate_markdown([v1, v2], [a1, a2])
    # The shared insight appears exactly once (deduplicated).
    assert md.count("**Bygge en AI-agent**") >= 1
    shared_lines = [l for l in md.splitlines() if "Delt innsikt" in l]
    assert len(shared_lines) == 1
    # Verified insight (✅) is listed before the low/unverified-only ones.
    verified_pos = md.find("Unik i video 2")
    unverified_pos = md.find("Unik i video 1")
    assert verified_pos != -1 and unverified_pos != -1
    assert verified_pos < unverified_pos
    # Techniques deduplicated across videos.
    techniques_section = md.split("## Teknikker og verktøy (samlet)")[1].split("##")[0]
    assert techniques_section.count("RAG") == 1
    assert "pytorch" in techniques_section and "transformer" in techniques_section


# ---------------------------------------------------------------------------
# #6: quality tests — timestamps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("90", 90.0),
        (" 02:00 ", 120.0),
        ("00:05:00:00", None),
        ("00:", None),
        (":30", None),
        ("-1:00", None),
        ("", None),
    ],
)
def test_parse_timestamp_edge_cases(raw, expected):
    assert analysis._parse_timestamp(raw) == expected


@pytest.mark.parametrize(
    "secs,expected",
    [
        (0, "00:00"),
        (-5, "00:00"),
        (59.4, "00:59"),
        (3599, "59:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
    ],
)
def test_format_timestamp_edge_cases(secs, expected):
    assert analysis._format_timestamp(secs) == expected


# ---------------------------------------------------------------------------
# #6: quality tests — long transcripts / chunking
# ---------------------------------------------------------------------------

def test_chunk_transcript_splits_giant_single_segment():
    t = Transcript("x", "en", False, [TranscriptSegment(0, 60, "A" * 60000)])
    chunks = analysis.chunk_transcript(t)
    assert len(chunks) > 3
    for chunk in chunks:
        assert len(chunk) <= analysis.MAX_TRANSCRIPT_CHARS
    # A long unbroken run survives, so no content is silently dropped.
    assert "A" * 1000 in "\n".join(chunks)


def test_chunk_transcript_empty_and_whitespace_only_return_no_chunks():
    assert analysis.chunk_transcript(Transcript("x", "en", False, [])) == []
    t = Transcript("x", "en", False, [TranscriptSegment(0, 1, "   "), TranscriptSegment(2, 1, "\t")])
    assert analysis.chunk_transcript(t) == []


def test_short_transcript_uses_single_chunk_path():
    t = Transcript("x", "en", False, [TranscriptSegment(0, 5, "kort transkript")])
    chunks = analysis.chunk_transcript(t)
    assert len(chunks) == 1
    assert "kort transkript" in chunks[0]


def test_estimate_cost_empty_transcript_is_zero(monkeypatch):
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", True)
    est = analysis.estimate_analysis_cost(Transcript("x", "en", False, []))
    assert est == {"tokens_in": 0, "tokens_out": 0, "estimated_cost_usd": 0.0}


# ---------------------------------------------------------------------------
# #6: quality tests — source linking (quote grounding)
# ---------------------------------------------------------------------------

def test_grounding_normalizes_punctuation_case_and_emoji():
    t = Transcript("x", "en", False, [TranscriptSegment(0, 5, "Big news! The model 🚀 ships on Monday.")])
    ins = Insight(insight="i", explanation="e", timestamp="00:00", confidence="high",
                  quote="the model ships on monday")
    [v] = analysis.verify_insights(t, [ins])
    assert v.verified is True
    assert v.verification_reason == ""


def test_grounding_tolerates_missing_words_between_quote_tokens():
    t = Transcript("x", "en", False, [TranscriptSegment(0, 5, "we use a big fancy transformer today")])
    ins = Insight(insight="i", explanation="e", timestamp="00:00", confidence="high",
                  quote="we use transformer")
    [v] = analysis.verify_insights(t, [ins])
    assert v.verified is True


def test_grounding_quote_spanning_segments():
    t = Transcript("x", "en", False, [
        TranscriptSegment(0, 2, "the model"),
        TranscriptSegment(100, 2, "uses attention"),
    ])
    ins = Insight(insight="i", explanation="e", timestamp="00:00", confidence="high",
                  quote="the model uses attention")
    [v] = analysis.verify_insights(t, [ins])
    assert v.verified is True


def test_timestamp_snap_boundary_is_exactly_window_seconds():
    t = Transcript("x", "en", False, [TranscriptSegment(0, 5, "the model works")])
    within = Insight(insight="i", explanation="e", timestamp="01:30", confidence="high",
                     quote="the model works")  # 90s off -> within window
    beyond = Insight(insight="i", explanation="e", timestamp="01:31", confidence="high",
                     quote="the model works")  # 91s off -> snapped
    [w, b] = analysis.verify_insights(t, [within, beyond])
    assert w.timestamp == "01:30" and w.verification_reason == ""
    assert b.timestamp == "00:00" and "justert" in b.verification_reason


def test_quote_without_claimed_timestamp_verifies_and_keeps_timestamp():
    t = Transcript("x", "en", False, [TranscriptSegment(120, 5, "the model works here")])
    ins = Insight(insight="i", explanation="e", timestamp=None, confidence="high",
                  quote="the model works")
    [v] = analysis.verify_insights(t, [ins])
    assert v.verified is True
    assert v.timestamp is None


# ---------------------------------------------------------------------------
# #6: quality tests — model error handling
# ---------------------------------------------------------------------------

def _valid_payload():
    return {
        "summary_no": "Opp", "insights": [], "techniques_tools": [], "actionable_steps": [],
        "relevance_score": 3, "depth_score": 3, "practical_score": 4, "redundancy_score": 1,
        "recommendation": "watch", "recommendation_justification": "g",
        "key_sections": [], "estimated_key_watch_minutes": 1.0,
        "low_confidence": False, "confidence_reason": "",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("summary_no", "   "),
        lambda p: p.__setitem__("recommendation", "maybe"),
        lambda p: p.__setitem__("relevance_score", True),
        lambda p: p.__setitem__("relevance_score", 6),
        lambda p: p.__setitem__("depth_score", 3.0),
        lambda p: p.__setitem__("techniques_tools", ["ok", 5]),
        lambda p: p.__setitem__("insights", "ikke en liste"),
        lambda p: p.__setitem__("insights", [{"insight": "i", "explanation": "", "timestamp": "", "confidence": "high"}]),
        lambda p: p.__setitem__("insights", [{"insight": "i", "explanation": "e", "timestamp": "", "confidence": "medium-high"}]),
        lambda p: p.__setitem__("key_sections", [{"start": "00:00", "end": 5, "description": "d"}]),
        lambda p: p.__setitem__("estimated_key_watch_minutes", -1),
        lambda p: p.__setitem__("estimated_key_watch_minutes", float("nan")),
        lambda p: p.__setitem__("low_confidence", "ja"),
    ],
)
def test_validate_analysis_payload_rejects_invalid_fields(mutate):
    p = _valid_payload()
    mutate(p)
    with pytest.raises(ValueError):
        analysis.validate_analysis_payload(p)


def _setup_analysis_env(tmp_path, monkeypatch, vid="abc123", text="nyttig transkript"):
    db.init_db(tmp_path)
    v = video(vid)
    db.save_video(v, tmp_path)
    t = Transcript(vid, "no", False, [TranscriptSegment(0, 10, text)])
    db.save_transcript(t, tmp_path)
    monkeypatch.setattr(analysis.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(analysis.Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(analysis.Config, "LLM_API_KEY", "k")
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "")
    monkeypatch.setattr(analysis.Config, "LLM_INPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "LLM_OUTPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(analysis.Config, "SPENDING_LIMIT_USD", 1.0)
    return v, t


def test_analyze_video_llm_exception_marks_failed_without_saving(tmp_path, monkeypatch):
    v, t = _setup_analysis_env(tmp_path, monkeypatch)

    def boom(prompt, model=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(analysis, "call_llm", boom)
    assert analysis.analyze_video(v, t) is None
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_FAILED
    with db.get_connection(tmp_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0


def test_analyze_video_non_dict_payload_marks_failed(tmp_path, monkeypatch):
    v, t = _setup_analysis_env(tmp_path, monkeypatch)
    monkeypatch.setattr(analysis, "call_llm", lambda *a, **kw: ["ikke", "et", "objekt"])
    assert analysis.analyze_video(v, t) is None
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_FAILED


def test_analyze_video_empty_transcript_marks_failed(tmp_path, monkeypatch):
    v, _ = _setup_analysis_env(tmp_path, monkeypatch)
    empty = Transcript(v.video_id, "no", False, [TranscriptSegment(0, 1, "   ")])
    assert analysis.analyze_video(v, empty) is None
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_FAILED


def test_analyze_video_keeps_verified_quote_and_snapped_timestamp(tmp_path, monkeypatch):
    v, t = _setup_analysis_env(tmp_path, monkeypatch, text="the model ships on monday")
    payload = _valid_payload()
    payload["insights"] = [
        {"insight": "Leveres mandag", "explanation": "e", "timestamp": "99:99",
         "confidence": "high", "quote": "the model ships on monday"}
    ]
    monkeypatch.setattr(analysis, "call_llm", lambda *a, **kw: payload)
    result = analysis.analyze_video(v, t)
    assert result is not None
    ins = result.insights[0]
    assert ins.verified is True
    assert ins.timestamp == "00:00"  # snapped to the quote's real location
    assert ins.quote == "the model ships on monday"


def test_ollama_response_strips_markdown_fences(monkeypatch):
    class Response:
        def raise_for_status(self): pass

        def json(self):
            return {"response": "```json\n" + json.dumps({"summary_no": "ok"}) + "\n```"}

    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", True)
    monkeypatch.setattr(analysis.Config, "OLLAMA_MODEL", "llama3.2")
    monkeypatch.setattr(analysis.Config, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(analysis.Config, "LLM_INPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "LLM_OUTPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "")
    monkeypatch.setattr(analysis, "_last_llm_call", 0.0)
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: Response())
    assert analysis.call_llm("prompt") == {"summary_no": "ok"}
