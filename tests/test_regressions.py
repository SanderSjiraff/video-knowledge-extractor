from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youtube_knowledge_extractor import analysis, db, discovery, orchestrator
from youtube_knowledge_extractor.models import Transcript, TranscriptSegment, Video, VideoAnalysis, VideoStatus


def video(vid="dQw4w9WgXcQ", status=VideoStatus.DISCOVERED):
    return Video(vid, "Title", f"https://youtube.com/watch?v={vid}", 60, "Channel", 1, status)


def test_video_url_requires_youtube_host_and_full_id():
    assert discovery.extract_video_id("https://evil-youtube.com/watch?v=dQw4w9WgXcQ") is None
    assert discovery.extract_video_id("ftp://youtube.com/watch?v=dQw4w9WgXcQ") is None
    assert discovery.extract_video_id("https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert discovery.extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_ytdlp_discovery_rejects_non_youtube_sources_and_invalid_ids(monkeypatch):
    import subprocess

    with pytest.raises(ValueError, match="YouTube URL"):
        discovery.discover_from_ytdlp("https://example.com/playlist")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: type("Result", (), {
        "returncode": 0,
        "stdout": json.dumps({"entries": [{"id": "../bad"}, {"id": "dQw4w9WgXcQ"}]}),
        "stderr": "",
    })())
    assert [v.video_id for v in discovery.discover_from_ytdlp("https://youtube.com/playlist?list=abc")] == ["dQw4w9WgXcQ"]


def test_discover_file_limit_applies_before_metadata(monkeypatch, tmp_path):
    source = tmp_path / "urls.txt"
    source.write_text("https://youtu.be/dQw4w9WgXcQ\nhttps://youtu.be/9bZkp7q19f0\n")
    enriched = []
    monkeypatch.setattr(discovery, "enrich_video_metadata", lambda v: enriched.append(v.video_id) or v)
    monkeypatch.setattr(discovery, "save_video", lambda *a, **kw: None)
    result = discovery.discover_playlist(str(source), limit=1, data_dir=tmp_path / "db")
    assert [v.video_id for v in result] == ["dQw4w9WgXcQ"]
    assert enriched == ["dQw4w9WgXcQ"]


@pytest.mark.parametrize("limit", [0, -1])
def test_discovery_rejects_nonpositive_limit(limit):
    with pytest.raises(ValueError):
        discovery.discover_from_file(Path("unused"), limit=limit)


def test_save_video_rediscovery_preserves_processing_state_and_timestamps(tmp_path):
    db.init_db(tmp_path)
    original = video(status=VideoStatus.ANALYSIS_OK)
    db.save_video(original, tmp_path)
    db.save_transcript(Transcript(original.video_id, "en", False, [TranscriptSegment(0, 3, "cached")]), tmp_path)
    refreshed = video()
    db.save_video(refreshed, tmp_path)
    saved = db.get_video(original.video_id, tmp_path)
    assert saved.status is VideoStatus.ANALYSIS_OK
    assert saved.created_at == original.created_at
    assert db.get_transcript(original.video_id, tmp_path).full_text == "cached"


def test_success_status_clears_previous_error(tmp_path):
    db.init_db(tmp_path)
    db.save_video(video(), tmp_path)
    db.update_video_status("dQw4w9WgXcQ", VideoStatus.ANALYSIS_FAILED, "old failure", tmp_path)
    db.update_video_status("dQw4w9WgXcQ", VideoStatus.ANALYSIS_OK, data_dir=tmp_path)
    assert db.get_video("dQw4w9WgXcQ", tmp_path).error_message is None


def test_rediscovery_does_not_blank_enriched_metadata(tmp_path):
    db.init_db(tmp_path)
    db.save_video(video(), tmp_path)
    db.save_video(Video("dQw4w9WgXcQ", "", "https://youtube.com/watch?v=dQw4w9WgXcQ", 0, "", 2), tmp_path)
    saved = db.get_video("dQw4w9WgXcQ", tmp_path)
    assert (saved.title, saved.duration_seconds, saved.channel) == ("Title", 60, "Channel")


def test_rediscovery_recovers_inaccessible_video_without_resetting_completed_state(tmp_path, monkeypatch):
    db.init_db(tmp_path)
    inaccessible = video(status=VideoStatus.INACCESSIBLE)
    inaccessible.error_message = "transient metadata failure"
    db.save_video(inaccessible, tmp_path)
    db.save_video(video(status=VideoStatus.DISCOVERED), tmp_path)
    recovered = db.get_video(inaccessible.video_id, tmp_path)
    assert recovered.status is VideoStatus.DISCOVERED
    assert recovered.error_message is None

    ready = video("9bZkp7q19f0", status=VideoStatus.ANALYSIS_OK)
    db.save_video(ready, tmp_path)
    cached = Transcript(ready.video_id, "no", False, [TranscriptSegment(0, 1, "cached")])
    db.save_transcript(cached, tmp_path)
    db.save_video(video(ready.video_id, status=VideoStatus.DISCOVERED), tmp_path)
    assert db.get_video(ready.video_id, tmp_path).status is VideoStatus.ANALYSIS_OK
    assert db.get_transcript(ready.video_id, tmp_path).full_text == "cached"

    import youtube_knowledge_extractor.transcripts as transcripts
    monkeypatch.setattr(transcripts.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(transcripts.time, "sleep", lambda _: None)
    fetched = []
    monkeypatch.setattr(transcripts, "fetch_transcript", lambda v, force=False: fetched.append(v.video_id) or cached)
    result = transcripts.process_transcripts([recovered])
    assert result == {"success": 1, "failed": 0}
    assert fetched == [inaccessible.video_id]


def test_local_and_unknown_model_costs_are_handled_explicitly(monkeypatch):
    transcript = Transcript("x", "en", False, [TranscriptSegment(0, 1, "text")])
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", True)
    assert analysis.estimate_analysis_cost(transcript, model="qwen-local")["estimated_cost_usd"] == 0
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    with pytest.raises(ValueError, match="rate"):
        analysis.estimate_analysis_cost(transcript, model="unknown-remote")


def test_shared_prompt_preserves_timestamps_and_flags_truncation():
    transcript = Transcript("x", "en", False, [TranscriptSegment(20, 1, "A" * 15000)])
    prompt, truncated = analysis.build_analysis_prompt(transcript)
    assert truncated is True
    assert "00:20" in prompt
    assert "upålitelig" in prompt.lower()
    assert analysis.estimate_analysis_cost(transcript)["tokens_in"] == analysis.estimate_tokens(prompt)


def test_invalid_llm_shape_is_rejected():
    with pytest.raises(ValueError):
        analysis.validate_analysis_payload({"summary_no": "ok"})


def test_valid_llm_shape_is_accepted():
    analysis.validate_analysis_payload({
        "summary_no": "Oppsummering", "insights": [{"insight": "I", "explanation": "E", "timestamp": "00:01", "confidence": "high"}],
        "techniques_tools": [], "actionable_steps": [], "relevance_score": 3, "depth_score": 3,
        "practical_score": 4, "redundancy_score": 1, "recommendation": "skim",
        "recommendation_justification": "grunn", "key_sections": [{"start": "00:00", "end": "00:01", "description": "del"}],
        "estimated_key_watch_minutes": 1.5, "low_confidence": False, "confidence_reason": "",
    })


def test_call_llm_passes_reasoning_effort_only_when_configured(monkeypatch):
    sent = []

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "medium")
    monkeypatch.setattr(analysis.Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(analysis.Config, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: sent.append(kw["json"]) or Response())
    analysis.call_llm("prompt")
    assert sent[0]["reasoning_effort"] == "medium"
    assert sent[0]["max_tokens"] == analysis.Config.LLM_MAX_OUTPUT_TOKENS


def test_qwen_high_reasoning_is_rejected_before_request(monkeypatch):
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(analysis.Config, "LLM_MODEL", "Qwen3.8-27B-FP8")
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: pytest.fail("must reject before request"))
    with pytest.raises(ValueError, match="does not support"):
        analysis.call_llm("prompt")


def test_run_analysis_skips_saved_analysis_even_if_video_status_was_reset(monkeypatch, tmp_path):
    db.init_db(tmp_path)
    v = video(status=VideoStatus.TRANSCRIPT_OK)
    db.save_video(v, tmp_path)
    db.save_transcript(Transcript(v.video_id, "no", False, [TranscriptSegment(0, 1, "ready")]), tmp_path)
    with db.get_connection(tmp_path) as conn:
        conn.execute("INSERT INTO analyses(video_id, summary_no, analyzed_at) VALUES (?, ?, ?)", (v.video_id, "saved", datetime.utcnow().isoformat()))
        conn.commit()
    monkeypatch.setattr(orchestrator.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "run_cost_estimate", lambda videos: {"estimated_cost_usd": 0})
    monkeypatch.setattr(orchestrator, "analyze_video", lambda *a, **kw: pytest.fail("should skip saved analysis"))
    assert orchestrator.run_analysis([v]) == []


def test_run_analysis_force_ignores_zero_and_negative_limits(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        orchestrator.run_analysis([], dry_run_limit=0)


def test_fallback_uses_preferred_manual_caption_and_provenance(monkeypatch, tmp_path):
    import youtube_knowledge_extractor.transcripts as transcripts

    monkeypatch.setattr(transcripts.tempfile, "tempdir", str(tmp_path))
    seen = []
    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        if "--write-subs" in cmd and cmd[cmd.index("--sub-langs") + 1] == "no":
            output = Path(cmd[cmd.index("-o") + 1])
            output = Path(str(output).replace("%(id)s", "dQw4w9WgXcQ").replace("%(ext)s", "no.json3"))
            output.write_text(json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Hei"}]}]}))
            return type("Result", (), {"returncode": 0})()
        return type("Result", (), {"returncode": 1})()
    monkeypatch.setattr(transcripts.subprocess, "run", fake_run)
    result = transcripts._fetch_with_ytdlp("dQw4w9WgXcQ")
    assert result and result.language == "no" and result.is_generated is False
    assert "--write-subs" in seen[0]


def test_cached_transcript_repairs_status_without_downgrading_analysis(tmp_path, monkeypatch):
    import youtube_knowledge_extractor.transcripts as transcripts
    db.init_db(tmp_path)
    v = video(status=VideoStatus.DISCOVERED)
    db.save_video(v, tmp_path)
    transcript = Transcript(v.video_id, "no", False, [TranscriptSegment(0, 1, "Hei")])
    db.save_transcript(transcript, tmp_path)
    monkeypatch.setattr(transcripts.Config, "DATA_DIR", tmp_path)
    transcripts.fetch_transcript(v)
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.TRANSCRIPT_OK
    db.update_video_status(v.video_id, VideoStatus.ANALYSIS_OK, data_dir=tmp_path)
    v.status = VideoStatus.ANALYSIS_OK
    transcripts.fetch_transcript(v)
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_OK


def valid_analysis_payload():
    return {
        "summary_no": "Norsk oppsummering", "insights": [{"insight": "innsikt", "explanation": "forklaring", "timestamp": "00:01", "confidence": "high"}],
        "techniques_tools": ["verktøy"], "actionable_steps": ["gjør dette"], "relevance_score": 3,
        "depth_score": 3, "practical_score": 4, "redundancy_score": 1, "recommendation": "skim",
        "recommendation_justification": "relevant", "key_sections": [{"start": "00:00", "end": "00:30", "description": "del"}],
        "estimated_key_watch_minutes": 1.5, "low_confidence": False, "confidence_reason": "",
    }


def prepare_analysis_db(tmp_path, monkeypatch, vid="dQw4w9WgXcQ", transcript_text="nyttig transkript"):
    db.init_db(tmp_path)
    v = video(vid, VideoStatus.TRANSCRIPT_OK)
    db.save_video(v, tmp_path)
    db.save_transcript(Transcript(vid, "no", False, [TranscriptSegment(0, 10, transcript_text)]), tmp_path)
    monkeypatch.setattr(orchestrator.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analysis.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(analysis.Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(analysis.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "")
    monkeypatch.setattr(analysis.Config, "LLM_INPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "LLM_OUTPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(analysis.Config, "SPENDING_LIMIT_USD", 1.0)
    return v


def test_force_analysis_calls_mock_api_and_persists_success_atomically(tmp_path, monkeypatch):
    v = prepare_analysis_db(tmp_path, monkeypatch)
    with db.get_connection(tmp_path) as conn:
        conn.execute("INSERT INTO analyses(video_id, summary_no, analyzed_at) VALUES (?, ?, ?)", (v.video_id, "old", datetime.utcnow().isoformat()))
        conn.commit()

    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": json.dumps(valid_analysis_payload())}}]}
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: calls.append(kw) or Response())
    results = orchestrator.run_analysis([v], force=True)
    with db.get_connection(tmp_path) as conn:
        saved = conn.execute("SELECT summary_no FROM analyses WHERE video_id = ?", (v.video_id,)).fetchone()
    assert len(results) == 1 and calls and saved["summary_no"] == "Norsk oppsummering"
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_OK


def test_invalid_numeric_output_marks_failed_without_saving(tmp_path, monkeypatch):
    v = prepare_analysis_db(tmp_path, monkeypatch)
    payload = valid_analysis_payload()
    payload["relevance_score"] = True
    monkeypatch.setattr(analysis, "call_llm", lambda prompt: payload)
    assert analysis.analyze_video(v) is None
    assert db.get_video(v.video_id, tmp_path).status is VideoStatus.ANALYSIS_FAILED
    with db.get_connection(tmp_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0


def test_truncated_transcript_forces_low_confidence(tmp_path, monkeypatch):
    v = prepare_analysis_db(tmp_path, monkeypatch, transcript_text="A" * 16000)
    monkeypatch.setattr(analysis, "call_llm", lambda prompt: valid_analysis_payload())
    result = analysis.analyze_video(v)
    assert result and result.low_confidence is True
    assert "truncated" in result.confidence_reason.lower()


def test_positive_unrounded_cost_over_zero_budget_blocks_requests(tmp_path, monkeypatch):
    v = prepare_analysis_db(tmp_path, monkeypatch)
    monkeypatch.setattr(analysis.Config, "LLM_INPUT_COST_PER_MILLION", 0.001)
    monkeypatch.setattr(analysis.Config, "LLM_OUTPUT_COST_PER_MILLION", 0.001)
    monkeypatch.setattr(analysis.Config, "SPENDING_LIMIT_USD", 0.0)
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: pytest.fail("budget must block before API"))
    assert orchestrator.run_analysis([v], force=True) == []


def test_failed_request_attempts_still_observe_rate_limit(monkeypatch):
    ticks = iter([10.0, 10.0, 10.2, 11.0])
    sleeps = []
    monkeypatch.setattr(analysis.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(analysis.Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(analysis.Config, "LLM_REASONING_EFFORT", "")
    monkeypatch.setattr(analysis.Config, "RATE_LIMIT_SECONDS", 1.0)
    monkeypatch.setattr(analysis, "_last_llm_call", 0.0)
    monkeypatch.setattr(analysis.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(analysis.time, "sleep", sleeps.append)
    monkeypatch.setattr(analysis.requests, "post", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="offline"):
            analysis.call_llm("prompt")
    assert sleeps == pytest.approx([0.8])


def test_dashboard_starts_with_fresh_temp_database(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from youtube_knowledge_extractor.config import Config
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(Config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(Config, "ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(Config, "EXPORTS_DIR", tmp_path / "exports")
    app = AppTest.from_file(str(Path(__file__).parent.parent / "src" / "youtube_knowledge_extractor" / "dashboard.py")).run(timeout=10)
    assert not app.exception
