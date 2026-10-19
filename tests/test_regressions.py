from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_knowledge_extractor import analysis, cross_video, db, discovery, orchestrator
from video_knowledge_extractor.models import (
    Insight,
    Transcript,
    TranscriptSegment,
    Video,
    VideoAnalysis,
    VideoStatus,
)


def video(vid="dQw4w9WgXcQ", status=VideoStatus.DISCOVERED):
    return Video(vid, "Title", f"https://youtube.com/watch?v={vid}", 60, "Channel", 1, status)


def test_video_url_requires_youtube_host_and_full_id():
    assert discovery.extract_video_id("https://evil-youtube.com/watch?v=dQw4w9WgXcQ") is None
    assert discovery.extract_video_id("ftp://youtube.com/watch?v=dQw4w9WgXcQ") is None
    assert discovery.extract_video_id("https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert discovery.extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://www.tiktok.com/@user/video/1234567890", "tiktok"),
        ("https://vm.tiktok.com/ABC123/", "tiktok"),
        ("https://www.instagram.com/reel/AbCdEf/", "instagram"),
        ("https://instagr.am/p/xyz/", "instagram"),
        ("https://example.com/watch?v=abc", None),
        ("not a url", None),
    ],
)
def test_detect_platform(url, expected):
    assert discovery.detect_platform(url) == expected


def test_discover_from_file_resolves_tiktok_via_ytdlp(tmp_path, monkeypatch):
    source = tmp_path / "urls.txt"
    source.write_text(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
        "https://www.tiktok.com/@user/video/1234567890\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        discovery,
        "_yt_dlp_single_video",
        lambda url: calls.append(url) or {"id": "777000111222", "title": "TikTok clip",
                                            "channel": "someuser", "duration": 30},
    )
    videos = discovery.discover_from_file(source)
    assert [(v.video_id, v.platform) for v in videos] == [
        ("dQw4w9WgXcQ", "youtube"),
        ("777000111222", "tiktok"),
    ]
    # Only the non-YouTube URL needed a yt-dlp resolution.
    assert calls == ["https://www.tiktok.com/@user/video/1234567890"]
    tiktok = videos[1]
    assert tiktok.title == "TikTok clip" and tiktok.duration_seconds == 30


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


def test_video_platform_roundtrips_through_db(tmp_path):
    db.init_db(tmp_path)
    db.save_video(Video("777000111222", "TikTok", "https://www.tiktok.com/@u/video/777000111222", 30, "u", 1, VideoStatus.DISCOVERED, platform="tiktok"), tmp_path)
    assert db.get_video("777000111222", tmp_path).platform == "tiktok"
    # A video saved without an explicit platform defaults to youtube.
    db.save_video(video(), tmp_path)
    assert db.get_video("dQw4w9WgXcQ", tmp_path).platform == "youtube"


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

    import video_knowledge_extractor.transcripts as transcripts
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
    # Hermetic: pretend no explicit rates are configured (a real .env may set them).
    monkeypatch.setattr(analysis.Config, "LLM_INPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(analysis.Config, "LLM_OUTPUT_COST_PER_MILLION", None)
    with pytest.raises(ValueError, match="rate"):
        analysis.estimate_analysis_cost(transcript, model="unknown-remote")


def test_shared_prompt_preserves_timestamps_and_flags_truncation():
    transcript = Transcript("x", "en", False, [TranscriptSegment(20, 1, "A" * 15000)])
    prompt, truncated = analysis.build_analysis_prompt(transcript)
    assert truncated is True
    assert "00:20" in prompt
    assert "upålitelig" in prompt.lower()
    # A short, single-chunk transcript's cost estimate matches its prompt tokens.
    short = Transcript("x", "en", False, [TranscriptSegment(20, 1, "kort tekst")])
    short_prompt, short_truncated = analysis.build_analysis_prompt(short)
    assert short_truncated is False
    assert analysis.estimate_analysis_cost(short)["tokens_in"] == analysis.estimate_tokens(short_prompt)


def test_chunk_transcript_never_drops_content_and_stays_within_budget():
    segments = [TranscriptSegment(i, 1, f"ord{i} " * 5) for i in range(400)]
    transcript = Transcript("x", "en", False, segments)
    chunks = analysis.chunk_transcript(transcript)
    assert len(chunks) > 1
    joined = "\n".join(chunks)
    for i in range(400):
        assert f"ord{i}" in joined
    for chunk in chunks:
        assert len(chunk) <= analysis.MAX_TRANSCRIPT_CHARS


def test_chunk_transcript_overlaps_consecutive_chunks():
    segments = [TranscriptSegment(i, 1, f"ord{i} " * 5) for i in range(400)]
    transcript = Transcript("x", "en", False, segments)
    chunks = analysis.chunk_transcript(transcript)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        last_word = chunks[i].split()[-1]
        assert last_word in chunks[i + 1].split()


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
    import video_knowledge_extractor.transcripts as transcripts

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
    import video_knowledge_extractor.transcripts as transcripts
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


def test_long_transcript_is_chunked_and_fully_analyzed(tmp_path, monkeypatch):
    v = prepare_analysis_db(tmp_path, monkeypatch, transcript_text="A" * 16000)
    prompts: list[str] = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return valid_analysis_payload()

    monkeypatch.setattr(analysis, "call_llm", fake_llm)
    result = analysis.analyze_video(v)
    # 16000 chars > 1 chunk -> 2 map calls + 1 merge call, with no truncation.
    assert result is not None
    assert len(prompts) == 3
    # No content lost: the two map prompts together cover the whole transcript.
    assert sum(p.count("A") for p in prompts[:2]) >= 16000
    # Chunking no longer forces a low-confidence / truncation flag.
    assert result.low_confidence is False
    assert "truncated" not in result.confidence_reason.lower()


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
    from video_knowledge_extractor.config import Config
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(Config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(Config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(Config, "ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(Config, "EXPORTS_DIR", tmp_path / "exports")
    app = AppTest.from_file(str(Path(__file__).parent.parent / "src" / "video_knowledge_extractor" / "dashboard.py")).run(timeout=10)
    assert not app.exception


def test_fts_search_finds_content_and_snippet(tmp_path):
    db.init_db(tmp_path)
    db.save_transcript(Transcript("vid1", "en", False, [TranscriptSegment(0, 5, "quantum entanglement explained simply")]), tmp_path)
    db.save_transcript(Transcript("vid2", "en", False, [TranscriptSegment(0, 5, "how to bake a simple cake")]), tmp_path)
    hits = db.search_transcripts("quantum", data_dir=tmp_path)
    assert len(hits) == 1
    assert hits[0]["video_id"] == "vid1"
    assert "quantum" in hits[0]["snippet"].lower()
    # multi-term query is an AND: both words must appear in the same transcript
    assert db.search_transcripts("quantum entanglement", data_dir=tmp_path)[0]["video_id"] == "vid1"
    assert db.search_transcripts("quantum cake", data_dir=tmp_path) == []
    assert db.search_transcripts("", data_dir=tmp_path) == []


def test_fts_search_fallback_without_fts5(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_fts5_available", lambda: False)
    db.init_db(tmp_path)
    db.save_transcript(Transcript("vid1", "en", False, [TranscriptSegment(0, 5, "quantum entanglement")]), tmp_path)
    db.save_transcript(Transcript("vid2", "en", False, [TranscriptSegment(0, 5, "how to bake a cake")]), tmp_path)
    hits = db.search_transcripts("quantum", data_dir=tmp_path)
    assert len(hits) == 1 and hits[0]["video_id"] == "vid1"
    assert hits[0]["snippet"]


def test_fts_search_rebuilds_missing_index(tmp_path):
    db.init_db(tmp_path)
    db.save_transcript(Transcript("vid1", "en", False, [TranscriptSegment(0, 5, "neural networks learn")]), tmp_path)
    # Simulate a DB created before FTS existed: drop the index, then search.
    with db.get_connection(tmp_path) as conn:
        conn.execute("DROP TABLE IF EXISTS transcripts_fts")
        conn.commit()
    hits = db.search_transcripts("neural", data_dir=tmp_path)
    assert len(hits) == 1 and hits[0]["video_id"] == "vid1"


def test_verify_insight_grounding_and_timestamp():
    transcript = Transcript("x", "en", False, [
        TranscriptSegment(0, 2, "welcome to the intro"),
        TranscriptSegment(120, 5, "now we cover attention layers in transformers"),
    ])
    grounded = Insight(insight="Attention", explanation="e", timestamp="00:05", confidence="high", quote="now we cover attention layers")
    hallucinated = Insight(insight="Fake", explanation="e", timestamp="00:00", confidence="high", quote="quantum flux capacitor")
    no_quote = Insight(insight="No cite", explanation="e", timestamp="00:00", confidence="medium")
    [g, h, n] = analysis.verify_insights(transcript, [grounded, hallucinated, no_quote])

    # grounded: quote found at 02:00, claimed 00:05 -> verified + snapped
    assert g.verified is True
    assert g.timestamp == "02:00"
    assert "justert" in g.verification_reason

    # hallucinated: quote not found -> unverified, confidence downgraded
    assert h.verified is False
    assert h.confidence == "low"
    assert h.verification_reason

    # no quote: not penalized, left unverified
    assert n.verified is False
    assert n.confidence == "medium"
    assert "Ingen sitat" in n.verification_reason


def test_verify_insight_near_timestamp_is_not_snapped():
    transcript = Transcript("x", "en", False, [
        TranscriptSegment(30, 5, "we use a transformer here"),
    ])
    ins = Insight(insight="Transformer", explanation="e", timestamp="00:35", confidence="high", quote="we use a transformer")
    [v] = analysis.verify_insights(transcript, [ins])
    assert v.verified is True
    assert v.timestamp == "00:35"
    assert v.verification_reason == ""


def test_parse_and_format_timestamp_roundtrip():
    assert analysis._parse_timestamp("00:05") == 5
    assert analysis._parse_timestamp("02:00") == 120
    assert analysis._parse_timestamp("01:00:00") == 3600
    assert analysis._parse_timestamp("nonsense") is None
    assert analysis._parse_timestamp(None) is None
    assert analysis._format_timestamp(120) == "02:00"
    assert analysis._format_timestamp(3600) == "01:00:00"


def _make_analysis(vid, n_insights=1):
    return VideoAnalysis(
        video_id=vid,
        summary_no="s",
        insights=[Insight(insight=f"i{k}", explanation="e", timestamp="00:00", confidence="high") for k in range(n_insights)],
        techniques_tools=["python"],
        actionable_steps=["a"],
        relevance_score=3, depth_score=3, practical_score=4, redundancy_score=1,
        recommendation="watch", recommendation_justification="g",
    )


def test_build_video_digests_caps_and_uses_meta():
    a = _make_analysis("v1", n_insights=10)
    digests = cross_video.build_video_digests([a], {"v1": {"title": "My Title"}})
    assert len(digests) == 1
    assert digests[0]["video_id"] == "v1"
    assert digests[0]["title"] == "My Title"
    assert len(digests[0]["innsikter"]) == 6  # capped


def test_synthesize_playlist_requires_two_videos():
    with pytest.raises(ValueError):
        cross_video.synthesize_playlist([_make_analysis("v1")])


def test_synthesize_playlist_validates_invented_video_ids(monkeypatch):
    good = {
        "theme_no": "AI", "shared_topics": ["python"], "contradictions": [],
        "unique_knowledge": [{"video_id": "v1", "knowledge": "x"}], "gaps": [],
        "watch_order": [{"video_id": "v2", "reason": "y"}], "recommendation": "ok",
    }
    monkeypatch.setattr(cross_video, "call_llm", lambda *a, **kw: good)
    result = cross_video.synthesize_playlist([_make_analysis("v1"), _make_analysis("v2")])
    assert result["theme_no"] == "AI"

    bad = dict(good)
    bad["unique_knowledge"] = [{"video_id": "v9", "knowledge": "x"}]  # invented id
    monkeypatch.setattr(cross_video, "call_llm", lambda *a, **kw: bad)
    with pytest.raises(ValueError):
        cross_video.synthesize_playlist([_make_analysis("v1"), _make_analysis("v2")])


def test_save_load_synthesis_roundtrip(tmp_path):
    db.save_synthesis({"theme_no": "x", "video_ids": ["v1"], "created_at": "now"}, tmp_path)
    assert db.load_synthesis(tmp_path)["theme_no"] == "x"
    db.save_synthesis({"theme_no": "y", "video_ids": ["v1"], "created_at": "now2"}, tmp_path)
    assert db.load_synthesis(tmp_path)["theme_no"] == "y"


def test_run_synthesis_end_to_end_mocked(tmp_path, monkeypatch):
    db.init_db(tmp_path)
    now = datetime.utcnow().isoformat()
    for vid in ("v1", "v2"):
        db.save_video(video(vid), tmp_path)
        with db.get_connection(tmp_path) as conn:
            conn.execute(
                "INSERT INTO analyses (video_id, summary_no, insights_json, techniques_tools_json, "
                "actionable_steps_json, relevance_score, depth_score, practical_score, redundancy_score, "
                "recommendation, recommendation_justification, key_sections_json, "
                "estimated_key_watch_minutes, low_confidence, confidence_reason, analyzed_at, "
                "llm_model, estimated_tokens_in, estimated_tokens_out) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (vid, "s", "[]", "[]", "[]", 3, 3, 4, 1, "watch", "g", "[]", 1.0, 0, "", now, "m", 0, 0),
            )
            conn.commit()
    monkeypatch.setattr(orchestrator.Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(orchestrator.Config, "USE_OLLAMA", False)
    monkeypatch.setattr(orchestrator.Config, "LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(orchestrator.Config, "LLM_API_KEY", "k")
    monkeypatch.setattr(orchestrator.Config, "LLM_REASONING_EFFORT", "")
    monkeypatch.setattr(orchestrator.Config, "LLM_INPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(orchestrator.Config, "LLM_OUTPUT_COST_PER_MILLION", None)
    monkeypatch.setattr(orchestrator.Config, "SPENDING_LIMIT_USD", 1.0)

    def fake_synthesize(analyses, video_meta, model=None):
        return {
            "theme_no": "AI", "shared_topics": ["python"], "contradictions": [],
            "unique_knowledge": [{"video_id": "v1", "knowledge": "x"}], "gaps": [],
            "watch_order": [{"video_id": "v2", "reason": "y"}], "recommendation": "ok",
        }
    monkeypatch.setattr(orchestrator, "synthesize_playlist", fake_synthesize)
    result = orchestrator.run_synthesis(force=True)
    assert result is not None
    assert sorted(result["video_ids"]) == ["v1", "v2"]
    assert db.load_synthesis(tmp_path)["theme_no"] == "AI"
