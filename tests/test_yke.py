"""Tests for Video Knowledge Extractor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from video_knowledge_extractor.discovery import (
    extract_video_id,
    parse_duration_to_seconds,
)
from video_knowledge_extractor.analysis import estimate_tokens, estimate_analysis_cost
from video_knowledge_extractor.models import Transcript, TranscriptSegment, Video


def test_extract_video_id_standard() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short() -> None:
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_shorts() -> None:
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid() -> None:
    assert extract_video_id("not a url") is None


def test_parse_duration_iso() -> None:
    assert parse_duration_to_seconds("PT1H23M45S") == 5025


def test_parse_duration_hms() -> None:
    assert parse_duration_to_seconds("1:23:45") == 5025


def test_parse_duration_ms() -> None:
    assert parse_duration_to_seconds("12:34") == 754


def test_parse_duration_empty() -> None:
    assert parse_duration_to_seconds("") == 0
    assert parse_duration_to_seconds(None) == 0


def test_estimate_tokens() -> None:
    text = "This is a test sentence with exactly enough words."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert tokens == len(text) // 4


def test_estimate_analysis_cost() -> None:
    transcript = Transcript(
        video_id="test",
        language="en",
        is_generated=False,
        segments=[
            TranscriptSegment(start=0, duration=5, text="Hello world " * 100),
        ],
    )
    cost = estimate_analysis_cost(transcript, model="gpt-4o-mini")
    assert "tokens_in" in cost
    assert "tokens_out" in cost
    assert "estimated_cost_usd" in cost
    assert cost["estimated_cost_usd"] >= 0


def test_video_duration_formatted() -> None:
    v = Video(
        video_id="test",
        title="Test",
        url="https://example.com",
        duration_seconds=3661,
        channel="Test",
        playlist_order=1,
    )
    assert v.duration_formatted == "01:01:01"

    v2 = Video(
        video_id="test2",
        title="Test2",
        url="https://example.com",
        duration_seconds=65,
        channel="Test",
        playlist_order=2,
    )
    assert v2.duration_formatted == "01:05"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
