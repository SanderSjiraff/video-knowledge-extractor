"""Dataclasses for the YouTube Knowledge Extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class VideoStatus(str, Enum):
    """Processing status for a video."""

    DISCOVERED = "discovered"
    TRANSCRIPT_PENDING = "transcript_pending"
    TRANSCRIPT_OK = "transcript_ok"
    TRANSCRIPT_UNAVAILABLE = "transcript_unavailable"
    ANALYSIS_PENDING = "analysis_pending"
    ANALYSIS_OK = "analysis_ok"
    ANALYSIS_FAILED = "analysis_failed"
    INACCESSIBLE = "inaccessible"


@dataclass
class Video:
    """Represents a discovered YouTube video."""

    video_id: str
    title: str
    url: str
    duration_seconds: int
    channel: str
    playlist_order: int
    status: VideoStatus = VideoStatus.DISCOVERED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    @property
    def duration_formatted(self) -> str:
        """Return duration as HH:MM:SS or MM:SS."""
        m, s = divmod(self.duration_seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


@dataclass
class TranscriptSegment:
    """A single transcript segment with timestamp."""

    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def timestamp(self) -> str:
        """Format as MM:SS or HH:MM:SS."""
        m, s = divmod(int(self.start), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


@dataclass
class Transcript:
    """Full transcript for a video."""

    video_id: str
    language: str
    is_generated: bool
    segments: list[TranscriptSegment] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class Insight:
    """A single extracted insight."""

    insight: str
    explanation: str
    timestamp: Optional[str] = None
    confidence: str = "medium"  # high, medium, low
    quote: Optional[str] = None  # verbatim transcript excerpt grounding the insight
    verified: bool = False  # quote found in the transcript
    verification_reason: str = ""  # human note when the quote/timestamp failed checks


@dataclass
class VideoAnalysis:
    """AI-generated analysis for a video."""

    video_id: str
    summary_no: str
    insights: list[Insight] = field(default_factory=list)
    techniques_tools: list[str] = field(default_factory=list)
    actionable_steps: list[str] = field(default_factory=list)
    relevance_score: int = 0  # 1-5
    depth_score: int = 0  # 1-5
    practical_score: int = 0  # 1-5
    redundancy_score: int = 0  # 1-5 (higher = more redundant)
    recommendation: str = ""  # watch / skim / skip
    recommendation_justification: str = ""
    key_sections: list[dict[str, Any]] = field(default_factory=list)
    estimated_key_watch_minutes: float = 0.0
    low_confidence: bool = False
    confidence_reason: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    llm_model: str = ""
    estimated_tokens_in: int = 0
    estimated_tokens_out: int = 0


@dataclass
class ProcessingProgress:
    """Track overall processing progress."""

    total_videos: int = 0
    discovered: int = 0
    transcripts_ok: int = 0
    transcripts_failed: int = 0
    analyses_ok: int = 0
    analyses_failed: int = 0
    total_duration_seconds: int = 0
    estimated_cost_usd: float = 0.0
