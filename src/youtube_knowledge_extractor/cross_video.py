"""Phase 4: Cross-video knowledge synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from .models import VideoAnalysis


console = Console()


@dataclass
class VideoCrossRef:
    video_id: str
    title: str
    relevance_score: int
    practical_score: int
    techniques: list[str] = field(default_factory=list)


@dataclass
class TopicCluster:
    name: str
    videos: list[VideoCrossRef] = field(default_factory=list)
    unique_insights: list[str] = field(default_factory=list)
    overlapping_concepts: list[str] = field(default_factory=list)


def cluster_videos(analyses: list[VideoAnalysis], video_titles: dict[str, str]) -> list[TopicCluster]:
    """Simple keyword-based clustering of videos into topics."""
    # Gather all techniques/tools as keywords
    all_keywords: dict[str, list[VideoCrossRef]] = {}
    for a in analyses:
        ref = VideoCrossRef(
            video_id=a.video_id,
            title=video_titles.get(a.video_id, a.video_id),
            relevance_score=a.relevance_score,
            practical_score=a.practical_score,
            techniques=a.techniques_tools,
        )
        for kw in a.techniques_tools:
            kw_lower = kw.lower().strip()
            if kw_lower not in all_keywords:
                all_keywords[kw_lower] = []
            if ref not in all_keywords[kw_lower]:
                all_keywords[kw_lower].append(ref)

    # Group keywords that appear together frequently
    clusters: list[TopicCluster] = []
    assigned: set[str] = set()

    for kw, refs in sorted(all_keywords.items(), key=lambda x: -len(x[1])):
        if len(refs) < 2 or kw in assigned:
            continue
        cluster_videos = list(refs)
        topic = TopicCluster(
            name=kw.title(),
            videos=cluster_videos,
            overlapping_concepts=[kw],
        )
        assigned.add(kw)
        # Find related keywords that share many videos
        for other_kw, other_refs in all_keywords.items():
            if other_kw == kw or other_kw in assigned:
                continue
            shared = len({r.video_id for r in cluster_videos} & {r.video_id for r in other_refs})
            if shared >= max(2, len(cluster_videos) // 2):
                topic.overlapping_concepts.append(other_kw)
                for r in other_refs:
                    if r.video_id not in {v.video_id for v in topic.videos}:
                        topic.videos.append(r)
                assigned.add(other_kw)
        clusters.append(topic)

    # Add catch-all for unclustered
    clustered_ids = {v.video_id for c in clusters for v in c.videos}
    unclustered = [a for a in analyses if a.video_id not in clustered_ids]
    if unclustered:
        clusters.append(
            TopicCluster(
                name="Diverse",
                videos=[
                    VideoCrossRef(
                        video_id=a.video_id,
                        title=video_titles.get(a.video_id, a.video_id),
                        relevance_score=a.relevance_score,
                        practical_score=a.practical_score,
                        techniques=a.techniques_tools,
                    )
                    for a in unclustered
                ],
                unique_insights=[],
                overlapping_concepts=[],
            )
        )

    return clusters


def rank_videos_by_value(analyses: list[VideoAnalysis]) -> list[tuple[str, float]]:
    """Rank videos by composite practical value score."""
    scored: list[tuple[str, float]] = []
    for a in analyses:
        # Composite: practical * relevance / (redundancy + 1)
        score = (a.practical_score * a.relevance_score) / (a.redundancy_score + 1)
        scored.append((a.video_id, round(score, 2)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
