"""Phase 4: Cross-video knowledge synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from .analysis import call_llm, cost_rates, estimate_tokens
from .config import Config
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


SYNTHESIS_PROMPT_TEMPLATE = """Du er en ekspertkunnskapsingeniør. Du får AI-analyser fra flere YouTube-videoer i samme samling. Din oppgave er å lage ÉN samlet syntese som hjelper en leser å forstå hele samlingen raskt: hva som overlapper, hva som strider mot hverandre, og hva hver video bidrar med unikt.

VIKTIGE REGLER:
- Basér deg kun på analysene under. Ikke finn på detaljer.
- Når du refererer til en video, bruk video_id-feltet fra analysene. Oppfinn aldri video-id-er.
- List kun motsetninger som faktisk fremgår (uoverensstemmende råd, tall eller konklusjoner). Hvis ingen, la listen være tom.
- For unik kunnskap: hva har én video som de andre mangler?
- Hold deg til norsk.

ANALYSER (én JSON-objekt per video):
{digests_json}

SVAR KUN MED JSON I DETTO FORMATET:
{{
  "theme_no": "Én-til-to setninger: hva hele samlingen handler om.",
  "shared_topics": ["..."],
  "contradictions": [{{"topic": "...", "videos": ["<video_id>", "<video_id>"], "note": "hva de sier ulike ting om"}}],
  "unique_knowledge": [{{"video_id": "<video_id>", "knowledge": "hva denne videoen bidrar med unikt"}}],
  "gaps": ["..."],
  "watch_order": [{{"video_id": "<video_id>", "reason": "hvorfor og i hvilken rekkefølge denne bør sees"}}],
  "recommendation": "Kort: hva leseren bør ta med seg."
}}
"""


def build_video_digests(
    analyses: list[VideoAnalysis],
    video_meta: dict[str, dict[str, Any]] | None = None,
    max_insights: int = 6,
    max_techniques: int = 10,
) -> list[dict[str, Any]]:
    """Compact per-video digests used as the synthesis prompt's source material."""
    video_meta = video_meta or {}
    digests: list[dict[str, Any]] = []
    for a in analyses:
        meta = video_meta.get(a.video_id, {})
        digests.append(
            {
                "video_id": a.video_id,
                "title": meta.get("title", a.video_id),
                "summary_no": a.summary_no,
                "recommendation": a.recommendation,
                "scores": {
                    "relevans": a.relevance_score,
                    "dybde": a.depth_score,
                    "praktisk": a.practical_score,
                    "redundans": a.redundancy_score,
                },
                "teknikker": a.techniques_tools[:max_techniques],
                "innsikter": [
                    {"innsikt": i.insight, "tidsstempel": i.timestamp}
                    for i in a.insights[:max_insights]
                ],
            }
        )
    return digests


def _synthesis_prompt(digests: list[dict[str, Any]]) -> str:
    return SYNTHESIS_PROMPT_TEMPLATE.format(
        digests_json=json.dumps(digests, ensure_ascii=False, indent=2)
    )


def estimate_synthesis_cost(
    analyses: list[VideoAnalysis],
    video_meta: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, float]:
    """Estimate token/cost for one playlist-synthesis call."""
    Config.validate()
    model = model or Config.get_llm_model()
    prompt = _synthesis_prompt(build_video_digests(analyses, video_meta))
    tokens_in = estimate_tokens(prompt)
    tokens_out = Config.LLM_MAX_OUTPUT_TOKENS
    rates = cost_rates(model)
    cost = (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "estimated_cost_usd": cost}


def validate_synthesis_payload(raw: Any, video_ids: set[str]) -> None:
    """Reject malformed synthesis JSON or invented video ids before persisting."""
    if not isinstance(raw, dict):
        raise ValueError("Synthesis response must be a JSON object")
    if not isinstance(raw.get("theme_no"), str) or not raw["theme_no"].strip():
        raise ValueError("Invalid synthesis theme_no")
    for name in ("shared_topics", "gaps"):
        if not isinstance(raw.get(name), list) or any(not isinstance(x, str) for x in raw[name]):
            raise ValueError(f"Invalid synthesis list: {name}")
    for name in ("contradictions", "unique_knowledge", "watch_order"):
        if not isinstance(raw.get(name), list):
            raise ValueError(f"Invalid synthesis list: {name}")
    for c in raw["contradictions"]:
        if (
            not isinstance(c, dict)
            or not isinstance(c.get("topic"), str)
            or not isinstance(c.get("note"), str)
            or not isinstance(c.get("videos"), list)
            or len(c["videos"]) < 2
            or any(not isinstance(v, str) or v not in video_ids for v in c["videos"])
        ):
            raise ValueError("Invalid synthesis contradiction")
    for u in raw["unique_knowledge"]:
        if (
            not isinstance(u, dict)
            or not isinstance(u.get("knowledge"), str)
            or not isinstance(u.get("video_id"), str)
            or u["video_id"] not in video_ids
        ):
            raise ValueError("Invalid synthesis unique_knowledge")
    for w in raw["watch_order"]:
        if (
            not isinstance(w, dict)
            or not isinstance(w.get("reason"), str)
            or not isinstance(w.get("video_id"), str)
            or w["video_id"] not in video_ids
        ):
            raise ValueError("Invalid synthesis watch_order")
    if not isinstance(raw.get("recommendation"), str):
        raise ValueError("Invalid synthesis recommendation")


def synthesize_playlist(
    analyses: list[VideoAnalysis],
    video_meta: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Produce one combined synthesis across multiple analyzed videos.

    Identifies shared topics, contradictions, and per-video unique knowledge via
    a single LLM call over compact per-video digests.
    """
    if len(analyses) < 2:
        raise ValueError("Need at least two analyzed videos to synthesize a playlist")
    prompt = _synthesis_prompt(build_video_digests(analyses, video_meta))
    raw = call_llm(prompt, model)
    validate_synthesis_payload(raw, {a.video_id for a in analyses})
    return raw
