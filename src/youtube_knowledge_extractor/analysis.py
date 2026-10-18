"""Phase 3: AI analysis of transcripts."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from rich.console import Console

from .config import Config
from .db import get_transcript, update_video_status
from .models import Insight, Transcript, Video, VideoAnalysis, VideoStatus


console = Console()


ANALYSIS_PROMPT_TEMPLATE = """Du er en ekspertkunnskapsingeniør. Din oppgave er å analysere et YouTube-video-transkript og trekke ut genuint nyttig kunnskap på norsk.

VIKTIGE REGLER:
- Oppsummer kun faktisk innhold fra transkriptet. Ikke finn på detaljer.
- Skill mellom fakta fra videoen og dine tolkninger.
- Marker usikkerhet tydelig hvis transkriptet er ufullstendig.

OPPGAVE:
1. Lag en konsis oppsummering på norsk (100-200 ord).
2. Trekk ut 5-12 høyverdige innsikter med forklaringer.
3. Identifiser konkrete teknikker, verktøy, algoritmer, rammeverk og arbeidsflyter.
4. Trekk ut handlingsbare steg og eksempler.
5. Bevar tidsstempler der det er relevant.
6. Vurder relevans (1-5), dybde (1-5), praktisk nytte (1-5) og redundans (1-5).
7. Gi anbefaling: se/hele (watch), skim (skim), eller hopp over (skip) med begrunnelse.
8. Foreslå de mest verdifulle delene å se med estimert tid.
9. Marker lav konfidens hvis transkriptet er ufullstendig.

SVAR KUN MED JSON I DETTE FORMATET:
{{
  "summary_no": "...",
  "insights": [
    {{"insight": "...", "explanation": "...", "timestamp": "MM:SS eller tom", "confidence": "high|medium|low"}}
  ],
  "techniques_tools": ["..."],
  "actionable_steps": ["..."],
  "relevance_score": 1-5,
  "depth_score": 1-5,
  "practical_score": 1-5,
  "redundancy_score": 1-5,
  "recommendation": "watch|skim|skip",
  "recommendation_justification": "...",
  "key_sections": [{{"start": "MM:SS", "end": "MM:SS", "description": "..."}}],
  "estimated_key_watch_minutes": 0.0,
  "low_confidence": false,
  "confidence_reason": ""
}}

TRANSCRIPT:
{transcript_text}
"""


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English-like text."""
    return max(1, len(text) // 4)


def estimate_analysis_cost(transcript: Transcript, model: str = "gpt-4o-mini") -> dict[str, float]:
    """Estimate token count and cost for analyzing a transcript."""
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(transcript_text=transcript.full_text[:15000])
    tokens_in = estimate_tokens(prompt)
    # Estimate output ~1500 tokens
    tokens_out = 1500

    # Rough pricing (USD per 1M tokens)
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "llama3.2": {"input": 0.0, "output": 0.0},  # local
    }
    p = pricing.get(model, pricing["gpt-4o-mini"])
    cost = (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "estimated_cost_usd": round(cost, 4),
    }


def call_llm(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Call configured LLM API."""
    model = model or Config.get_llm_model()

    if Config.USE_OLLAMA:
        return _call_ollama(prompt, model)
    else:
        return _call_openai_compatible(prompt, model)


def _call_openai_compatible(prompt: str, model: str) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
    }
    if Config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {Config.LLM_API_KEY}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Du er en kunnskapsanalytiker som svarer på norsk."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        Config.LLM_API_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def _call_ollama(prompt: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3},
    }
    resp = requests.post(
        Config.OLLAMA_URL,
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("response", "")
    # Ollama may return markdown fences; strip them
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())


def analyze_video(
    video: Video,
    transcript: Transcript | None = None,
    force: bool = False,
) -> VideoAnalysis | None:
    """Analyze a single video transcript.

    Returns VideoAnalysis or None if analysis fails.
    """
    if transcript is None:
        transcript = get_transcript(video.video_id, data_dir=Config.DATA_DIR)
    if transcript is None:
        console.print(f"  [yellow]No transcript for {video.video_id}, skipping analysis[/yellow]")
        return None

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        transcript_text=transcript.full_text[:15000]
    )

    try:
        raw = call_llm(prompt)
        analysis = VideoAnalysis(
            video_id=video.video_id,
            summary_no=raw.get("summary_no", ""),
            insights=[
                Insight(
                    insight=i.get("insight", ""),
                    explanation=i.get("explanation", ""),
                    timestamp=i.get("timestamp") or None,
                    confidence=i.get("confidence", "medium"),
                )
                for i in raw.get("insights", [])
            ],
            techniques_tools=raw.get("techniques_tools", []),
            actionable_steps=raw.get("actionable_steps", []),
            relevance_score=raw.get("relevance_score", 0),
            depth_score=raw.get("depth_score", 0),
            practical_score=raw.get("practical_score", 0),
            redundancy_score=raw.get("redundancy_score", 0),
            recommendation=raw.get("recommendation", ""),
            recommendation_justification=raw.get("recommendation_justification", ""),
            key_sections=raw.get("key_sections", []),
            estimated_key_watch_minutes=raw.get("estimated_key_watch_minutes", 0.0),
            low_confidence=raw.get("low_confidence", False),
            confidence_reason=raw.get("confidence_reason", ""),
            llm_model=Config.get_llm_model(),
            estimated_tokens_in=estimate_tokens(prompt),
            estimated_tokens_out=estimate_tokens(json.dumps(raw)),
        )
        update_video_status(video.video_id, VideoStatus.ANALYSIS_OK, data_dir=Config.DATA_DIR)
        console.print(f"  [green]Analysis OK for {video.video_id}[/green]")
        return analysis
    except Exception as e:
        update_video_status(
            video.video_id,
            VideoStatus.ANALYSIS_FAILED,
            error_message=str(e)[:500],
            data_dir=Config.DATA_DIR,
        )
        console.print(f"  [red]Analysis failed for {video.video_id}: {e}[/red]")
        return None
