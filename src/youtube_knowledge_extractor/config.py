"""Configuration management for YouTube Knowledge Extractor."""

from __future__ import annotations

import os
import math
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration from environment variables."""

    # LLM settings
    LLM_API_URL: str = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Ollama local mode
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Cost controls
    SPENDING_LIMIT_USD: float = float(os.getenv("SPENDING_LIMIT_USD", "5.0"))
    DRY_RUN_LIMIT: int = int(os.getenv("DRY_RUN_LIMIT", "5"))
    RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "1.0"))
    LLM_INPUT_COST_PER_MILLION = os.getenv("LLM_INPUT_COST_PER_MILLION", "").strip() or None
    LLM_OUTPUT_COST_PER_MILLION = os.getenv("LLM_OUTPUT_COST_PER_MILLION", "").strip() or None
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "").strip()
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))

    # On-screen text capture (OCR of video frames). Opt-in via the
    # 'screen-text' command; ffmpeg + an OCR engine must be installed.
    VISUAL_FRAME_INTERVAL: float = float(os.getenv("VISUAL_FRAME_INTERVAL", "10"))
    VISUAL_MAX_FRAMES: int = int(os.getenv("VISUAL_MAX_FRAMES", "200"))
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "easyocr").strip().lower()

    # Vision-model descriptions of on-screen visuals (code, diagrams, UI).
    # Opt-in via the 'describe-frames' command; costs vision tokens, so the
    # number of frames is capped. Provider 'ollama' (local, free) or 'api'
    # (OpenAI-compatible Chat Completions endpoint that accepts images).
    VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "ollama").strip().lower()
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llava").strip()
    VISION_MAX_FRAMES: int = int(os.getenv("VISION_MAX_FRAMES", "6"))
    VISION_API_URL: str = os.getenv("VISION_API_URL", "").strip()

    @classmethod
    def validate(cls) -> None:
        for name, value in (("SPENDING_LIMIT_USD", cls.SPENDING_LIMIT_USD), ("RATE_LIMIT_SECONDS", cls.RATE_LIMIT_SECONDS)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if cls.DRY_RUN_LIMIT < 1 or cls.LLM_MAX_OUTPUT_TOKENS < 1:
            raise ValueError("DRY_RUN_LIMIT and LLM_MAX_OUTPUT_TOKENS must be at least 1")
        rates = (cls.LLM_INPUT_COST_PER_MILLION, cls.LLM_OUTPUT_COST_PER_MILLION)
        if (rates[0] is None) != (rates[1] is None):
            raise ValueError("Configure both LLM_INPUT_COST_PER_MILLION and LLM_OUTPUT_COST_PER_MILLION")
        for value in rates:
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError("LLM cost rates must be finite and nonnegative")
        if cls.LLM_REASONING_EFFORT and cls.LLM_REASONING_EFFORT not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("LLM_REASONING_EFFORT must be low, medium, high, or xhigh")
        if "qwen3.8-27b-fp8" in cls.get_llm_model().lower() and cls.LLM_REASONING_EFFORT == "high":
            raise ValueError("Qwen3.8-27B-FP8 does not support high reasoning effort")

    COOKIES_FILE: str | None = os.getenv("YOUTUBE_COOKIES_FILE")

    # Paths
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("data/cache")
    TRANSCRIPTS_DIR: Path = Path("data/transcripts")
    ANALYSES_DIR: Path = Path("data/analyses")
    EXPORTS_DIR: Path = Path("data/exports")
    AUDIO_DIR: Path = Path("data/audio")
    VIDEO_DIR: Path = Path("data/video")

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.DATA_DIR, cls.CACHE_DIR, cls.TRANSCRIPTS_DIR, cls.ANALYSES_DIR, cls.EXPORTS_DIR, cls.AUDIO_DIR, cls.VIDEO_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def is_paid_api_ready(cls) -> bool:
        return bool(cls.LLM_API_KEY) and not cls.USE_OLLAMA

    @classmethod
    def is_ollama_ready(cls) -> bool:
        return cls.USE_OLLAMA

    @classmethod
    def get_llm_model(cls) -> str:
        if cls.USE_OLLAMA:
            return cls.OLLAMA_MODEL
        return cls.LLM_MODEL
