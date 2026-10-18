"""Configuration management for YouTube Knowledge Extractor."""

from __future__ import annotations

import os
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
    # Reasoning effort for models that support it (e.g., Qwen)
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "xhigh")

    # Ollama local mode
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Cost controls
    SPENDING_LIMIT_USD: float = float(os.getenv("SPENDING_LIMIT_USD", "5.0"))
    DRY_RUN_LIMIT: int = int(os.getenv("DRY_RUN_LIMIT", "5"))
    RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "1.0"))

    COOKIES_FILE: str | None = os.getenv("YOUTUBE_COOKIES_FILE")

    # Paths
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("data/cache")
    TRANSCRIPTS_DIR: Path = Path("data/transcripts")
    ANALYSES_DIR: Path = Path("data/analyses")
    EXPORTS_DIR: Path = Path("data/exports")
    AUDIO_DIR: Path = Path("data/audio")

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.DATA_DIR, cls.CACHE_DIR, cls.TRANSCRIPTS_DIR, cls.ANALYSES_DIR, cls.EXPORTS_DIR, cls.AUDIO_DIR]:
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
