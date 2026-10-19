"""Embedding-based semantic search for RAG / context retrieval.

Complements the keyword (FTS5) search with meaning-based retrieval: chunks of
each transcript are embedded, and at query time the question is embedded and
chunks are ranked by cosine similarity. That is much better than keywords for
"find the part that explains X" retrieval when building LLM chat context.

Providers
---------
``ollama`` (default, local, free): a small embedding model such as
``nomic-embed-text`` or ``mxbai-embed-large``.

``api``: an OpenAI-compatible ``/embeddings`` endpoint (set ``EMBED_API_URL``).

The embedding model is a separate, small model. It does not use the
Chat-Completions analysis endpoint, so it does not change the analysis cost
model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console

from .config import Config
from .models import Video
from .analysis import chunk_transcript


console = Console()


def _ollama_embed_url() -> str:
    """Ollama embeddings endpoint, from EMBED_URL or derived from OLLAMA_URL."""
    if Config.EMBED_URL:
        return Config.EMBED_URL
    base = (Config.OLLAMA_URL or "").strip()
    if not base:
        return ""
    if "/api/" in base:
        return base.rsplit("/api/", 1)[0] + "/api/embeddings"
    return base.rstrip("/") + "/api/embeddings"


def embeddings_configured() -> bool:
    """Whether an embedding model is available for the configured provider."""
    provider = (Config.EMBED_PROVIDER or "ollama").lower()
    if provider == "ollama":
        return bool(_ollama_embed_url()) and bool(Config.EMBED_MODEL or "nomic-embed-text")
    return bool(Config.EMBED_API_URL or Config.LLM_API_URL) and bool(Config.EMBED_MODEL)


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    url = _ollama_embed_url()
    if not url:
        raise RuntimeError(
            "EMBED_PROVIDER=ollama needs an Ollama URL (set OLLAMA_URL or EMBED_URL)."
        )
    model = Config.EMBED_MODEL or "nomic-embed-text"
    out: list[list[float]] = []
    for t in texts:
        resp = requests.post(url, json={"model": model, "prompt": t}, timeout=180)
        resp.raise_for_status()
        emb = resp.json().get("embedding")
        if not emb:
            raise RuntimeError("Ollama returned an empty embedding.")
        out.append(emb)
    return out


def _embed_api(texts: list[str]) -> list[list[float]]:
    url = Config.EMBED_API_URL or Config.LLM_API_URL
    if not url:
        raise RuntimeError("EMBED_PROVIDER=api needs EMBED_API_URL (or an embeddings LLM_API_URL).")
    headers = {"Content-Type": "application/json"}
    key = Config.EMBED_API_KEY or Config.LLM_API_KEY
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.post(
        url, headers=headers, json={"model": Config.EMBED_MODEL, "input": texts}, timeout=180
    )
    resp.raise_for_status()
    data = resp.json()
    items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in items]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of (non-empty) texts; returns one vector per input."""
    provider = (Config.EMBED_PROVIDER or "ollama").lower()
    if provider == "ollama":
        return _embed_ollama(texts)
    if provider == "api":
        return _embed_api(texts)
    raise ValueError(f"Unknown EMBED_PROVIDER {Config.EMBED_PROVIDER!r} (use 'ollama' or 'api')")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search_embeddings(
    query: str, data_dir: Path = Path("data"), top_k: int = 5
) -> list[dict]:
    """Rank stored transcript chunks by cosine similarity to the query.

    Returns up to ``top_k`` dicts: ``{video_id, chunk_index, score, snippet}``
    in descending score order. Returns [] for an empty query or no embeddings.
    """
    if not query or not query.strip():
        return []
    from .db import get_all_embeddings

    stored = get_all_embeddings(data_dir=data_dir)
    if not stored:
        return []
    qvec = embed_texts([query.strip()])[0]
    results = [
        {
            "video_id": e["video_id"],
            "chunk_index": e["chunk_index"],
            "score": _cosine(qvec, e["vector"]),
            "snippet": e["text"],
        }
        for e in stored
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[: max(0, int(top_k))]


def process_embeddings(videos: list[Video], force: bool = False) -> dict[str, int]:
    """Build embeddings for a list of videos that have transcripts."""
    from .db import get_transcript, has_embeddings, save_embeddings

    if not embeddings_configured():
        console.print("[red]No embedding model configured — set EMBED_MODEL and EMBED_PROVIDER.[/red]")
        return {"success": 0, "failed": len(videos)}

    total = len(videos)
    success = failed = 0
    for i, video in enumerate(videos, 1):
        if video.status is None or str(video.status) == "inaccessible":
            continue
        if not force and has_embeddings(video.video_id, data_dir=Config.DATA_DIR):
            success += 1
            console.print(f"  [{i}/{total}] [green]cached[/green] {video.video_id}")
            continue
        transcript = get_transcript(video.video_id, data_dir=Config.DATA_DIR)
        if transcript is None or not transcript.full_text.strip():
            console.print(f"  [{i}/{total}] [yellow]no transcript[/yellow] {video.video_id}")
            failed += 1
            continue
        console.print(f"  [{i}/{total}] {video.video_id} — {video.title}")
        try:
            chunks = chunk_transcript(
                transcript,
                chunk_chars=Config.EMBED_CHUNK_CHARS,
                overlap_chars=max(0, Config.EMBED_CHUNK_CHARS // 15),
            )
            chunks = [c for c in chunks if c.strip()]
            if not chunks:
                failed += 1
                console.print(f"      [red]FAILED[/red]: no embeddable chunks")
                continue
            vectors = embed_texts(chunks)
            if len(vectors) != len(chunks):
                raise RuntimeError("embedding count mismatch")
            save_embeddings(
                video.video_id,
                [{"chunk_index": idx, "text": t, "vector": v} for idx, (t, v) in enumerate(zip(chunks, vectors))],
                data_dir=Config.DATA_DIR,
            )
            success += 1
            console.print(f"      [green]OK[/green] {len(chunks)} chunk(s) embedded")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            console.print(f"      [red]FAILED[/red]: {exc}")
    return {"success": success, "failed": failed}
