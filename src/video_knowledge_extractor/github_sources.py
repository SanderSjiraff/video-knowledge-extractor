"""GitHub repositories as a knowledge source.

Treats a GitHub repo as a text source in the same pipeline as videos: repo
metadata, README, markdown docs, a compact file tree and selected source
files are fetched from the GitHub REST API and stored as the source's
"transcript".  The analysis model then works on that text unchanged — insight
quotes verify against it, and an insight "timestamp" becomes a file path
(e.g. ``src/main.py``) that shows up in the notes and dashboard.

Unauthenticated by default (60 API calls/hour per IP, plenty for adding a
repo now and then); set ``GITHUB_TOKEN`` in ``.env`` for 5000 calls/hour and
access to private repos.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

from .config import Config
from .db import save_transcript, save_video, update_video_status
from .discovery import parse_github_repo
from .models import Transcript, TranscriptSegment, Video, VideoStatus


API_BASE = "https://api.github.com"

# Content budgets (chars) so a repo stays analysable without huge prompts.
README_MAX_CHARS = 12_000
DOCS_MAX_CHARS = 12_000      # all markdown docs combined
CODE_MAX_CHARS = 24_000      # all source files combined
TREE_MAX_CHARS = 2_500       # file-structure overview
PER_FILE_MAX_CHARS = 4_000   # a single file is truncated beyond this

MAX_DOC_FILES = 6
MAX_CODE_FILES = 12

# Directories that never carry repo knowledge worth ingesting.
_EXCLUDED_DIRS = {
    ".git", ".github", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "out", "target", ".next", ".nuxt", "coverage", ".idea",
    ".vscode", "vendor", "third_party", "site", "assets", "static", "public",
}
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".tar",
    ".gz", ".mp3", ".mp4", ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".eot",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".wasm", ".parquet", ".npy",
    ".onnx", ".safetensors", ".pt", ".ckpt",
}
_SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".go", ".rs", ".java",
    ".rb", ".php", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp", ".swift", ".kt",
    ".sh", ".ps1", ".bat", ".sql", ".lua", ".zig", ".ex", ".exs", ".clj",
    ".r", ".m", ".vue", ".svelte",
}
_ENTRY_POINTS = {
    "main.py", "app.py", "cli.py", "index.js", "index.ts", "server.py",
    "main.go", "main.rs", "index.go", "lib.rs", "main.ts",
}


class GithubError(Exception):
    """User-facing failure while fetching a GitHub repo."""


def _api_get(url: str, raw: bool = False, timeout: int = 30) -> requests.Response:
    """GET from the GitHub REST API with token support and clear errors."""
    headers = {"User-Agent": "video-knowledge-extractor"}
    headers["Accept"] = "application/vnd.github.raw" if raw else "application/vnd.github+json"
    if Config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {Config.GITHUB_TOKEN}"
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
        raise GithubError(
            "GitHub-APIet er ratelimited (60 kall/time uten token). "
            "Prøv igjen senere, eller sett GITHUB_TOKEN i .env."
        )
    if resp.status_code == 404:
        raise GithubError("Fant ikke repoet (slettet, eller privat uten GITHUB_TOKEN).")
    if resp.status_code >= 400:
        raise GithubError(f"GitHub-APIet svarte {resp.status_code} for {url}")
    return resp


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... resten av fila er klippet ({len(text) - limit} tegn)]"


def _is_excluded(path: str) -> bool:
    """Heuristic: does this path carry useful knowledge for ingestion?"""
    lower = path.lower()
    parts = lower.split("/")
    if any(p in _EXCLUDED_DIRS for p in parts[:-1]):
        return True
    name = parts[-1]
    if name.endswith((".min.js", ".min.css", ".map", ".lock", ".lockb")):
        return True
    if name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "uv.lock"}:
        return True
    if name.startswith(".") and name not in {".env.example", ".env.sample"}:
        return True
    if Path(name).suffix.lower() in _BINARY_EXTS:
        return True
    # Tests rarely carry reusable knowledge; they bloat the context.
    if re.search(r"(_tests?\.|(^|/)tests?/|(^|/)test_[^/]+$|_test\.go$|\.spec\.)", lower):
        return True
    return False


def _doc_priority(path: str) -> tuple:
    """Rank markdown docs: top-level and docs/ first, common names boosted."""
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    depth = lower.count("/")
    boost = 0
    if name in {"contributing.md", "usage.md", "getting-started.md", "installation.md", "guide.md"}:
        boost = -1
    return (0 if boost == -1 else 1, depth, path)


def _pick_code_files(tree: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Select the most informative source files from the git tree.

    Uses the size reported by the tree API (no extra API calls) plus a
    path-based heuristic: entry points first, then shallow src/ files,
    skipping stubs and huge dumps.
    """
    candidates: list[tuple[int, str, int]] = []  # (-score, path, size)
    for entry in tree:
        if entry.get("type") != "blob" or not entry.get("path"):
            continue
        path = entry["path"]
        if _is_excluded(path):
            continue
        name = path.rsplit("/", 1)[-1]
        if Path(name).suffix.lower() not in _SOURCE_EXTS:
            continue
        size = int(entry.get("size") or 0)
        if size < 200 or size > 40_000:
            continue
        score = 0
        if name in _ENTRY_POINTS:
            score += 100
        depth = path.count("/")
        if depth <= 1:
            score += 40
        elif depth <= 2:
            score += 20
        if path.split("/", 1)[0] in {"src", "lib", "app", "internal", "pkg", "cmd"}:
            score += 15
        # Slightly prefer medium-sized files over one-liners and wall-of-code.
        if 500 <= size <= 15_000:
            score += 5
        candidates.append((-score, path, size))
    candidates.sort()
    return [(p, s) for _, p, s in candidates[:MAX_CODE_FILES]]


def _tree_text(tree: list[dict[str, Any]], limit: int = TREE_MAX_CHARS) -> str:
    """Compact two-level file tree so the analysis sees the layout."""
    dirs: dict[str, dict[str, int]] = {}
    top_files: list[str] = []
    for entry in tree:
        if entry.get("type") != "blob" or not entry.get("path"):
            continue
        if _is_excluded(entry["path"]):
            continue
        parts = entry["path"].split("/")
        if len(parts) == 1:
            top_files.append(parts[0])
        elif len(parts) == 2:
            dirs.setdefault(parts[0], {})[parts[1]] = 1
        else:
            dirs.setdefault(parts[0], {}).setdefault(parts[1], 0)
            dirs[parts[0]][parts[1]] += 1
    lines = [f"(toppnivå: {len(top_files)} filer)"]
    for f in top_files[:25]:
        lines.append(f"  {f}")
    for d in sorted(dirs):
        entries = sorted(dirs[d].items())
        lines.append(f"{d}/ ({sum(v for v in dirs[d].values() if isinstance(v, int) and v > 0) or len(entries)} filer)")
        for name, count in entries[:20]:
            suffix = f" +{count - 1} til" if count > 1 else ""
            lines.append(f"  {name}{suffix}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n[…]"


def _overview_text(meta: dict[str, Any]) -> str:
    """Human-readable repo summary block (stars, topics, license, ...)."""
    license_name = (meta.get("license") or {}).get("spdx_id") or "Ukjent"
    lines = [
        f"Repo: {meta.get('full_name', '')}",
        f"Beskrivelse: {meta.get('description') or '—'}",
        f"Stjerner: {meta.get('stargazers_count', 0):,}",
        f"Forks: {meta.get('forks_count', 0):,}",
        f"Åpne issues: {meta.get('open_issues_count', 0)}",
        f"Hovedspråk: {meta.get('language') or '—'}",
        f"Lisens: {license_name}",
        f"Emneord: {', '.join(meta.get('topics', []) or []) or '—'}",
        f"Hjemmeside: {meta.get('homepage') or '—'}",
        f"Oppdatert: {meta.get('pushed_at', '')}",
        f"Default-branch: {meta.get('default_branch', '')}",
    ]
    return "\n".join(lines)


def fetch_repo_content(
    owner: str, repo: str
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Fetch a repo's knowledge content as (label, text) sections.

    Section order: repo overview, README, other docs, file tree, source code.
    Each fetch costs one GitHub API call; the whole run stays well inside
    the anonymous rate limit.
    """
    meta = _api_get(f"{API_BASE}/repos/{owner}/{repo}").json()
    if not isinstance(meta, dict) or not meta.get("full_name"):
        raise GithubError("Uventet svar fra GitHub-APIet (repo-meta).")
    default_branch = meta.get("default_branch") or "main"
    sections: list[tuple[str, str]] = [("REPO-OVERSIKT", _overview_text(meta))]

    try:
        readme = _api_get(f"{API_BASE}/repos/{owner}/{repo}/readme", raw=True).text
        readme = readme.strip()
        if readme:
            sections.append(("README.md", _cap(readme, README_MAX_CHARS)))
    except GithubError:
        pass  # a repo without a README is fine

    tree_resp = _api_get(
        f"{API_BASE}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    )
    tree = tree_resp.json().get("tree", [])
    files = [t["path"] for t in tree if t.get("type") == "blob" and t.get("path")]

    # Other markdown docs (README already fetched via /readme).
    docs = [
        p for p in files
        if p.lower().endswith(".md")
        and p.rsplit("/", 1)[-1].lower() != "readme.md"
        and not _is_excluded(p)
    ]
    docs.sort(key=_doc_priority)
    docs_budget = DOCS_MAX_CHARS
    for path in docs[:MAX_DOC_FILES]:
        if docs_budget <= 0:
            break
        try:
            text = _api_get(
                f"{API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={default_branch}",
                raw=True,
            ).text.strip()
        except GithubError:
            continue
        if not text:
            continue
        take = _cap(text, min(docs_budget, PER_FILE_MAX_CHARS * 2))
        sections.append((path, take))
        docs_budget -= len(take)

    sections.append(("FILSTRUKTUR", _tree_text(tree)))

    # Source code: best candidates by heuristic, one raw fetch each.
    code_budget = CODE_MAX_CHARS
    for path, size in _pick_code_files(tree):
        if code_budget <= 0:
            break
        try:
            text = _api_get(
                f"{API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={default_branch}",
                raw=True,
            ).text
        except GithubError:
            continue
        text = text.strip()
        if not text:
            continue
        take = _cap(text, min(code_budget, PER_FILE_MAX_CHARS))
        sections.append((f"KILDEKODE: {path}", take))
        code_budget -= len(take)

    return meta, sections


def process_github_repo(video: Video, data_dir: Path = Path("data")) -> Transcript:
    """Fetch the repo behind *video.url* and store its content as a transcript.

    Updates the video record (canonical URL, owner as channel) and marks the
    video transcript_ok.  Raises GithubError with a user-facing message on
    failure — the worker turns that into a failed job.
    """
    parsed = parse_github_repo(video.url)
    if parsed is None:
        raise GithubError("Ikke en gyldig GitHub-repolenke")
    owner, repo = parsed
    meta, sections = fetch_repo_content(owner, repo)
    # REPO-OVERSIKT and FILSTRUKTUR always exist; only real content counts.
    has_content = any(
        t.strip() for label, t in sections if label not in {"REPO-OVERSIKT", "FILSTRUKTUR"}
    )
    if not has_content:
        raise GithubError("Repoet har ingen README, dokumentasjon eller kildekode å hente")

    video.title = meta.get("full_name") or f"{owner}/{repo}"
    video.channel = f"GitHub ({owner})"
    video.url = f"https://github.com/{owner}/{repo}"
    video.duration_seconds = 0
    save_video(video, data_dir=data_dir)

    segments: list[TranscriptSegment] = []
    for idx, (label, text) in enumerate(sections):
        text = text.strip()
        if not text:
            continue
        segments.append(TranscriptSegment(start=float(idx), duration=0.0, text=f"[{label}]\n{text}"))
    if not segments:
        raise GithubError("Ingen innhold å hente fra repoet")

    transcript = Transcript(
        video_id=video.video_id,
        language="code",
        is_generated=True,
        segments=segments,
    )
    save_transcript(transcript, data_dir=data_dir)
    update_video_status(video.video_id, VideoStatus.TRANSCRIPT_OK, data_dir=data_dir)
    return transcript
