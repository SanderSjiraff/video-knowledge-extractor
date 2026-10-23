"""Tests for GitHub repo ingestion as a knowledge source."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from video_knowledge_extractor import github_sources
from video_knowledge_extractor.db import get_transcript, get_video, init_db
from video_knowledge_extractor.discovery import detect_platform, parse_github_repo
from video_knowledge_extractor.github_sources import (
    GithubError,
    _is_excluded,
    _pick_code_files,
    process_github_repo,
)
from video_knowledge_extractor.analysis import build_analysis_prompt
from video_knowledge_extractor.models import Transcript, TranscriptSegment, Video, VideoStatus


# ── URL parsing ─────────────────────────────────────────────────────────


def test_detect_platform_github() -> None:
    assert detect_platform("https://github.com/owner/repo") == "github"
    assert detect_platform("https://github.com/owner/repo/blob/main/src/main.py") == "github"


def test_parse_github_repo_root() -> None:
    assert parse_github_repo("https://github.com/owner/repo") == ("owner", "repo")


def test_parse_github_repo_subpages() -> None:
    assert parse_github_repo("https://github.com/owner/repo/blob/main/src/main.py") == ("owner", "repo")
    assert parse_github_repo("https://github.com/owner/repo/tree/dev/docs") == ("owner", "repo")
    assert parse_github_repo("https://github.com/owner/repo/issues") == ("owner", "repo")


def test_parse_github_repo_git_suffix() -> None:
    assert parse_github_repo("https://github.com/owner/repo.git") == ("owner", "repo")


def test_parse_github_repo_invalid() -> None:
    assert parse_github_repo("https://github.com/owner") is None
    assert parse_github_repo("https://gist.github.com/owner/1234") is None
    assert parse_github_repo("https://github.com/topics/python") is None
    assert parse_github_repo("https://example.com/owner/repo") is None
    assert parse_github_repo("not a url") is None


# ── File filtering heuristics ───────────────────────────────────────────


def test_is_excluded_variants() -> None:
    assert _is_excluded("node_modules/pkg/index.js")
    assert _is_excluded("tests/test_main.py")
    assert _is_excluded("src/test_util.py")
    assert _is_excluded("src/util_test.go")
    assert _is_excluded("assets/logo.png")
    assert _is_excluded("package-lock.json")
    assert _is_excluded("dist/bundle.min.js")
    assert _is_excluded(".github/workflows/ci.yml")
    assert not _is_excluded("src/main.py")
    assert not _is_excluded("docs/guide.md")
    assert not _is_excluded(".env.example")


def _tree_entry(path: str, size: int = 1000, type_: str = "blob") -> dict:
    return {"path": path, "type": type_, "size": size}


def test_pick_code_files_prefers_entry_points() -> None:
    tree = [
        _tree_entry("README.md", 1234),
        _tree_entry("docs/guide.md", 900),
        _tree_entry("src/main.py", 3000),
        _tree_entry("src/util.py", 800),
        _tree_entry("src/deep/nested/thing.py", 600),
        _tree_entry("tests/test_main.py", 500),
        _tree_entry("node_modules/x/index.js", 300),
        _tree_entry("assets/logo.png", 1000),
    ]
    picked = _pick_code_files(tree)
    paths = [p for p, _ in picked]
    assert "src/main.py" in paths
    assert paths[0] == "src/main.py"
    assert all("tests" not in p and "node_modules" not in p and "assets" not in p for p in paths)
    assert all(p.endswith((".py", ".js", ".ts")) for p in paths)


def test_pick_code_files_skips_stubs_and_huge_files() -> None:
    tree = [
        _tree_entry("src/tiny.py", 100),        # stub
        _tree_entry("src/huge.py", 999_999),    # dump
        _tree_entry("src/good.py", 5000),
    ]
    paths = [p for p, _ in _pick_code_files(tree)]
    assert paths == ["src/good.py"]


# ── Fetch + pipeline integration ────────────────────────────────────────


class FakeResp:
    def __init__(self, text: str = "", json_data=None, status_code: int = 200, headers=None):
        self.text = text
        self._json = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._json


META = {
    "full_name": "o/r",
    "description": "Test repo",
    "stargazers_count": 42,
    "forks_count": 3,
    "open_issues_count": 1,
    "language": "Python",
    "license": {"spdx_id": "MIT"},
    "topics": ["ai"],
    "homepage": "",
    "pushed_at": "2026-10-01T00:00:00Z",
    "default_branch": "main",
}

TREE = [
    _tree_entry("README.md", 1234),
    _tree_entry("docs/guide.md", 900),
    _tree_entry("src/main.py", 3000),
    _tree_entry("src/util.py", 800),
    _tree_entry("tests/test_main.py", 500),
    _tree_entry("assets/logo.png", 1000),
]

FILE_TEXTS = {
    "docs/guide.md": "# Guide\n\nHow to use r.",
    "src/main.py": "print('main')\n" * 20,
    "src/util.py": "def util(): pass",
}


def _route(url: str, headers: dict | None = None, timeout: int = 30) -> FakeResp:
    if url.endswith("/repos/o/r"):
        return FakeResp(json_data=META)
    if url.endswith("/repos/o/r/readme"):
        return FakeResp(text="# r\n\nA test repo.")
    if "/git/trees/main" in url:
        return FakeResp(json_data={"tree": TREE})
    m = re.search(r"/contents/(.+)\?ref=main$", url)
    if m:
        return FakeResp(text=FILE_TEXTS.get(m.group(1), ""))
    raise AssertionError(f"unexpected URL: {url}")


def _fake_requests_get(monkeypatch, route) -> None:
    """Patch requests.get so _api_get's status-code handling still runs."""
    monkeypatch.setattr(github_sources.requests, "get", route)


def _github_video() -> Video:
    return Video(
        video_id="gh-o-r",
        title="o/r",
        url="https://github.com/o/r",
        duration_seconds=0,
        channel="GitHub",
        playlist_order=0,
        platform="github",
    )


def test_process_github_repo_builds_transcript(tmp_path, monkeypatch) -> None:
    _fake_requests_get(monkeypatch, _route)
    init_db(tmp_path)
    video = _github_video()

    transcript = process_github_repo(video, data_dir=tmp_path)

    text = transcript.full_text
    assert "[REPO-OVERSIKT]" in text
    assert "[README.md]" in text
    assert "A test repo." in text
    assert "[docs/guide.md]" in text
    assert "[FILSTRUKTUR]" in text
    assert "[KILDEKODE: src/main.py]" in text
    # Excluded paths never make it into the content.
    assert "tests/test_main.py" not in text
    assert "node_modules" not in text

    stored = get_transcript("gh-o-r", data_dir=tmp_path)
    assert stored is not None and len(stored.segments) >= 4

    v = get_video("gh-o-r", data_dir=tmp_path)
    assert v is not None
    assert v.status == VideoStatus.TRANSCRIPT_OK
    assert v.channel == "GitHub (o)"
    assert v.url == "https://github.com/o/r"


def test_process_github_repo_no_content(tmp_path, monkeypatch) -> None:
    def empty_get(url, headers=None, timeout=30) -> FakeResp:
        if url.endswith("/repos/o/r"):
            return FakeResp(json_data=META)
        if url.endswith("/repos/o/r/readme"):
            return FakeResp(status_code=404)
        if "/git/trees/main" in url:
            return FakeResp(json_data={"tree": [{"path": "assets/logo.png", "type": "blob", "size": 10}]})
        raise AssertionError(f"unexpected URL: {url}")

    _fake_requests_get(monkeypatch, empty_get)
    init_db(tmp_path)
    with pytest.raises(GithubError):
        process_github_repo(_github_video(), data_dir=tmp_path)


def test_github_rate_limit_error(tmp_path, monkeypatch) -> None:
    def limited(url, headers=None, timeout=30) -> FakeResp:
        return FakeResp(
            status_code=403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1234567890"},
        )

    _fake_requests_get(monkeypatch, limited)
    with pytest.raises(GithubError, match="ratelimited"):
        process_github_repo(_github_video(), data_dir=tmp_path)


def test_github_missing_repo_error(tmp_path, monkeypatch) -> None:
    def missing(url, headers=None, timeout=30) -> FakeResp:
        return FakeResp(status_code=404)

    _fake_requests_get(monkeypatch, missing)
    with pytest.raises(GithubError, match="Fant ikke repoet"):
        process_github_repo(_github_video(), data_dir=tmp_path)


def test_github_invalid_url_raises(tmp_path) -> None:
    v = _github_video()
    v.url = "https://github.com/only-owner"
    with pytest.raises(GithubError, match="gyldig GitHub-repolenke"):
        process_github_repo(v, data_dir=tmp_path)


# ── Worker + analysis wiring ────────────────────────────────────────────


def test_worker_resolves_github_url() -> None:
    from video_knowledge_extractor.worker import JobAborted, resolve_video

    video = resolve_video("https://github.com/SomeOwner/SomeRepo", "github")
    assert video.video_id == "gh-someowner-somerepo"
    assert video.platform == "github"
    assert video.url == "https://github.com/SomeOwner/SomeRepo"

    with pytest.raises(JobAborted):
        resolve_video("https://github.com/only-owner", "github")


def test_make_repo_thumbnail_creates_jpg(tmp_path) -> None:
    video = _github_video()
    out = github_sources.make_repo_thumbnail(video, META, tmp_path)
    assert out is not None
    assert out.name == "gh-o-r.jpg"
    assert out.exists() and out.stat().st_size > 0


def test_wrap_lines_truncates_with_ellipsis() -> None:
    from PIL import Image, ImageDraw

    from video_knowledge_extractor.github_sources import _load_font, _wrap_lines

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    font = _load_font(12)
    lines = _wrap_lines(draw, "word " * 50, font, 100, 2)
    assert len(lines) == 2
    assert lines[1].endswith("…")


def test_analysis_prompt_github_wording() -> None:
    transcript = Transcript(
        video_id="gh-o-r",
        language="code",
        is_generated=True,
        segments=[TranscriptSegment(start=0, duration=0, text="[README.md]\nSome content here.")],
    )
    prompt, truncated = build_analysis_prompt(transcript, platform="github")
    assert "GitHub-repo" in prompt
    assert "filsti" in prompt
    assert not truncated

    video_prompt, _ = build_analysis_prompt(transcript)
    assert "YouTube-video-transkript" in video_prompt
