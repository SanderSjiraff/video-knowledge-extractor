"""Tests for TikTok slideshow (photo carousel) support and job kinds."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from video_knowledge_extractor import jobs
from video_knowledge_extractor import slides


def _slideshow_html(item_id, desc, author, n_images, scope_key="webapp.reflow.video.detail"):
    """Build a synthetic TikTok mobile page with the rehydration JSON blob."""
    images = [
        {
            "imageURL": {"urlList": [f"https://cdn.example.com/img{i}.jpeg"]},
            "imageWidth": 1080,
            "imageHeight": 1440,
        }
        for i in range(n_images)
    ]
    item = {
        "id": item_id,
        "desc": desc,
        "author": {"uniqueId": author},
        "imagePost": {"images": images},
    }
    data = {"__DEFAULT_SCOPE__": {scope_key: {"statusCode": 0, "itemInfo": {"itemStruct": item}}}}
    return (
        "<html><body>"
        f'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">'
        f"{json.dumps(data)}"
        "</script></body></html>"
    )


URL = "https://www.tiktok.com/@someone/photo/7697007762215030048"


# ---------------------------------------------------------------------------
# HTML parsing (offline)
# ---------------------------------------------------------------------------
def test_parse_slideshow_html() -> None:
    html = _slideshow_html("7697007762215030048", "My AI setup #coding", "someone", 3)
    info = slides.parse_slideshow_html(html, URL)
    assert info["item_id"] == "7697007762215030048"
    assert info["desc"] == "My AI setup #coding"
    assert info["author"] == "someone"
    assert info["images"] == [
        "https://cdn.example.com/img0.jpeg",
        "https://cdn.example.com/img1.jpeg",
        "https://cdn.example.com/img2.jpeg",
    ]


def test_parse_falls_back_to_item_id_when_url_has_no_photo_id() -> None:
    html = _slideshow_html("1234567890", "no url id", "someone", 1)
    info = slides.parse_slideshow_html(html, "https://www.tiktok.com/@someone/")
    assert info["item_id"] == "1234567890"


def test_parse_raises_without_rehydration_block() -> None:
    with pytest.raises(slides.SlideshowError):
        slides.parse_slideshow_html("<html><body>no json</body></html>", URL)


def test_parse_raises_without_item_struct() -> None:
    data = {"__DEFAULT_SCOPE__": {"webapp.app-context": {"language": "en"}}}
    html = (
        '<html><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">'
        f"{json.dumps(data)}</script></html>"
    )
    with pytest.raises(slides.SlideshowError):
        slides.parse_slideshow_html(html, URL)


def test_parse_raises_without_images() -> None:
    html = _slideshow_html("12345", "no images", "someone", 0)
    with pytest.raises(slides.SlideshowError):
        slides.parse_slideshow_html(html, URL)


def test_fetch_rejects_non_photo_url(monkeypatch) -> None:
    monkeypatch.setattr(
        slides, "resolve_photo_url", lambda u: "https://www.tiktok.com/@someone/video/123"
    )
    with pytest.raises(slides.SlideshowError):
        slides.fetch_slideshow("https://vm.tiktok.com/xyz/")


def test_download_slides(tmp_path, monkeypatch) -> None:
    class FakeResp:
        status_code = 200
        content = b"\xff\xd8fakejpeg"

        def raise_for_status(self) -> None:
            pass

    seen: list[str] = []

    def fake_get(url, **_kw):
        seen.append(url)
        return FakeResp()

    monkeypatch.setattr(slides.requests, "get", fake_get)
    paths = slides.download_slides(["a", "b", "c"], tmp_path / "slides" / "photo-1")
    assert [p.name for p in paths] == ["slide-01.jpg", "slide-02.jpg", "slide-03.jpg"]
    assert all(p.read_bytes() == b"\xff\xd8fakejpeg" for p in paths)
    assert seen == ["a", "b", "c"]


def test_guess_language() -> None:
    assert slides._guess_language("Hei, dette er en test med æ og ø") == "no"
    assert slides._guess_language("Hello world, this is a setup") == "en"


# ---------------------------------------------------------------------------
# Job queue: kind column
# ---------------------------------------------------------------------------
def test_job_kind_defaults_to_video(tmp_path) -> None:
    job = jobs.enqueue_job("web", "https://youtu.be/dQw4w9WgXcQ", data_dir=tmp_path)
    assert job["kind"] == "video"


def test_update_job_kind(tmp_path) -> None:
    job = jobs.enqueue_job("web", URL, platform="tiktok", data_dir=tmp_path)
    jobs.update_job(job["job_id"], kind="slides", data_dir=tmp_path)
    assert jobs.get_job(job["job_id"], data_dir=tmp_path)["kind"] == "slides"
    with pytest.raises(ValueError):
        jobs.update_job(job["job_id"], kind="bogus", data_dir=tmp_path)


def test_kind_migration_adds_column_to_old_table(tmp_path) -> None:
    """Databases created before the kind column must be migrated in place."""
    db_file = tmp_path / "knowledge_base.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, source TEXT, video_id TEXT, title TEXT,
            url TEXT, platform TEXT, options_json TEXT, status TEXT,
            progress INTEGER, note TEXT, error_message TEXT,
            created_at TEXT, updated_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (job_id, source, url, status, progress, created_at, updated_at) "
        "VALUES ('old', 'web', 'u', 'done', 100, 't', 't')"
    )
    conn.commit()
    conn.close()

    jobs.init_jobs_table(tmp_path)
    job = jobs.get_job("old", data_dir=tmp_path)
    assert job is not None
    assert job["kind"] == "video"  # default backfilled
