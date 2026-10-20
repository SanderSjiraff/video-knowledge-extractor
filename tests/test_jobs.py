"""Tests for the send platform: job queue, worker resolution, and the API."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from video_knowledge_extractor import jobs
from video_knowledge_extractor.config import Config
from video_knowledge_extractor.worker import resolve_video
from video_knowledge_extractor.models import VideoStatus


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------
def test_enqueue_and_get(tmp_path) -> None:
    job = jobs.enqueue_job("web", "https://youtu.be/dQw4w9WgXcQ", platform="youtube", data_dir=tmp_path)
    assert job["status"] == "queued"
    assert job["platform"] == "youtube"
    assert job["options"] == jobs.DEFAULT_OPTIONS

    fetched = jobs.get_job(job["job_id"], data_dir=tmp_path)
    assert fetched is not None
    assert fetched["job_id"] == job["job_id"]


def test_enqueue_merges_options(tmp_path) -> None:
    job = jobs.enqueue_job(
        "web",
        "https://youtu.be/dQw4w9WgXcQ",
        options={"ocr": True, "embed": False, "bogus": "ignored"},
        data_dir=tmp_path,
    )
    assert job["options"]["ocr"] is True
    assert job["options"]["embed"] is False
    assert job["options"]["transcribe"] is True  # default kept


def test_get_missing_job(tmp_path) -> None:
    assert jobs.get_job("nope", data_dir=tmp_path) is None


def test_list_jobs_newest_first(tmp_path) -> None:
    a = jobs.enqueue_job("web", "u1", data_dir=tmp_path)
    b = jobs.enqueue_job("web", "u2", data_dir=tmp_path)
    listed = jobs.list_jobs(limit=10, data_dir=tmp_path)
    assert [j["job_id"] for j in listed][:2] == [b["job_id"], a["job_id"]]


def test_claim_next_job(tmp_path) -> None:
    a = jobs.enqueue_job("web", "u1", data_dir=tmp_path)
    b = jobs.enqueue_job("web", "u2", data_dir=tmp_path)

    claimed = jobs.claim_next_job(data_dir=tmp_path)
    assert claimed is not None
    assert claimed["job_id"] == a["job_id"]  # oldest first
    assert claimed["status"] == "discovering"

    # The claim is persisted: a second claim returns the other job.
    second = jobs.claim_next_job(data_dir=tmp_path)
    assert second["job_id"] == b["job_id"]
    assert jobs.claim_next_job(data_dir=tmp_path) is None


def test_claim_empty_queue(tmp_path) -> None:
    jobs.init_jobs_table(tmp_path)
    assert jobs.claim_next_job(data_dir=tmp_path) is None


def test_update_job(tmp_path) -> None:
    job = jobs.enqueue_job("web", "u", data_dir=tmp_path)
    jobs.update_job(job["job_id"], status="transcribing", progress=15, video_id="vid",
                    title="Hei", data_dir=tmp_path)
    fetched = jobs.get_job(job["job_id"], data_dir=tmp_path)
    assert fetched["status"] == "transcribing"
    assert fetched["progress"] == 15
    assert fetched["video_id"] == "vid"
    assert fetched["title"] == "Hei"

    with pytest.raises(ValueError):
        jobs.update_job(job["job_id"], status="nonsense", data_dir=tmp_path)


def test_fail_job(tmp_path) -> None:
    job = jobs.enqueue_job("web", "u", data_dir=tmp_path)
    jobs.fail_job(job["job_id"], "boom", data_dir=tmp_path)
    fetched = jobs.get_job(job["job_id"], data_dir=tmp_path)
    assert fetched["status"] == "failed"
    assert fetched["error_message"] == "boom"


def test_count_active_jobs(tmp_path) -> None:
    a = jobs.enqueue_job("web", "u1", data_dir=tmp_path)
    jobs.enqueue_job("web", "u2", data_dir=tmp_path)
    jobs.enqueue_job("web", "u3", data_dir=tmp_path)
    assert jobs.count_active_jobs(data_dir=tmp_path) == 3
    jobs.update_job(a["job_id"], status="done", progress=100, data_dir=tmp_path)
    assert jobs.count_active_jobs(data_dir=tmp_path) == 2


# ---------------------------------------------------------------------------
# Worker video resolution (no network needed for youtube + local)
# ---------------------------------------------------------------------------
def test_resolve_youtube(tmp_path) -> None:
    video = resolve_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube")
    assert video.video_id == "dQw4w9WgXcQ"
    assert video.platform == "youtube"


def test_resolve_youtube_bad_url() -> None:
    from video_knowledge_extractor.worker import JobAborted

    with pytest.raises(JobAborted):
        resolve_video("https://www.youtube.com/playlist?list=abc", "youtube")


def test_resolve_local_file(tmp_path) -> None:
    target = tmp_path / "opptak.mp4"
    target.write_bytes(b"x")
    video = resolve_video(str(target), "local")
    assert video.platform == "local"
    assert video.title == "opptak"
    assert video.video_id.startswith("local-")
    assert Path(video.url) == target.resolve()

    # Stable id: same path -> same id.
    again = resolve_video(str(target), "local")
    assert again.video_id == video.video_id


# ---------------------------------------------------------------------------
# API (TestClient, everything local)
# ---------------------------------------------------------------------------
def _client(tmp_path):
    from fastapi.testclient import TestClient

    from video_knowledge_extractor.api import create_app

    return TestClient(create_app(data_dir=tmp_path))


def test_api_submit_and_list(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/api/videos",
        json={"urls": ["https://youtu.be/dQw4w9WgXcQ"], "source": "test"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["platform"] == "youtube"

    jobs_resp = client.get("/api/jobs").json()
    assert len(jobs_resp["jobs"]) == 1


def test_api_submit_rejects_unknown_platform(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/videos", json={"urls": ["https://example.com/video"]})
    assert resp.status_code == 400


def test_api_submit_empty(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.post("/api/videos", json={"urls": []}).status_code == 400


def test_api_file_upload(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/api/videos",
        data={"urls": ""},
        files={"files": ("opptak.mp4", b"fake-bytes", "video/mp4")},
    )
    assert resp.status_code == 200
    job = resp.json()["jobs"][0]
    assert job["platform"] == "local"
    assert Path(job["url"]).exists()


def test_api_search_empty_db(tmp_path) -> None:
    client = _client(tmp_path)
    body = client.get("/api/search", params={"q": "nårings"}).json()
    assert body["keyword"] == []
    assert body["semantic"] == []


def test_api_health(tmp_path) -> None:
    client = _client(tmp_path)
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "model" in body
