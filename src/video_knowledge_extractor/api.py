"""FastAPI platform API: video submission, job status, live events, search.

Endpoints:
    POST /api/videos          submit URLs (JSON) and/or uploaded files (multipart)
    GET  /api/jobs            recent jobs (newest first)
    GET  /api/jobs/{job_id}   one job
    GET  /api/events          Server-Sent Events: job list changes (live queue)
    GET  /api/videos          library (all videos)
    GET  /api/videos/{vid}    one video with its analysis
    GET  /api/search?q=       keyword + semantic search
    GET  /api/health          server / model mode

Run with ``python src/cli.py api`` (or as part of ``python src/cli.py platform``).
The API is a local single-user tool for now: CORS is wide open and there is no
auth.  Access control comes with the multi-user phase.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from .config import Config
from .db import (
    get_all_videos,
    get_progress_stats,
    init_db,
    search_transcripts,
)
from .discovery import detect_platform
from .jobs import enqueue_job, get_job, init_jobs_table, list_jobs, merge_options


def create_app(data_dir: Optional[Path] = None) -> FastAPI:
    data_dir = data_dir or Config.DATA_DIR
    init_db(data_dir)
    init_jobs_table(data_dir)
    Config.ensure_dirs()

    app = FastAPI(title="VKE Platform API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ submit

    @app.post("/api/videos")
    async def submit_videos(request: Request) -> dict[str, Any]:
        """Submit videos for processing.

        JSON body:  {"urls": ["..."], "options": {...}, "source": "web"}
        Multipart:  fields "urls" (one per line) and "files" (video uploads).
        """
        urls: list[str] = []
        options: dict[str, Any] = {}
        source = "web"

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            if isinstance(payload.get("urls"), list):
                urls = [str(u).strip() for u in payload["urls"] if str(u).strip()]
            options = payload.get("options") or {}
            source = str(payload.get("source") or "web")
        else:
            form = await request.form()
            raw = str(form.get("urls") or "")
            urls = [line.strip() for line in raw.splitlines() if line.strip()]
            options = {}
            try:
                raw_opts = str(form.get("options") or "").strip()
                if raw_opts:
                    options = json.loads(raw_opts)
            except ValueError:
                pass
            source = str(form.get("source") or "web")

        opts = merge_options(options)
        jobs: list[dict[str, Any]] = []

        for url in urls:
            platform = detect_platform(url)
            if platform is None:
                raise HTTPException(status_code=400, detail=f"Usttt plattform: {url}")
            jobs.append(enqueue_job(source, url, platform=platform, options=opts, data_dir=data_dir))

        for upload in await _get_uploads(request):
            dest = _store_upload(upload, data_dir)
            jobs.append(
                enqueue_job(
                    source,
                    str(dest),
                    platform="local",
                    title=upload.filename or dest.name,
                    options=opts,
                    data_dir=data_dir,
                )
            )

        if not jobs:
            raise HTTPException(status_code=400, detail="Ingen videoer angitt (urls tom og ingen filer).")
        return {"jobs": jobs}

    async def _get_uploads(request: Request) -> list[UploadFile]:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            return []
        form = await request.form()
        files = form.getlist("files")
        return [f for f in files if isinstance(f, UploadFile)]

    def _store_upload(upload: UploadFile, base_dir: Path) -> Path:
        suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".m4a", ".mp3", ".wav", ".ogg", ".flac"}:
            raise HTTPException(status_code=400, detail=f"Filetypen {suffix} stttes ikke.")
        uploads_dir = base_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / f"{uuid.uuid4().hex}{suffix}"
        with dest.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                out.write(chunk)
        return dest

    # ------------------------------------------------------------------- jobs

    @app.get("/api/jobs")
    def jobs_list(limit: int = 20) -> dict[str, Any]:
        return {"jobs": list_jobs(limit=min(max(limit, 1), 100), data_dir=data_dir)}

    @app.get("/api/jobs/{job_id}")
    def job_one(job_id: str) -> dict[str, Any]:
        job = get_job(job_id, data_dir=data_dir)
        if job is None:
            raise HTTPException(status_code=404, detail="Jobb ikke funnet.")
        return {"job": job}

    # ------------------------------------------------------------------- events

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        """SSE stream of the current job list; one message per change."""

        def generate():
            last: Optional[str] = None
            while True:
                snapshot = json.dumps(
                    list_jobs(limit=15, data_dir=data_dir), ensure_ascii=False
                )
                if snapshot != last:
                    yield f"data: {snapshot}\n\n"
                    last = snapshot
                else:
                    yield ": keep-alive\n\n"
                time.sleep(1.5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ library

    def _video_dict(v) -> dict[str, Any]:
        return {
            "video_id": v.video_id,
            "title": v.title,
            "url": v.url,
            "duration_seconds": v.duration_seconds,
            "channel": v.channel,
            "status": v.status.value,
            "platform": v.platform,
            "error_message": v.error_message,
            "created_at": v.created_at.isoformat(),
        }

    @app.get("/api/videos")
    def videos_list() -> dict[str, Any]:
        return {"videos": [_video_dict(v) for v in get_all_videos(data_dir=data_dir)]}

    @app.get("/api/videos/{video_id}")
    def video_one(video_id: str) -> dict[str, Any]:
        v = get_all_videos(data_dir=data_dir)
        match = next((x for x in v if x.video_id == video_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Video ikke funnet.")
        out = _video_dict(match)
        from .db import get_all_analyses

        analysis = next((a for a in get_all_analyses(data_dir=data_dir) if a.video_id == video_id), None)
        if analysis is not None:
            out["analysis"] = {
                "summary_no": analysis.summary_no,
                "insights": [
                    {
                        "insight": i.insight,
                        "explanation": i.explanation,
                        "timestamp": i.timestamp,
                        "verified": i.verified,
                        "quote": i.quote,
                    }
                    for i in analysis.insights
                ],
                "recommendation": analysis.recommendation,
                "relevance_score": analysis.relevance_score,
                "depth_score": analysis.depth_score,
                "practical_score": analysis.practical_score,
                "redundancy_score": analysis.redundancy_score,
                "estimated_key_watch_minutes": analysis.estimated_key_watch_minutes,
            }
        return out

    # ------------------------------------------------------------------- search

    @app.get("/api/search")
    def search(q: str = "", top: int = 10) -> dict[str, Any]:
        q = (q or "").strip()
        if not q:
            return {"keyword": [], "semantic": []}
        videos = {v.video_id: v for v in get_all_videos(data_dir=data_dir)}
        keyword = []
        for hit in search_transcripts(q, limit=50, data_dir=data_dir):
            v = videos.get(hit["video_id"])
            keyword.append(
                {
                    "video_id": hit["video_id"],
                    "title": v.title if v else hit["video_id"],
                    "snippet": hit["snippet"],
                }
            )
        semantic = []
        try:
            from .embeddings import embeddings_configured, search_embeddings

            if embeddings_configured():
                for r in search_embeddings(q, data_dir=data_dir, top_k=top):
                    v = videos.get(r["video_id"])
                    semantic.append(
                        {
                            "video_id": r["video_id"],
                            "title": v.title if v else r["video_id"],
                            "snippet": r["snippet"],
                            "score": r["score"],
                        }
                    )
        except Exception:  # noqa: BLE001 - missing embedding model must not break search
            pass
        return {"keyword": keyword, "semantic": semantic}

    # ------------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        stats = get_progress_stats(data_dir=data_dir)
        return {
            "ok": True,
            "mode": "ollama" if Config.USE_OLLAMA else "api",
            "model": Config.get_llm_model(),
            "data_dir": str(Path(data_dir).resolve()),
            "videos_total": stats["total_videos"],
        }

    return app
