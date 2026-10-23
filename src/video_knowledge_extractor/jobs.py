"""SQLite-backed job queue for the send platform.

A *job* is one submitted video (a URL or a local file) that the worker will
push through the pipeline: discover -> transcribe -> [OCR] -> [vision] ->
analyze -> [embed].  Jobs live in the same SQLite database as everything
else, so any process (CLI, Streamlit, Telegram bot, API) can enqueue and
any worker can pick them up.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .db import get_connection

# Pipeline stages, in order.  `queued` means not started, `done`/`failed`
# are terminal.  The worker reports its current stage here.
JOB_STAGES = (
    "queued",
    "discovering",
    "transcribing",
    "ocr",
    "vision",
    "analyzing",
    "embedding",
    "done",
    "failed",
    "cancelled",
)

TERMINAL_STAGES = {"done", "failed", "cancelled"}

# Default pipeline options applied when the submitter does not specify any.
DEFAULT_OPTIONS = {
    "transcribe": True,
    "analyze": True,
    "ocr": False,
    "vision": False,
    "embed": True,
}


def _now() -> str:
    return datetime.utcnow().isoformat()


def init_jobs_table(data_dir: Path = Path("data")) -> None:
    with get_connection(data_dir) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'web',
                video_id TEXT,
                title TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT 'youtube',
                options_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'queued',
                progress INTEGER NOT NULL DEFAULT 0,
                note TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'video'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
        # Migrate databases created before the kind column existed.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        if "kind" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'video'")
        if "pinned" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def merge_options(options: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Validate/normalise submitted options against the defaults."""
    merged = dict(DEFAULT_OPTIONS)
    if options:
        for key in merged:
            if key in options and options[key] is not None:
                merged[key] = bool(options[key])
    return merged


def enqueue_job(
    source: str,
    url: str,
    platform: str = "youtube",
    title: str = "",
    options: Optional[dict[str, Any]] = None,
    data_dir: Path = Path("data"),
) -> dict[str, Any]:
    """Add a video to the queue and return the new job record."""
    init_jobs_table(data_dir)
    job_id = uuid.uuid4().hex
    opts = merge_options(options)
    now = _now()
    with get_connection(data_dir) as conn:
        conn.execute(
            """
            INSERT INTO jobs
            (job_id, source, video_id, title, url, platform, options_json,
             status, progress, note, error_message, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?, ?, ?, 'queued', 0, NULL, NULL, ?, ?)
            """,
            (job_id, source, title, url, platform, json.dumps(opts), now, now),
        )
        conn.commit()
    return get_job(job_id, data_dir=data_dir)


def get_job(job_id: str, data_dir: Path = Path("data")) -> Optional[dict[str, Any]]:
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row is not None else None


def list_jobs(
    limit: int = 20,
    data_dir: Path = Path("data"),
    include_terminal: bool = True,
) -> list[dict[str, Any]]:
    """Recent jobs, newest first.  Non-terminal jobs always come back."""
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        if include_terminal:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status NOT IN ('done', 'failed', 'cancelled') "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_job(r) for r in rows]


def list_active_jobs(data_dir: Path = Path("data")) -> list[dict[str, Any]]:
    """All non-terminal jobs, oldest first (queue processing order).

    Unlike ``list_jobs`` (which is capped at the most recent N rows), this
    always returns every job still in the pipeline.
    """
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status NOT IN ('done', 'failed', 'cancelled') "
            "ORDER BY pinned DESC, created_at"
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def count_active_jobs(data_dir: Path = Path("data")) -> int:
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status NOT IN ('done', 'failed', 'cancelled')"
        ).fetchone()
    return int(row["n"])


def count_finished_jobs(data_dir: Path = Path("data")) -> int:
    """Total number of jobs that reached a terminal state (done or failed).

    Unlike ``list_jobs`` (which is capped), this counts every job ever
    finished — used for the queue stats bar.
    """
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('done', 'failed')"
        ).fetchone()
    return int(row["n"])


def cancel_job(job_id: str, data_dir: Path = Path("data")) -> bool:
    """Cancel a job that is still queued.  Returns False if not possible."""
    init_jobs_table(data_dir)
    now = _now()
    with get_connection(data_dir) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'cancelled', pinned = 0, note = 'Avbrutt av bruker', "
            "updated_at = ? WHERE job_id = ? AND status = 'queued'",
            (now, job_id),
        )
        conn.commit()
    return cur.rowcount > 0


def retry_job(job_id: str, data_dir: Path = Path("data")) -> bool:
    """Re-queue a terminal job (done/failed/cancelled) for another run."""
    init_jobs_table(data_dir)
    now = _now()
    with get_connection(data_dir) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'queued', progress = 0, note = NULL, "
            "error_message = NULL, updated_at = ? "
            "WHERE job_id = ? AND status IN ('done', 'failed', 'cancelled')",
            (now, job_id),
        )
        conn.commit()
    return cur.rowcount > 0


def pin_job(job_id: str, data_dir: Path = Path("data")) -> bool:
    """Move a queued job to the front of the queue."""
    init_jobs_table(data_dir)
    now = _now()
    with get_connection(data_dir) as conn:
        cur = conn.execute(
            "UPDATE jobs SET pinned = 1, updated_at = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (now, job_id),
        )
        conn.commit()
    return cur.rowcount > 0


def unpin_job(job_id: str, data_dir: Path = Path("data")) -> bool:
    """Drag a pinned queued job back down to its normal (FIFO) position."""
    init_jobs_table(data_dir)
    now = _now()
    with get_connection(data_dir) as conn:
        cur = conn.execute(
            "UPDATE jobs SET pinned = 0, updated_at = ? "
            "WHERE job_id = ? AND status = 'queued' AND pinned = 1",
            (now, job_id),
        )
        conn.commit()
    return cur.rowcount > 0


def claim_next_job(data_dir: Path = Path("data")) -> Optional[dict[str, Any]]:
    """Atomically claim the oldest queued job (status queued -> discovering).

    Returns the claimed job, or None when the queue is empty.  The claim is
    a single UPDATE guarded by the queued status, so two workers never pick
    up the same job.
    """
    init_jobs_table(data_dir)
    with get_connection(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' "
            "ORDER BY pinned DESC, created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        now = _now()
        cur = conn.execute(
            "UPDATE jobs SET status = 'discovering', pinned = 0, updated_at = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (now, row["job_id"]),
        )
        conn.commit()
        if cur.rowcount == 0:  # lost a race
            return None
    job = _row_to_job(row)
    job["status"] = "discovering"  # the row snapshot predates the claim UPDATE
    job["updated_at"] = now
    return job


def update_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    video_id: Optional[str] = None,
    title: Optional[str] = None,
    note: Optional[str] = None,
    error_message: Optional[str] = None,
    kind: Optional[str] = None,
    data_dir: Path = Path("data"),
) -> None:
    if status is not None and status not in JOB_STAGES:
        raise ValueError(f"unknown job stage: {status}")
    sets = ["updated_at = ?"]
    values: list[Any] = [_now()]
    if status is not None:
        sets.append("status = ?")
        values.append(status)
    if progress is not None:
        sets.append("progress = ?")
        values.append(int(progress))
    if video_id is not None:
        sets.append("video_id = ?")
        values.append(video_id)
    if title is not None:
        sets.append("title = ?")
        values.append(title)
    if note is not None:
        sets.append("note = ?")
        values.append(note)
    if error_message is not None:
        sets.append("error_message = ?")
        values.append(error_message)
    if kind is not None:
        if kind not in ("video", "slides"):
            raise ValueError(f"unknown job kind: {kind}")
        sets.append("kind = ?")
        values.append(kind)
    values.append(job_id)
    with get_connection(data_dir) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", values)
        conn.commit()


def fail_job(job_id: str, error: str, data_dir: Path = Path("data")) -> None:
    update_job(job_id, status="failed", error_message=error[:500], data_dir=data_dir)


# Stages a worker reports while actively processing a job (not queued/terminal).
IN_FLIGHT_STAGES = ("discovering", "transcribing", "ocr", "vision", "analyzing", "embedding")


def reclaim_orphaned_jobs(data_dir: Path = Path("data")) -> list[str]:
    """Reset in-flight jobs back to the queue.  Returns the reclaimed job ids.

    Called when a worker starts up: any job sitting in an in-flight stage was
    owned by a worker that is no longer alive (it crashed, was killed, or was
    hung and restarted), so it can never finish on its own.  Re-queueing is
    safe because every pipeline stage is idempotent (``force=False``) — an
    already-finished part of the job is simply skipped on the re-run.
    """
    init_jobs_table(data_dir)
    placeholders = ", ".join("?" for _ in IN_FLIGHT_STAGES)
    now = _now()
    with get_connection(data_dir) as conn:
        rows = conn.execute(
            f"SELECT job_id FROM jobs WHERE status IN ({placeholders})",
            IN_FLIGHT_STAGES,
        ).fetchall()
        if rows:
            conn.execute(
                f"UPDATE jobs SET status = 'queued', updated_at = ?, "
                "note = 'Worker ble startet pa nytt - jobben satt tilbake i ko' "
                f"WHERE status IN ({placeholders})",
                [now, *IN_FLIGHT_STAGES],
            )
            conn.commit()
    return [r["job_id"] for r in rows]


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "source": row["source"],
        "video_id": row["video_id"],
        "title": row["title"],
        "url": row["url"],
        "platform": row["platform"],
        "options": json.loads(row["options_json"] or "{}"),
        "status": row["status"],
        "progress": row["progress"],
        "note": row["note"],
        "error_message": row["error_message"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "kind": row["kind"],
        "pinned": bool(row["pinned"]),
    }
