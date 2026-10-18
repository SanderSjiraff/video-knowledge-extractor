"""SQLite database for persistent tracking and caching."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import (
    Insight,
    Transcript,
    TranscriptSegment,
    Video,
    VideoAnalysis,
    VideoStatus,
)


def get_db_path(data_dir: Path = Path("data")) -> Path:
    return data_dir / "knowledge_base.db"


@contextmanager
def get_connection(data_dir: Path = Path("data")):
    db_path = get_db_path(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(data_dir: Path = Path("data")) -> None:
    """Initialize database tables."""
    with get_connection(data_dir) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                channel TEXT NOT NULL DEFAULT '',
                playlist_order INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'discovered',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                video_id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                is_generated INTEGER NOT NULL DEFAULT 0,
                segments_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                video_id TEXT PRIMARY KEY,
                summary_no TEXT NOT NULL,
                insights_json TEXT NOT NULL DEFAULT '[]',
                techniques_tools_json TEXT NOT NULL DEFAULT '[]',
                actionable_steps_json TEXT NOT NULL DEFAULT '[]',
                relevance_score INTEGER NOT NULL DEFAULT 0,
                depth_score INTEGER NOT NULL DEFAULT 0,
                practical_score INTEGER NOT NULL DEFAULT 0,
                redundancy_score INTEGER NOT NULL DEFAULT 0,
                recommendation TEXT NOT NULL DEFAULT '',
                recommendation_justification TEXT NOT NULL DEFAULT '',
                key_sections_json TEXT NOT NULL DEFAULT '[]',
                estimated_key_watch_minutes REAL NOT NULL DEFAULT 0.0,
                low_confidence INTEGER NOT NULL DEFAULT 0,
                confidence_reason TEXT NOT NULL DEFAULT '',
                analyzed_at TEXT NOT NULL,
                llm_model TEXT NOT NULL DEFAULT '',
                estimated_tokens_in INTEGER NOT NULL DEFAULT 0,
                estimated_tokens_out INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_videos_order ON videos(playlist_order);
            """
        )
        conn.commit()


def save_video(video: Video, data_dir: Path = Path("data")) -> None:
    """Insert a video or refresh metadata without resetting processing state."""
    with get_connection(data_dir) as conn:
        conn.execute(
            """
            INSERT INTO videos
            (video_id, title, url, duration_seconds, channel, playlist_order,
             status, created_at, updated_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = CASE WHEN excluded.title = '' THEN videos.title ELSE excluded.title END,
                url = excluded.url,
                duration_seconds = CASE WHEN excluded.duration_seconds = 0 THEN videos.duration_seconds ELSE excluded.duration_seconds END,
                channel = CASE WHEN excluded.channel = '' THEN videos.channel ELSE excluded.channel END,
                playlist_order = excluded.playlist_order,
                updated_at = excluded.updated_at,
                status = CASE WHEN videos.status = 'inaccessible' AND excluded.status = 'discovered' THEN excluded.status ELSE videos.status END,
                error_message = CASE WHEN videos.status = 'inaccessible' AND excluded.status = 'discovered' THEN NULL ELSE videos.error_message END
            """,
            (
                video.video_id,
                video.title,
                video.url,
                video.duration_seconds,
                video.channel,
                video.playlist_order,
                video.status.value,
                video.created_at.isoformat(),
                video.updated_at.isoformat(),
                video.error_message,
            ),
        )
        conn.commit()


def get_video(video_id: str, data_dir: Path = Path("data")) -> Optional[Video]:
    with get_connection(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_video(row)


def get_all_videos(data_dir: Path = Path("data")) -> list[Video]:
    with get_connection(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM videos ORDER BY playlist_order"
        ).fetchall()
        return [_row_to_video(row) for row in rows]


def get_analyzed_video_ids(data_dir: Path = Path("data")) -> set[str]:
    with get_connection(data_dir) as conn:
        return {row[0] for row in conn.execute("SELECT video_id FROM analyses")}


def update_video_status(
    video_id: str,
    status: VideoStatus,
    error_message: Optional[str] = None,
    data_dir: Path = Path("data"),
) -> None:
    with get_connection(data_dir) as conn:
        if error_message is not None:
            conn.execute(
                "UPDATE videos SET status = ?, updated_at = ?, error_message = ? WHERE video_id = ?",
                (
                    status.value,
                    datetime.utcnow().isoformat(),
                    error_message,
                    video_id,
                ),
            )
        else:
            conn.execute(
                "UPDATE videos SET status = ?, updated_at = ?, error_message = NULL WHERE video_id = ?",
                (status.value, datetime.utcnow().isoformat(), video_id),
            )
        conn.commit()


def save_transcript(transcript: Transcript, data_dir: Path = Path("data")) -> None:
    segments_json = json.dumps(
        [
            {"start": seg.start, "duration": seg.duration, "text": seg.text}
            for seg in transcript.segments
        ]
    )
    with get_connection(data_dir) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO transcripts
            (video_id, language, is_generated, segments_json, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transcript.video_id,
                transcript.language,
                1 if transcript.is_generated else 0,
                segments_json,
                transcript.fetched_at.isoformat(),
            ),
        )
        conn.commit()


def get_transcript(video_id: str, data_dir: Path = Path("data")) -> Optional[Transcript]:
    with get_connection(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM transcripts WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            return None
        segments = [
            TranscriptSegment(start=s["start"], duration=s["duration"], text=s["text"])
            for s in json.loads(row["segments_json"])
        ]
        return Transcript(
            video_id=row["video_id"],
            language=row["language"],
            is_generated=bool(row["is_generated"]),
            segments=segments,
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )


def _row_to_video(row: sqlite3.Row) -> Video:
    return Video(
        video_id=row["video_id"],
        title=row["title"],
        url=row["url"],
        duration_seconds=row["duration_seconds"],
        channel=row["channel"],
        playlist_order=row["playlist_order"],
        status=VideoStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        error_message=row["error_message"],
    )


def get_progress_stats(data_dir: Path = Path("data")) -> dict[str, Any]:
    with get_connection(data_dir) as conn:
        total = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        discovered = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'discovered'"
        ).fetchone()[0]
        transcript_ok = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status IN ('transcript_ok', 'analysis_pending', 'analysis_ok', 'analysis_failed')"
        ).fetchone()[0]
        transcript_fail = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'transcript_unavailable'"
        ).fetchone()[0]
        analysis_ok = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'analysis_ok'"
        ).fetchone()[0]
        analysis_fail = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'analysis_failed'"
        ).fetchone()[0]
        inaccessible = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'inaccessible'"
        ).fetchone()[0]
        total_duration = conn.execute(
            "SELECT COALESCE(SUM(duration_seconds), 0) FROM videos WHERE status != 'inaccessible'"
        ).fetchone()[0]

    return {
        "total_videos": total,
        "discovered": discovered,
        "transcripts_ok": transcript_ok,
        "transcripts_failed": transcript_fail,
        "analyses_ok": analysis_ok,
        "analyses_failed": analysis_fail,
        "inaccessible": inaccessible,
        "total_duration_seconds": total_duration,
        "total_duration_formatted": _format_duration(total_duration),
    }


def _format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"
