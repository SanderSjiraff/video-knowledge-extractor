"""SQLite database for persistent tracking and caching."""

from __future__ import annotations

import json
import re
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


_FTS5_AVAILABLE: Optional[bool] = None


def _fts5_available() -> bool:
    """Whether this SQLite build supports the FTS5 module (cached)."""
    global _FTS5_AVAILABLE
    if _FTS5_AVAILABLE is None:
        try:
            probe = sqlite3.connect(":memory:")
            probe.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
            probe.close()
            _FTS5_AVAILABLE = True
        except sqlite3.OperationalError:
            _FTS5_AVAILABLE = False
    return _FTS5_AVAILABLE


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
        if _fts5_available():
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5("
                    "video_id UNINDEXED, full_text, tokenize='unicode61')"
                )
            except sqlite3.OperationalError:
                pass
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
        if _fts5_available():
            try:
                full_text = " ".join(seg.text for seg in transcript.segments)
                conn.execute("DELETE FROM transcripts_fts WHERE video_id = ?", (transcript.video_id,))
                conn.execute(
                    "INSERT INTO transcripts_fts (video_id, full_text) VALUES (?, ?)",
                    (transcript.video_id, full_text),
                )
            except sqlite3.OperationalError:
                pass
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


def _transcript_full_text(segments_json: str) -> str:
    """Reconstruct plain full text from a stored segments_json column."""
    try:
        segs = json.loads(segments_json)
    except (ValueError, TypeError):
        return ""
    return " ".join(s.get("text", "") for s in segs)


def _fts_match_expression(query: str) -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Each alphanumeric token is wrapped in double quotes (a literal phrase) so
    FTS5 query operators in the input cannot change the query's meaning.
    Space-separated phrases are implicit AND in FTS5.
    """
    tokens = re.findall(r"[0-9a-zA-ZæøåÆØÅ_]+", query)
    if not tokens:
        return ""
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def rebuild_fts(data_dir: Path = Path("data")) -> int:
    """Rebuild the FTS index from all stored transcripts. Returns count indexed."""
    if not _fts5_available():
        return 0
    with get_connection(data_dir) as conn:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5("
            "video_id UNINDEXED, full_text, tokenize='unicode61')"
        )
        rows = conn.execute("SELECT video_id, segments_json FROM transcripts").fetchall()
        conn.execute("DELETE FROM transcripts_fts")
        for row in rows:
            conn.execute(
                "INSERT INTO transcripts_fts (video_id, full_text) VALUES (?, ?)",
                (row["video_id"], _transcript_full_text(row["segments_json"])),
            )
        conn.commit()
    return len(rows)


def search_transcripts(query: str, limit: int = 20, data_dir: Path = Path("data")) -> list[dict[str, Any]]:
    """Full-text keyword search across all transcript content.

    Returns a list of ``{"video_id", "snippet", "rank"}`` ordered by relevance
    (bm25). Uses SQLite FTS5 when available, otherwise a substring fallback.
    """
    query = (query or "").strip()
    if not query:
        return []
    if not _fts5_available():
        return _search_transcripts_fallback(query, limit, data_dir)

    # Self-heal: rebuild if the index is missing or out of sync with transcripts.
    with get_connection(data_dir) as conn:
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM transcripts_fts").fetchone()[0]
        except sqlite3.OperationalError:
            fts_count = -1
        tr_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    if fts_count != tr_count:
        rebuild_fts(data_dir)

    terms = _fts_match_expression(query)
    if not terms:
        return []
    with get_connection(data_dir) as conn:
        try:
            rows = conn.execute(
                """
                SELECT video_id,
                       snippet(transcripts_fts, 1, '«', '»', '…', 12) AS snippet,
                       bm25(transcripts_fts) AS rank
                FROM transcripts_fts
                WHERE transcripts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (terms, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return _search_transcripts_fallback(query, limit, data_dir)
    return [
        {"video_id": r["video_id"], "snippet": r["snippet"], "rank": r["rank"]}
        for r in rows
    ]


def _search_transcripts_fallback(query: str, limit: int, data_dir: Path = Path("data")) -> list[dict[str, Any]]:
    """Substring search used when FTS5 is unavailable (e.g. older SQLite builds)."""
    tokens = re.findall(r"[0-9a-zA-ZæøåÆØÅ_]+", query)
    if not tokens:
        return []
    lowered = [t.lower() for t in tokens]
    with get_connection(data_dir) as conn:
        rows = conn.execute("SELECT video_id, segments_json FROM transcripts").fetchall()
    results: list[dict[str, Any]] = []
    for r in rows:
        text = _transcript_full_text(r["segments_json"]).lower()
        hits = sum(text.count(t) for t in lowered)
        if hits <= 0:
            continue
        idx = text.find(lowered[0])
        snippet = text[max(0, idx - 40): idx + 80].strip()
        results.append({"video_id": r["video_id"], "snippet": "…" + snippet + "…", "rank": -float(hits)})
    results.sort(key=lambda x: x["rank"])
    return results[:limit]


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
