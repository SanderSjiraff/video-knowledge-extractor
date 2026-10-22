"""Telegram send-channel: forward links or videos to the bot, it queues them.

The bot uses long polling, so it works from a home connection with no public
webhook.  It shares the same job queue as the web UI, so videos sent from the
phone end up in the same knowledge base.

Setup (once):
  1. Talk to @BotFather on Telegram: /newbot -> copy the token.
  2. .env:  TELEGRAM_BOT_TOKEN=...
            TELEGRAM_ALLOWED_IDS=<your numeric Telegram user id>
     (find your id by messaging @userinfobot).  If the allow-list is empty the
     bot refuses everyone -- it is safe by default.
  3. Run:  python src/cli.py bot
"""

from __future__ import annotations

import asyncio
import html
import logging
from pathlib import Path
from typing import Optional

from .config import Config
from .db import get_progress_stats, init_db, search_transcripts
from .jobs import list_jobs
from .models import VideoStatus

log = logging.getLogger("vke.telegram")

# Max file size the bot will download (Bot API hard limit is 20 MB without a
# local Bot API server).
MAX_UPLOAD_MB = 19

URL_HINT = (
    "Sende videoer til meg:\n"
    "• Del en TikTok/Instagram-video til Telegram og velg meg, eller\n"
    "• Lim inn en YouTube/TikTok/Instagram-lenke, eller\n"
    "• Send en video-/lydfil direkte.\n\n"
    "Kommandoer: /status, /search <term>, /cost"
)


def allowed_user_ids() -> set[int]:
    raw = Config.TELEGRAM_ALLOWED_IDS or ""
    ids = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    return ids


def _guard(update) -> bool:
    """Allow-list check.  With an empty allow-list nobody is allowed."""
    user = update.effective_user
    if user is None:
        return False
    allowed = allowed_user_ids()
    if not allowed:
        return False
    return user.id in allowed


def _fmt_job_line(job: dict) -> str:
    status_icons = {
        "queued": "⏳",
        "discovering": "🔎",
        "transcribing": "🎧",
        "ocr": "📝",
        "vision": "👁",
        "analyzing": "🧠",
        "embedding": "🔢",
        "done": "✅",
        "failed": "❌",
        "cancelled": "✖️",
    }
    icon = status_icons.get(job["status"], "•")
    title = job["title"] or job["url"]
    if len(title) > 60:
        title = title[:57] + "…"
    line = f"{icon} {html.escape(title)}"
    if job["status"] == "failed":
        err = (job["error_message"] or "")[:80]
        line += f"\n    {html.escape(err)}"
    return line


def build_application():
    """Create the PTB Application with all handlers attached."""
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    init_db(data_dir=Config.DATA_DIR)

    def _guarded(handler):
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not _guard(update):
                if update.effective_message:
                    await update.effective_message.reply_text(
                        "⛔ Ikke tillatt. Legg fra ditt Telegram-ID i TELEGRAM_ALLOWED_IDS."
                    )
                return
            await handler(update, context)

        return wrapper

    # ------------------------------------------------------------------ commands

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.effective_message.reply_text(URL_HINT)

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = get_progress_stats(data_dir=Config.DATA_DIR)
        lines = [
            "📊 Kunnskapsbase",
            f"• Videoer: {stats['total_videos']}  (transkribert {stats['transcripts_ok']}, analysert {stats['analyses_ok']})",
            f"• Total varighet: {stats['total_duration_formatted']}",
        ]
        jobs = list_jobs(limit=8, data_dir=Config.DATA_DIR)
        active = [j for j in jobs if j["status"] not in ("done", "failed", "cancelled")]
        if active:
            lines.append("")
            lines.append("Kø:")
            lines.extend(_fmt_job_line(j) for j in active)
        elif jobs:
            lines.append("")
            lines.append("Siste jobber:")
            lines.extend(_fmt_job_line(j) for j in jobs[:3])
        await update.effective_message.reply_text("\n".join(lines))

    async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = " ".join(context.args or [])
        if not query:
            await update.effective_message.reply_text("Bruk: /search <term>")
            return
        hits = search_transcripts(query, limit=5, data_dir=Config.DATA_DIR)
        if not hits:
            await update.effective_message.reply_text(f"Ingen treff for «{query}».")
            return
        from .db import get_all_videos

        videos = {v.video_id: v for v in get_all_videos(data_dir=Config.DATA_DIR)}
        lines = [f"🔎 «{query}» — {len(hits)} treff:"]
        for hit in hits:
            v = videos.get(hit["video_id"])
            title = v.title if v else hit["video_id"]
            snippet = hit["snippet"][:90]
            lines.append(f"• {html.escape(title)}\n  {html.escape(snippet)}")
        await update.effective_message.reply_text("\n".join(lines))

    async def cmd_cost(update: Update, context: ContextTypes.DEFAULT_TYPE):
        from .orchestrator import run_cost_estimate

        try:
            info = run_cost_estimate()
        except ValueError as exc:
            await update.effective_message.reply_text(f"Kostnadsestimat feilet: {exc}")
            return
        lines = [
            "💰 Neste analysebatch",
            f"• Videoer klare: {info['videos_ready']}",
            f"• Estimert: {info['tokens_in']:,} inn / {info['tokens_out']:,} ut token",
            f"• Estimert pris: ${info['estimated_cost_usd']:.4f} USD",
            f"• Grense: ${info['spending_limit_usd']:.2f} USD",
            f"• Modell: {info['model']} ({'lokal' if Config.USE_OLLAMA else 'API'})",
        ]
        await update.effective_message.reply_text("\n".join(lines))

    # ------------------------------------------------------------------ messages

    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        from .discovery import detect_platform
        from .jobs import enqueue_job

        text = update.effective_message.text or ""
        urls = [u for u in _find_urls(text) if detect_platform(u)]
        if not urls:
            await update.effective_message.reply_text(
                "Jeg fant ingen YouTube/TikTok/Instagram-lenke i meldingen. " + URL_HINT
            )
            return
        added = []
        for url in urls:
            job = enqueue_job("telegram", url, platform=detect_platform(url), data_dir=Config.DATA_DIR)
            added.append(job)
        lines = [f"📨 Mottatt {len(added)} video(er) — i kø:"]
        for job in added:
            lines.append(_fmt_job_line(job))
        await update.effective_message.reply_text("\n".join(lines))

    async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
        from .jobs import enqueue_job

        msg = update.effective_message
        media = msg.video or msg.document or msg.audio or msg.animation
        if media is None:
            return
        size_mb = (media.file_size or 0) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            await msg.reply_text(
                f"⚠️ Fila er {size_mb:.0f} MB — boten kan bare laste ned opptil "
                f"{MAX_UPLOAD_MB} MB. Send lenken i stedet, eller filen via "
                "web-UI-et."
            )
            return
        suffix = ".mp4"
        if media.file_name and "." in media.file_name:
            suffix = "." + media.file_name.rsplit(".", 1)[-1].lower()
        uploads_dir = Config.DATA_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / f"tg_{context.bot_data.get('n', 0) + 1}_{media.file_id[-12:]}{suffix}"
        tfile = await media.get_file()
        await tfile.download_to_drive(str(dest))
        context.bot_data["n"] = context.bot_data.get("n", 0) + 1

        title = (msg.caption or media.file_name or "video").strip()[:120]
        job = enqueue_job(
            "telegram", str(dest), platform="local", title=title, data_dir=Config.DATA_DIR
        )
        await msg.reply_text("\n".join(["📥 Mottatt fil fra Telegram — i kø:", _fmt_job_line(job)]))

    application = (
        Application.builder()
        .token(Config.TELEGRAM_BOT_TOKEN)
        .build()
    )
    application.add_handler(CommandHandler("start", _guarded(cmd_start)))
    application.add_handler(CommandHandler("status", _guarded(cmd_status)))
    application.add_handler(CommandHandler("search", _guarded(cmd_search)))
    application.add_handler(CommandHandler("cost", _guarded(cmd_cost)))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _guarded(on_text)))
    application.add_handler(
        MessageHandler(
            filters.VIDEO | filters.Document.ALL | filters.AUDIO | filters.ANIMATION,
            _guarded(on_media),
        )
    )
    return application


def _find_urls(text: str) -> list[str]:
    import re

    return re.findall(r"https?://[^\s\"'<>)\]]+", text or "")


def run_bot() -> None:
    """Blocking entry point: run long polling until interrupted."""
    if not Config.TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN er ikke satt. Lag en bot med @BotFather og "
            "legg tokenet i .env. (Tom TELEGRAM_ALLOWED_IDS = boten svarer på ingen.)"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from telegram.error import InvalidToken

    try:
        application = build_application()
        log.info("VKE Telegram-bot starter (long polling). Ctrl+C stenger.")
        application.run_polling()
    except InvalidToken:
        raise SystemExit(
            "Telegram avviste tokenet (401 Unauthorized).\n"
            "Kontroller at TELEGRAM_BOT_TOKEN i .env er kopiert helt og holdent fra\n"
            "@BotFather: /mybots -> velg boten din -> API Token -> kopier alt."
        )
