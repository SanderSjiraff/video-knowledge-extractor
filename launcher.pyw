"""VKE Platform – desktop launcher.

Double-click (or run the built "VKE Platform.exe"):
  1. Starts API + worker + Telegram bot + dashboard (if not already running).
  2. Starts the Cloudflare quick tunnel (if not already running).
  3. Opens the dashboard in your browser (tunnel URL when available,
     otherwise localhost).

Service output is written to data/logs/<service>.log.  The tunnel URL is
cached in data/tunnel_url.txt so a later run can reuse it.

Set VKE_NO_BROWSER=1 / VKE_NO_POPUP=1 to skip those steps (for testing).
"""
from __future__ import annotations

import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# When frozen by PyInstaller (onefile), __file__ points at a temp dir;
# fall back to the real project location.
if getattr(sys, "frozen", False):
    ROOT = Path(r"C:\Users\Sander Dahl\Desktop\videoer med info\video-knowledge-extractor")
else:
    ROOT = Path(__file__).resolve().parent

VENV = Path(r"C:\vke_venv\Scripts")
PYTHON = VENV / "python.exe"
STREAMLIT = VENV / "streamlit.exe"
LOGS = ROOT / "data" / "logs"
TUNNEL_URL_FILE = ROOT / "data" / "tunnel_url.txt"
DETACH = 0x00000008 | 0x00000010  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def port_open(port: int) -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def process_running(image: str) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return image.lower() in out.lower()
    except Exception:
        return False


def spawn(name: str, cmd: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logf = open(LOGS / f"{name}.log", "ab")
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACH,
        close_fds=True,
    )


def start_tunnel() -> str | None:
    """Start cloudflared quick tunnel and wait for its trycloudflare URL."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(ROOT),
        creationflags=DETACH,
        text=True,
        errors="replace",
        close_fds=True,
    )
    q: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        try:
            for line in proc.stdout:
                q.put(line)
        except Exception:
            pass

    threading.Thread(target=reader, daemon=True).start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            line = q.get(timeout=1)
        except queue.Empty:
            continue
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            TUNNEL_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
            TUNNEL_URL_FILE.write_text(url, encoding="utf-8")
            return url
    return None


def show_popup(url: str) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("VKE Platform")
    root.configure(bg="#0d1117")
    root.resizable(False, False)
    tk.Label(
        root, text="VKE Platform er startet", bg="#0d1117", fg="#f0f6fc",
        font=("Segoe UI", 12, "bold"),
    ).pack(padx=18, pady=(16, 4))
    tk.Label(
        root, text=url, bg="#0d1117", fg="#58a6ff", font=("Consolas", 9),
        wraplength=380, justify="left",
    ).pack(padx=18, pady=4)
    tk.Label(
        root, text="Dashboardet åpnes i nettleseren.",
        bg="#0d1117", fg="#8b949e", font=("Segoe UI", 9),
    ).pack(padx=18, pady=(0, 12))
    tk.Button(
        root, text="Lukk", bg="#21262d", fg="#c9d1d9", activebackground="#30363d",
        relief="flat", padx=24, pady=4, command=root.destroy,
    ).pack(pady=(0, 12))
    root.mainloop()


def main() -> None:
    if not port_open(8501):
        spawn("api", [str(PYTHON), "src/cli.py", "api"])
        spawn("worker", [str(PYTHON), "src/cli.py", "worker"])
        spawn("bot", [str(PYTHON), "src/cli.py", "bot"])
        spawn("dashboard", [str(STREAMLIT), "run", "src/video_knowledge_extractor/dashboard.py"])
        for _ in range(90):
            if port_open(8501):
                break
            time.sleep(1)

    url = None
    if not process_running("cloudflared.exe") and port_open(8501):
        url = start_tunnel()
    if url is None and TUNNEL_URL_FILE.exists():
        url = TUNNEL_URL_FILE.read_text(encoding="utf-8").strip() or None

    target = url or "http://localhost:8501"
    if not os.environ.get("VKE_NO_BROWSER"):
        webbrowser.open(target)
    if not os.environ.get("VKE_NO_POPUP"):
        try:
            show_popup(target)
        except Exception:
            pass


if __name__ == "__main__":
    main()
