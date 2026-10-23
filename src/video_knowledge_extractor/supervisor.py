"""Process supervisor: keeps the VKE platform services alive.

The platform has four long-running services:

    api        FastAPI server             (port 8000)
    worker     job-queue worker           (no port)
    bot        Telegram send-channel bot  (no port)
    dashboard  Streamlit dashboard        (port 8501)

``launcher.pyw`` (or ``start-platform.bat``) spawns this supervisor as a
single detached process, so no terminal windows are needed.  The
supervisor:

* starts any service that is not running yet, and
* restarts a service it started when that service exits, with a back-off
  delay that grows while the service crash-loops.

Services that are already running when the supervisor starts (for example
started manually in a terminal) are left alone; if one of those stops
later, the supervisor takes it over and keeps it running.

All service output is appended to ``data/logs/<service>.log``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "data" / "logs"

DASHBOARD_PORT = 8501
SERVICES = ("api", "worker", "bot", "dashboard")
POLL_SECONDS = 10.0
BASE_RESTART_DELAY = 5.0
MAX_RESTART_DELAY = 120.0
# An exit faster than this many seconds after start counts as a fast crash
# (crash-loop) and increases the restart delay.
FAST_CRASH_WINDOW = 60.0


def _log(msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def port_open(port: int) -> bool:
    import socket

    try:
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def _service_command(name: str) -> list[str]:
    dashboard = ROOT / "src" / "video_knowledge_extractor" / "dashboard.py"
    commands = {
        "api": [sys.executable, "src/cli.py", "api"],
        "worker": [sys.executable, "src/cli.py", "worker"],
        "bot": [sys.executable, "src/cli.py", "bot"],
        "dashboard": [sys.executable, "-m", "streamlit", "run", str(dashboard)],
    }
    return commands[name]


def _external_python_running(marker: str) -> bool:
    """True when a python process whose command line contains *marker* exists."""
    if os.name != "nt":
        return False
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.CommandLine -like '*{marker}*' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def _service_running(name: str) -> bool:
    """Best-effort check whether *name* is running (ours or started elsewhere)."""
    from video_knowledge_extractor.config import Config

    if name == "api":
        return port_open(Config.API_PORT)
    if name == "dashboard":
        return port_open(DASHBOARD_PORT)
    return _external_python_running(f"cli.py {name}")


def _spawn(name: str) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = open(LOGS / f"{name}.log", "ab")
    return subprocess.Popen(
        _service_command(name),
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def supervise(
    poll_seconds: float = POLL_SECONDS,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Keep all platform services running until stop_event is set (or forever)."""
    _log("Supervisor started (api, worker, bot, dashboard).")
    children: dict[str, subprocess.Popen] = {}
    started_at: dict[str, float] = {}
    fast_crashes: dict[str, int] = {}
    suppress_until: dict[str, float] = {}

    while stop_event is None or not stop_event.is_set():
        for name in SERVICES:
            try:
                _supervise_one(
                    name, children, started_at, fast_crashes, suppress_until, stop_event
                )
            except Exception as exc:  # noqa: BLE001 - one bad service must not kill the supervisor
                _log(f"error while supervising {name}: {exc}")
        time.sleep(poll_seconds if stop_event is None else 0.2)

    _log("Supervisor stopping - terminating services it started")
    for name, proc in children.items():
        if proc.poll() is None:
            proc.terminate()


def _supervise_one(
    name: str,
    children: dict[str, subprocess.Popen],
    started_at: dict[str, float],
    fast_crashes: dict[str, int],
    suppress_until: dict[str, float],
    stop_event: Optional[threading.Event],
) -> None:
    """Check one service: restart a dead child of ours, or start it if missing."""
    proc = children.get(name)
    if proc is not None and proc.poll() is None:
        return  # still running
    if proc is not None:
        age = time.monotonic() - started_at.pop(name)
        children.pop(name, None)
        fast_crashes[name] = fast_crashes.get(name, 0) + 1 if age < FAST_CRASH_WINDOW else 0
        delay = min(BASE_RESTART_DELAY * (2 ** fast_crashes[name]), MAX_RESTART_DELAY)
        suppress_until[name] = time.monotonic() + delay
        _log(f"{name} exited (code {proc.returncode}) after {age:.0f}s - restarting in {delay:.0f}s")
        return
    now = time.monotonic()
    if now < suppress_until.get(name, 0.0):
        return  # backing off after a crash
    suppress_until.pop(name, None)
    if _service_running(name):
        return  # already running (started elsewhere)
    _log(f"starting {name}")
    children[name] = _spawn(name)
    started_at[name] = now
