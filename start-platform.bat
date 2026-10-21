@echo off
rem Starter VKE-plattformen: API + worker + dashboard + Telegram-bot + cloudflared-tunnel.
rem Kjøres automatisk ved pålogging (oppgaven "VKE Platform" i Oppgavesskjemeren).
cd /d "C:\Users\Sander Dahl\Desktop\videoer med info\video-knowledge-extractor"

start "" ".venv\Scripts\python.exe" src/cli.py platform
rem Dashboardet trenger et par sekunder; botten starter uavhengig.
start "" ".venv\Scripts\python.exe" src/cli.py bot
start "" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8501 --no-autoupdate
