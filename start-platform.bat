@echo off
:: VKE Platform startup script
:: Kobler til virtuelt miljø i C:\vke_venv

title VKE Platform
cd /d "C:\Users\Sander Dahl\Desktop\videoer med info\video-knowledge-extractor"

:: Sjekk at miljøet eksisterer
if not exist "C:\vke_venv\Scripts\python.exe" (
    echo [FEIL] Virtuelt miljoe ikke funnet. Kjoer setup forst.
    pause
    exit /b 1
)

echo ==========================================
echo VKE - Video Knowledge Extractor
echo ==========================================
echo.

:: Start API
echo [1/5] Starter API-server...
start "VKE API" cmd /k "C:\vke_venv\Scripts\python.exe src\cli.py api"

:: Start worker
echo [2/5] Starter worker...
start "VKE Worker" cmd /k "C:\vke_venv\Scripts\python.exe src\cli.py worker"

:: Start Telegram-bot
echo [3/5] Starter Telegram-bot...
start "VKE Bot" cmd /k "C:\vke_venv\Scripts\python.exe src\cli.py bot"

:: Start Streamlit dashboard (vent 2 sek for at API skal starte)
echo [4/5] Starter Streamlit-dashboard...
timeout /t 2 /nobreak >nul
start "VKE Dashboard" cmd /k "C:\vke_venv\Scripts\streamlit.exe run src\video_knowledge_extractor\dashboard.py"

:: Start Cloudflare quick tunnel (dashboard fra telefon)
echo [5/5] Starter Cloudflare-tunnel...
start "VKE Tunnel" cmd /k "cloudflared tunnel --url http://localhost:8501"

echo.
echo Alle tjenester startet!
echo API:       http://127.0.0.1:8000
echo Dashboard: http://localhost:8501
echo Tunnel-URL: se i vinduet "VKE Tunnel" (https://....trycloudflare.com)
echo.
pause
