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
echo [1/3] Starter API-server...
start "VKE API" cmd /k "C:\vke_venv\Scripts\python.exe src\cli.py api"

:: Start worker
echo [2/3] Starter worker...
start "VKE Worker" cmd /k "C:\vke_venv\Scripts\python.exe src\cli.py worker"

:: Start Streamlit dashboard (vent 2 sek for at API skal starte)
echo [3/3] Starter Streamlit-dashboard...
timeout /t 2 /nobreak >nul
start "VKE Dashboard" cmd /k "C:\vke_venv\Scripts\streamlit.exe run src\video_knowledge_extractor\dashboard.py"

echo.
echo Alle tjenester startet!
echo API:       http://127.0.0.1:8000
echo Dashboard: http://localhost:8501
echo.
pause
