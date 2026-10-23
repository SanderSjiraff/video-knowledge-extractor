@echo off
:: VKE Platform startup script
:: Starter alt via supervisoren - INGEN terminalvinduer behøver å holdes
:: aapne. Supervisoren holder API + worker + Telegram-bot + dashboard i
:: live, og starter pa nytt det som stopper.  Logs: data\logs\

title VKE Platform
cd /d "C:\Users\Sander Dahl\Desktop\videoer med info\video-knowledge-extractor"

:: Sjekk at miljoeet eksisterer
if not exist "C:\vke_venv\Scripts\pythonw.exe" (
    echo [FEIL] Virtuelt miljoe ikke funnet. Kjoer setup forst.
    pause
    exit /b 1
)

start "" "C:\vke_venv\Scripts\pythonw.exe" launcher.pyw
