@echo off
rem ============================================================
rem  VKE - automatisk spilleliste-sync
rem  Kjoeres av planlagte oppgaven "VKE Playlist Sync".
rem  Henter YouTube-spillelisten og legger NYE videoer i VKE-koen.
rem  Workeren (platform/supervisor) behandler de deretter automatisk.
rem  Logg: data\logs\playlist_sync.log
rem ============================================================
setlocal
chcp 65001 >nul
set "ROOT=C:\Users\Sander Dahl\Desktop\videoer med info\video-knowledge-extractor"
set "URL=https://www.youtube.com/playlist?list=PLWXumi0v4Zjs"
cd /d "%ROOT%"
(
  echo.
  echo ==== %date% %time% - playlist sync ====
  ".venv\Scripts\python.exe" "src\cli.py" sync-playlist "%URL%"
) >> "%ROOT%\data\logs\playlist_sync.log" 2>&1
