# YouTube Knowledge Extractor

A lightweight Python tool that discovers YouTube videos, extracts subtitles, and creates a searchable Norwegian knowledge base using an LLM.

## Quick Install
```bash
cd youtube-knowledge-extractor
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and edit your credentials:
```bash
cp .env.example .env   # then edit .env
```

## Core Commands (via `src/cli.py`)
| Command | What it does |
|---------|--------------|
| `discover <source>` | Pull video metadata from a playlist URL or a text file (one URL per line). Use `--limit N` to restrict.
| `transcripts` | Download subtitles (auto‑fallback to `yt‑dlp` when blocked). Use `--force` to re‑fetch.
| `transcribe [SOURCE]` | Local fallback when the subtitle API is IP‑blocked: download each video's audio with `yt‑dlp` and transcribe it with **Whisper** (GPU/CUDA). Options: `--model` (`tiny`/`base`/`small`/`medium`/`large-v3`, default `medium`), `--force`, `--no-keep-audio`. Writes `.txt` + `.srt` to `data/transcripts/`.
| `cost` | Show token & USD estimate for AI analysis.
| `analyze` | Run the LLM on transcripts (dry‑run defaults to 5 videos). Use `--limit N` for more or `--force` to re‑analyze.
| `dashboard` | Launch the Streamlit UI.
| `status` | Show current processing statistics.

## Minimal Project Layout
```
youtube-knowledge-extractor/
├── src/
│   ├── cli.py               # CLI entry point
│   └── youtube_knowledge_extractor/
│       ├── config.py        # Settings via .env
│       ├── db.py            # SQLite persistence
│       ├── discovery.py     # Phase 1 – video discovery
│       ├── transcripts.py   # Phase 2 – subtitle extraction
│       ├── analysis.py      # Phase 3 – LLM analysis
│       ├── cross_video.py   # Phase 4 – clustering
│       ├── dashboard.py     # Phase 5 – Streamlit UI
│       └── orchestrator.py  # Workflow glue
├── tests/                    # Unit tests (11 passing)
├── data/                     # Cached DB, transcripts, analyses (ignored in Git)
├── .env.example              # Example environment variables (remove before publishing)
├── requirements.txt
└── README.md                  # You are reading it!
```

## Cost Controls (at a glance)
- **Dry‑run** processes only the first 5 videos (configurable via `DRY_RUN_LIMIT`).
- `run_cost_estimate` prints token counts and a USD estimate.
- `SPENDING_LIMIT_USD` blocks any call that would exceed the limit.
- Enable `USE_OLLAMA=true` for a free local model.

## Environment Variables (summary)
| Variable | Purpose |
|----------|---------|
| `LLM_API_URL` | OpenAI‑compatible endpoint |
| `LLM_API_KEY` | Your API key (never commit) |
| `LLM_MODEL` | Default model (e.g., `gpt-4o-mini`) |
| `USE_OLLAMA` | Toggle local Ollama mode |
| `OLLAMA_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Local model name |
| `SPENDING_LIMIT_USD` | Max allowed spend |
| `DRY_RUN_LIMIT` | Default dry‑run size |
| `RATE_LIMIT_SECONDS` | Seconds between LLM calls |
| `LLM_REASONING_EFFORT` | Reasoning effort for models like Qwen (`xhigh`, `medium`, `low`) |



### YouTube cookies

Pass a Netscape cookies export to the transcript fallback:

```powershell
python src/cli.py transcripts --cookies "../cookies.txt"
```

You can also set `YOUTUBE_COOKIES_FILE`. Keep this login file private.
