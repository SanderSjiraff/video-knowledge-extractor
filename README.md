# YouTube Knowledge Extractor

A complete local Python application that analyzes YouTube playlists and creates a searchable Norwegian knowledge base. Optimized for extracting genuinely useful knowledge — not generic summaries.

## Features

- **Phase 1 — Discovery**: Retrieve all accessible videos from a playlist or text file with URLs
- **Phase 2 — Transcripts**: Extract English or Norwegian subtitles with timestamps, group into chunks
- **Phase 3 — AI Analysis**: Generate concise Norwegian summaries, insights, technique extraction, and recommendations
- **Phase 4 — Cross-Video Knowledge**: Cluster related videos, identify overlaps and gaps
- **Phase 5 — Dashboard**: Streamlit-based searchable UI with filters, rankings, and exports

## Tech Stack

- Python 3.11+
- `yt-dlp` for playlist/video metadata
- `youtube-transcript-api` for subtitle extraction
- OpenAI-compatible LLM API (or Ollama for local/no-cost mode)
- SQLite for persistent tracking and caching
- Streamlit for the dashboard
- `python-dotenv` for configuration

## Installation

```bash
cd youtube-knowledge-extractor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your API key or Ollama settings.

## Usage

### 1. Discover videos

From a playlist URL:
```bash
python src/cli.py discover "https://www.youtube.com/playlist?list=YOUR_LIST"
```

From a text file with one URL per line:
```bash
python src/cli.py discover "watch-later.txt"
```

With a limit:
```bash
python src/cli.py discover "watch-later.txt" --limit 5
```

### 2. Extract transcripts

```bash
python src/cli.py transcripts
```

Force re-fetch:
```bash
python src/cli.py transcripts --force
```

### 3. Estimate costs

```bash
python src/cli.py cost
```

### 4. Run AI analysis (dry-run default: 5 videos)

```bash
python src/cli.py analyze
```

Process all videos:
```bash
python src/cli.py analyze --limit 9999
```

### 5. Launch dashboard

```bash
python src/cli.py dashboard
```

Or directly:
```bash
streamlit run src/youtube_knowledge_extractor/dashboard.py
```

### Check status

```bash
python src/cli.py status
```

## Project Structure

```
youtube-knowledge-extractor/
├── src/
│   ├── cli.py                              # CLI entry point
│   └── youtube_knowledge_extractor/
│       ├── __init__.py
│       ├── config.py                       # Configuration from environment
│       ├── models.py                       # Pydantic/dataclass models
│       ├── db.py                           # SQLite persistence
│       ├── discovery.py                    # Phase 1: Video discovery
│       ├── transcripts.py                  # Phase 2: Transcript extraction
│       ├── analysis.py                     # Phase 3: AI analysis
│       ├── cross_video.py                  # Phase 4: Cross-video synthesis
│       ├── dashboard.py                    # Phase 5: Streamlit UI
│       └── orchestrator.py                 # Workflow orchestration
├── tests/
│   └── test_yke.py                         # Unit tests
├── data/
│   ├── cache/
│   ├── transcripts/                        # Raw transcript JSON files
│   ├── analyses/                           # Analysis outputs
│   └── exports/                            # Exported reports
├── .env.example                            # Example environment variables
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Cost Controls

- **Dry-run**: Only 5 videos analyzed by default
- **Cost estimation** before any paid API calls
- **Spending limit** in `.env` (`SPENDING_LIMIT_USD`)
- **Local mode** with Ollama (set `USE_OLLAMA=true`)
- **Caching**: Transcripts and analyses are cached; unchanged videos are never re-analyzed
- **Force flag** to override caching

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_URL` | OpenAI-compatible API endpoint | `https://api.openai.com/v1/chat/completions` |
| `LLM_API_KEY` | API key (never commit!) | _(empty)_ |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `USE_OLLAMA` | Use Ollama local LLM | `false` |
| `OLLAMA_URL` | Ollama API URL | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `SPENDING_LIMIT_USD` | Max cost before blocking | `5.00` |
| `DRY_RUN_LIMIT` | Default videos in dry-run | `5` |
| `RATE_LIMIT_SECONDS` | Seconds between API calls | `1.0` |

## Git Setup (Ready for GitHub)

```bash
git init
git add .
git commit -m "Initial commit: YouTube Knowledge Extractor v1.0.0"
# git remote add origin https://github.com/YOUR_USERNAME/youtube-knowledge-extractor.git
# git branch -M main
# git push -u origin main
```

## License

MIT
