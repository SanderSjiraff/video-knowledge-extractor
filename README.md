# Video Knowledge Extractor

Et Python-verktøy som henter videometadata og tilgjengelige undertekster fra **YouTube, TikTok og Instagram**, lager norske analyseutkast med en språkmodell og viser resultatene lokalt med Streamlit. Data lagres i SQLite og JSON på maskinen din. I tillegg er det en **send-plattform**: du sender videoer fra nettleseren (lenker eller filer) eller fra telefonen via en Telegram-bot, og en jobb-kø + worker transkriberer og analyserer dem automatisk.

## Installasjon

Du trenger Python 3.10 eller nyere, nettverk for YouTube-innhenting og enten lokal Ollama eller et OpenAI-kompatibelt Chat Completions-endepunkt.

```sh
git clone https://github.com/SanderSjiraff/video-knowledge-extractor.git
cd video-knowledge-extractor
python -m venv .venv
```

**Windows / PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Hvis PowerShell blokkerer aktiveringsskriptet, bruk `.\.venv\Scripts\python.exe` i stedet for `python`. Det krever ingen endring av maskinens sikkerhetspolicy.

**macOS / Linux:**

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Rediger `.env` før analyse. Eksempelfilen inneholder bare plassholdere og velger lokal Ollama. Kjør kommandoene fra prosjektmappen: både `.env` og `data/` er relative til gjeldende mappe. Miljøvariabler som allerede er satt, overstyrer `.env`.

## Modelloppsett

### Lokal Ollama

Installer og start [Ollama](https://ollama.com/), og last ned modellen:

```sh
ollama pull llama3.2
```

Sett `USE_OLLAMA=true`, `OLLAMA_MODEL=llama3.2` og `OLLAMA_URL=http://localhost:11434/api/generate` i `.env`. Programmet bruker `/api/generate`. API-prisestimatet er null i denne modusen; maskinressurser og strøm inngår ikke. Ingen ekstern analysemodell brukes som reserve dersom Ollama feiler.

### OpenAI-kompatibelt API

Sett `USE_OLLAMA=false`, hele endepunktet, modellnavnet og din egen nøkkel. For NTNU kan oppsettet for eksempel være:

```dotenv
USE_OLLAMA=false
LLM_API_URL=https://llm.hpc.ntnu.no/v1/chat/completions
LLM_API_KEY=
LLM_MODEL=Qwen/Qwen3.8-27B-FP8
LLM_REASONING_EFFORT=medium
LLM_INPUT_COST_PER_MILLION=
LLM_OUTPUT_COST_PER_MILLION=
```

Sett begge prisfeltene til leverandørens faktiske USD-priser per million input- og outputtoken. Bruk `0` i begge feltene bare hvis ruten faktisk er gratis for deg. Ukjent modell uten eksplisitte priser stopper før analyse. Modelltilgang avhenger av kontoen din.

`LLM_REASONING_EFFORT` sendes bare når feltet er satt. Qwen-modellen ovenfor aksepterer `low`, `medium` og `xhigh`; `high` gir feil. La feltet være tomt for modeller som ikke støtter parameteren. Endepunktet må også støtte JSON-svar og de øvrige forespørselsfeltene programmet bruker.

## Plattformen (send videoer fra web eller telefon)

I tillegg til CLI-et finnes det en plattform: en web-UI der du sender videoer (lenker eller filer), en jobb-kø, en worker som kjører hele pipeline-en automatisk, et REST-API, og en Telegram-bot som fungerer som sendekanalt fra telefonen.

```sh
python src/cli.py platform
```

starter alt med én kommando: API-serveren, worker-en og dashboardet (Streamlit, mørkt tema, topp-tabs: **Send · Bibliotek · Søk · Syntese · Ranking · Eksport**). Du kan også kjøre delene hver for seg i egne terminaler:

| Kommando | Virkning |
|---|---|
| `platform` | Starter API + worker + dashboard i én pakke (Ctrl+C stenger alt). |
| `worker` | Bakgrunns-prosess som tar jobber ut av køen og kjører: oppdaging → transkribering → [OCR] → [visjon] → analyse → [embeddings]. |
| `api` | REST-API på `http://127.0.0.1:8000` (se under). |
| `bot` | Telegram-sendekanalen (long polling – krever ingen offentlig URL). |

**Sende videoer:**
- **Web-UI**: fanen «📥 Send» – lim inn lenker (YouTube/TikTok/Instagram) og/eller dra inn videofiler, velg hvilke pipeline-trinn som skal kjøre, trykk Send. Køen oppdateres live i vinduet.
- **Telegram**: lag en bot med `@BotFather`, sett `TELEGRAM_BOT_TOKEN` og `TELEGRAM_ALLOWED_IDS` (finn ID-en din hos `@userinfobot`) i `.env`, kjør `python src/cli.py bot`. Del en TikTok/IG-video til Telegram → velg boten, eller lim inn en lenke, eller send en fil (opp til ~19 MB). Kommandoer i chatten: `/status`, `/search <term>`, `/cost`. **Tom allow-list = boten svarer på ingen** (sikker standard).
- **API** (for fremtidige klienter, f.eks. en PWA): `POST /api/videos` med JSON `{"urls": [...], "options": {...}}` eller multipart (felt `urls` + filfelt `files`); `GET /api/jobs`, `GET /api/events` (live via Server-Sent Events), `GET /api/videos`, `GET /api/search?q=`, `GET /api/health`.

Lokale filer (opplasting/bot) transkriberes direkte med Whisper uten nedlasting. Jobber som feiler får feilmeldingen lagret i køen og på videoen; OCR/visjon/embeddings-feil er ikke fatale og logges som merknad. API-et er et lokalt, enkeltbruker-verktøy i denne fasen (ingen auth ennå).

**TikTok-slideshow (bildekaruseller):** lenker på formen `…/photo/<id>` (også `vm.tiktok.com`-korthenker) behandles automatisk: bildene lastes ned til `data/slides/<id>/`, hvert bilde OCR-leses og beskrives av visjonsmodellen (altid på for karuseller – det er hele innholdet), og det lages et syntetisk «transkript» av tekstene slik at analyse, søk og embeddings fungerer som for vanlige videoer. Instagram-bildeinnlegg støttes foreløpig ikke.

**Se dashboardet fra telefonen (valgfritt):** dashboardet kjører lokalt på `http://localhost:8501`. Er telefonen på samme nettverk (hjemme-Wi-Fi), kan du åpne PCens LAN-IP i stedet, f.eks. `http://10.52.224.197:8501` (IPen vises av Streamlit ved oppstart, eller `ipconfig` i PowerShell). Ingen tunnel eller Cloudflare trengs. Telegram-botten trenger heller ingen offentlig URL – den bruker long polling utgående, så lenker sendt mens PCen er av hentes opp når plattformen starter (opptil 24 timer).

**Desktop-app (Windows):** «VKE Platform»-snoppen på skrivebordet (bygd fra `launcher.pyw` med PyInstaller → `dist/VKE Platform.exe`) starter API + worker + bot + dashboard hvis de ikke allerede kjører, og åpner dashboardet i nettleseren. `start-platform.bat` gjør det samme via konsolvinduer; tjenesloggene ligger i `data/logs/`.

## Bruk (CLI)

Start med noen få videoer:

```sh
python src/cli.py discover "https://www.youtube.com/playlist?list=DIN_SPILLELISTE" --limit 5
python src/cli.py transcripts
python src/cli.py cost
python src/cli.py analyze --limit 2
python src/cli.py status
python src/cli.py dashboard
```

`discover` kan også lese en lokal UTF-8-tekstfil med én YouTube-lenke per linje. Tomme linjer og linjer som begynner med `#`, ignoreres. Hold personlige spillelister og lenkefiler utenfor Git.

| Kommando | Virkning |
|---|---|
| `discover <kilde> --limit N` | Henter metadata fra en YouTube-playlist eller en tekstfil med én URL per linje. Støtter også TikTok- og Instagram-URL-er (tekstfil; ID hentes via yt-dlp). Duplikate video-ID-er slås sammen. |
| `transcripts` | Henter undertekster og gjenbruker lagrede resultater. Prøver `yt-dlp` dersom primærmetoden ikke lykkes. |
| `transcripts --force` | Henter undertekster på nytt. |
| `transcribe [KILDE]` | Lokal reserve når YouTube-undertekst-API-et er IP-blokkert: laster ned lyden med `yt-dlp` og transkriberer den lokalt med Whisper (GPU/CUDA). Valg: `--model` (`tiny`/`base`/`small`/`medium`/`large-v3`), `--force`, `--no-keep-audio`. Skriver `.txt` og `.srt` til `data/transcripts/`. |
| `screen-text [--limit N]` | Fanger **visuell informasjon på skjermen**: laster ned videoen, tar ut ett bilde hvert N-te sekund (ffmpeg) og kjører OCR over teksten/koden. Lagrer resultater per video; `analyze` melder den inn som en ekstra kanal i prompten. Krever `ffmpeg` + OCR-motor (`OCR_ENGINE`: `easyocr` eller `paddleocr`). Valg: `--force`, `--frame-interval N`, `--keep-media`. |
| `describe-frames [--limit N]` | **Forstå** skjermen med en visjonsmodell: uttrekker noen jevnt fordelt frames (ffmpeg) og spør en multimodal modell om å beskrive hva som vises (kode, diagrammer, UI). Supplerer `screen-text` (OCR leser tekst; dette forstår layout). Kostnadsbegrenset: maks `VISION_MAX_FRAMES` frames per video. Krever `ffmpeg` + en visjonsmodell (`VISION_MODEL`; lokal Ollama = gratis, eller `VISION_PROVIDER=api`). Valg: `--force`, `--max-frames N`, `--keep-media`. |
| `embed [--limit N]` | Bygger **semantisk søk**: embedder transkriptchunks med en liten embeddingsmodell (lokal Ollama som standard) slik at du senere kan hente relevans etter *betydning* i stedet for nøkkelord. Bruker en separat modell — påvirker ikke analyse-kostnaden. Valg: `--force`, `--limit N`. |
| `search-semantic "SPØRSMÅL" --top N` | Semantisk søk over transkriptchunks: embedder spørringen og rangerer chunks etter cosineslikt. Brukes til konteksthenting (RAG) før du fôrer en modell. |
| `cost` | Viser et token- og prisestimat uten modellkall. |
| `analyze --limit N` | Sender inntil N ventende transkripter til modellen. Lagrede analyser hoppes over. |
| `analyze --limit N --force` | Analyserer også videoer som har en lagret analyse. Dette kan koste på nytt. |
| `synthesize [--force]` | Lag ÉN samlet syntese over alle analyserte videoer: felles tema, motsetninger og unik kunnskap. Én modellkall med kostnadsestimat; lastes fra cache uten `--force`. |
| `export` | Eksporter alle analyserte videoer som Markdown-notater (med YAML-metadata og klikkbare kildelenker til video + tidsstempel), plus `INDEKS.md` (syntese + rangert videoliste) og `KUNNSKAP.md` (kompakt, deduplisert kunnskapskonsentrat optimert for inntasting som kontekst til en språkmodell). Ingen modellkall; skriver til `data/exports/markdown/`. |
| `status` | Viser lagret behandlingsstatus. |
| `dashboard` | Starter plattform-UI-et (Streamlit): Send (lenker/filer + live-kø), Bibliotek, Søk, Syntese, Ranking og Eksport. Samme UI som `platform`, men uten API og worker. |

Se alle valg med `python src/cli.py --help`. `--limit` må være et positivt heltall.

`transcripts --force` oppdaterer undertekstene, men beholder allerede lagrede analyser. Kjør også `analyze --force --limit N` dersom analysene skal bygges på nytt fra de oppdaterte undertekstene.

`transcribe` er en fullstendig lokal alternativ: den laster ned bare lyden og transkriberer med Whisper, så den fungerer selv når YouTube-undertekst-API-et er IP-blokkert. Resultatene lagres i samme `transcripts`-tabell som `transcripts`, så `analyze` fungerer uendret. På Windows-maskiner uten CUDA-installasjon kan `ctranslate2` kreve cuBLAS/cuDNN-biblioteker; se kommentarene i `requirements.txt`.

## Kostnader og begrensninger

- Uten `--limit` kjøres opptil `DRY_RUN_LIMIT` videoer, normalt 5. Dette er **ekte analyse og ekte modellkall**, selv om eldre meldinger omtaler det som «dry-run».
- `SPENDING_LIMIT_USD` avviser en analysejobb når estimatet overstiger grensen. Dette er **ikke en garantert fakturagrense eller et samlet budsjett på tvers av kjøringer**. Sett også en grense hos leverandøren ved behov.
- Tokenberegningen er en grov tegnbasert tilnærming. Innebygde referansepriser for enkelte kjente modeller kan være utdaterte. Eksplisitte prisfelt overstyrer dem. Andre leverandørgebyrer inngår ikke.
- `LLM_MAX_OUTPUT_TOKENS` begrenser forespurt svarlengde og brukes i estimatet. For lav grense kan gi avbrutt eller ugyldig JSON.
- `RATE_LIMIT_SECONDS` styrer pausen mellom analyseforsøk. Også mislykkede API-kall kan bli fakturert.
- Lange transkripter deles i overlappende deler som analyseres hver for seg og slås sammen (map-reduce), slik at hele videoen dekkes uten avkorting. Antall deler – og dermed modellkall og kostnad – øker med videoens lengde; kortere videoer bruker fortsatt ett enkelt kall.
- Hvert innsikt ber modellen om et kort sitat fra transkriptet. Sitater som ikke finnes i transkriptet markeres ⚠️ og senkes til lav konfidens (sannsynlig hallusinasjon), og tidsstempler som ligger langt fra sitatets faktiske plass justeres automatisk. Dashboardet viser tidsstempler som klikkbare YouTube-lenker.
- Fulltekst-søket (Søk i dashboardet) bruker SQLite FTS5 over hele transkriptinnholdet med bm25-rangering og uthevet snutt. Hvis SQLite-bygget mangler FTS5, brukes automatisk delstrekk-søk som reserve. Nye transkripter indekseres ved lagring; eksisterende databaser bygges om ved første søk.
- **Semantisk søk (embeddings)**: fulltekst-søk matcher nøkkelord. `embed` bygger i tillegg en semantisk indeks ved å embedde transkriptchunks, slik at `search-semantic` kan finne deler etter *betydning* (bra for konteksthenting/RAG). Den bruker en separat, liten embeddingsmodell (lokal Ollama = gratis som standard, `EMBED_MODEL=nomic-embed-text`; `EMBED_PROVIDER=api` for et betalt endepunkt). Den legger ingen omkostninger på analysebanen og krever ingen tunge Python-avhengigheter. Kjør `embed` etter `transcribe`/`transcripts`, før du bruker `search-semantic`.
- Samlet syntese (Syntese) er ett enkelt modellkall over kompakte per-video-sammendrag og krever minst to analyserte videoer. Den cache-reses i databasen og gjengis uten nytt modellkall med mindre du bruker `--force` eller videosettet har endret seg.
- Markdown-eksporten (`export`, også tilgjengelig som knapp under Eksport i dashboardet) lager ingen modellkall; den formaterer kun lagrede analyser til Obsidian/OpenCode-vennlige notater med kildelenker. Hvert notat har YAML-metadata (frontmatter) med video_id, platform, url, skår og tags for filtrering og parsing. `KUNNSKAP.md` er et kompakt, deduplisert konsentrat beregnet på inntasting som kontekst til en språkmodell. Filnavnet er deterministisk (tittel + video-id), så omgjøring skriver over de samme filene.
- Modellen vurderer transkriptet uten uavhengig kildekontroll. Strukturelt gyldige svar kan fortsatt inneholde feil.
- Undertekstene må finnes på YouTube. IP-blokkering kan stoppe begge innhentingsmetodene; da kan `transcribe` brukes til å transkribere lyden lokalt med Whisper (GPU/CUDA, eller CPU som reserve).
- **YouTube, TikTok og Instagram**: oppdagelse og transkribering støtter alle tre. YouTube kan bruke undertekster (`transcripts`) eller Whisper (`transcribe`). TikTok/Instagram har ingen undertekst-API, så `transcripts` hoppes over automatisk for disse — bruk `transcribe` (Whisper) der. Kildelenker i notatene er plattform-tilpasset: YouTube har tidsstempel-dype-lenker, mens TikTok/Instagram lenker til den originale URL-en (uten hopp). Instagram kan kreve cookies (sett `YOUTUBE_COOKIES_FILE`).
- **Tekst på skjermen (OCR)**: pipeline-en leser som standard **kun lyden**, så tekst/kode/tall som *vises* på skjermen går ellers tapt. `screen-text` dekker dette ved å laste ned videoen (begrenset til 720p — mer enn nok for OCR), uttrekke frames med ffmpeg og kjøre OCR. Den er **valgfri** (opt-in) og legger ingen omkostninger på den vanlige lyd-banen: tunge avhengigheter (OCR-motoren) importeres bare når funksjonen brukes, og videoen slettes etterpå med mindre `--keep-media` brukes. Fanges teksten, meldes den inn i analysen som en egen kanal («TEKST PÅ SKJERMEN») — bare for videoer som faktisk har fanget skjermt tekst. Kjør `screen-text` **før** `analyze` dersom du vil at OCR-teksten skal tas med. easyocr bruker GPU automatisk dersom CUDA-versjonen av torch er installert (for Windows: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`), ellers CPU. Nye yt-dlp-versjoner krever et JavaScript-runtime for videodownloads fra YouTube: `winget install DenoLand.Deno`.
- **Forståelse av skjermen (visjonsmodell)**: OCR leser bare tekst. `describe-frames` går et steg videre og spør en multimodal modell om å *beskrive* hva som vises (kode-layout, diagrammer, UI), noe OCR ikke greier. Den er også **valgfri og kostnadsbegrenset** (maks `VISION_MAX_FRAMES` jevnt fordelt frames per video) for å unngå store visjonskostnader. Standard er **lokal Ollama** (gratis, `VISION_MODEL=llava`); `VISION_PROVIDER=api` bruker et OpenAI-kompatibelt visjons-endepunkt som mottar bildeinnhold (f.eks. NTNUs gratis IDUN-modeller, eller Ollama som API). Beskrivelsene meldes inn i analysen som «BESKRIVELSE AV SKJERMEN» — bare for videoer som faktisk har fanget dem. Kjør `describe-frames` **før** `analyze`.
- Søk i oversikten gjelder tittel og kanal. Temavisningen er foreløpig en plassholder; hjelpefunksjonene i `cross_video.py` er ikke koblet til en full temavisning.

## Miljøvariabler

| Variabel | Standard / betydning |
|---|---|
| `USE_OLLAMA` | `false` i koden; eksempelfilen velger `true`. |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | `llama3.2` |
| `LLM_API_URL` | Hele URL-en til Chat Completions-endepunktet. |
| `LLM_API_KEY` | Leverandørnøkkel. Hold den utenfor Git og feilmeldinger. |
| `LLM_MODEL` | `gpt-4o-mini` i koden; må passe leverandøren. |
| `LLM_REASONING_EFFORT` | Tomt som standard; modellavhengig resonneringsnivå. |
| `LLM_INPUT_COST_PER_MILLION` | Valgfri eksplisitt USD-pris per million inputtoken. Krever også outputprisen. |
| `LLM_OUTPUT_COST_PER_MILLION` | Valgfri eksplisitt USD-pris per million outputtoken. |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` – forespurt maksimal svarlengde. |
| `SPENDING_LIMIT_USD` | `5.0` – grense for estimatet per analysejobb. |
| `DRY_RUN_LIMIT` | `5` – antall analyser uten eksplisitt `--limit`. |
| `RATE_LIMIT_SECONDS` | `1.0` – pause mellom analyseforsøk. |
| `OCR_ENGINE` | `easyocr` – OCR-motor for `screen-text` (`easyocr` eller `paddleocr`). Må installeres separat (`pip install easyocr` eller `pip install paddleocr paddlepaddle`) + `ffmpeg` på PATH. |
| `VISUAL_FRAME_INTERVAL` | `10` – antall sekunder mellom uttrekkede frames i `screen-text`. |
| `VISUAL_MAX_FRAMES` | `200` – øvre grense for antall frames per video. |
| `VISION_PROVIDER` | `ollama` – leverandør for `describe-frames` (`ollama` = lokal/gratis, `api` = OpenAI-kompatibelt visjons-endepunkt). |
| `VISION_MODEL` | `llava` – navnet på visjonsmodellen (Ollama: `llava`, `qwen2.5vl`, `llama3.2-vision` ...). |
| `VISION_MAX_FRAMES` | `6` – øvre grense for antall frames som beskrives per video (begrenser visjonskostnad). |
| `VISION_API_URL` | Tom som standard – kun for `VISION_PROVIDER=api`; OpenAI-kompatibel Chat Completions-URL (annakun faller tilbake på `LLM_API_URL`). |
| `EMBED_PROVIDER` | `ollama` – leverandør for `embed`/`search-semantic` (`ollama` = lokal/gratis, `api` = OpenAI-kompatibelt `/embeddings`-endepunkt). |
| `EMBED_MODEL` | `nomic-embed-text` – embeddingsmodellen (Ollama: `nomic-embed-text`, `mxbai-embed-large`, `all-minilm` ...). |
| `EMBED_URL` | Tom som standard – Ollama-embeddings-URL; annars utledet fra `OLLAMA_URL`. |
| `EMBED_API_URL` | Tom som standard – kun for `EMBED_PROVIDER=api`; OpenAI-kompatibel `/embeddings`-URL (annakun `LLM_API_URL`). |
| `EMBED_API_KEY` | Tom som standard – nøkkel for `EMBED_PROVIDER=api` (annakun `LLM_API_KEY`). |
| `EMBED_CHUNK_CHARS` | `15000` – chunk-størrelse (i tegn) for embeddings. |
| `API_HOST` | `127.0.0.1` – bind-adresse for plattform-API-et. |
| `API_PORT` | `8000` – port for plattform-API-et. |
| `TELEGRAM_BOT_TOKEN` | Tomt = boten er deaktivert. Token fra @BotFather. |
| `TELEGRAM_ALLOWED_IDS` | Kommaseparerte numeriske Telegram-ID-er. Tom = boten nekter alle. |

## Lagring og personvern

`data/knowledge_base.db` inneholder metadata, transkripter, analyser og plattformens jobb-kø (`jobs`-tabellen). Råtranskripter ligger også under `data/transcripts/`, JSON-eksport fra grensesnittet under `data/exports/`, og opplastede videofiler (fra web-UI eller Telegram) under `data/uploads/`. Disse filene og `.env` ignoreres av Git. `.env.example` er en offentlig mal uten hemmeligheter.

I API-modus sendes transkriptutdrag til leverandøren. I Ollama-modus sendes de til `OLLAMA_URL`; kontroller adressen dersom analysen skal forbli på egen maskin. YouTube-innhenting krever fortsatt nettverk. Bruk innhold du har rett til å hente og behandle.

## Utvikling og tester

```sh
python -m pytest -q
python src/cli.py --help
```

Testene bruker midlertidige data og erstatter eksterne nettverks- og modellkall. De bekrefter ikke live YouTube-tilgang, modelltilgang eller leverandørpriser. GitHub Actions kjører testene ved push og pull requests.

```text
src/cli.py                          Kommandolinje
src/video_knowledge_extractor/
  config.py                         Miljøoppsett
  jobs.py                           Jobb-kø (SQLite) for plattformen
  worker.py                         Jobb-worker: kjører pipeline-en per jobb
  api.py                            Plattform-API (FastAPI): sending, status, SSE, søk
  telegram_bot.py                   Telegram-sendekanal (long polling, allow-list)
  dashboard.py                      Streamlit-plattformen (Send, Bibliotek, Søk, Syntese, Ranking, Eksport)
  discovery.py                      Metadata og lenkelister
  transcripts.py                    Undertekster og lokal cache
  whisper_transcripts.py            Lokal Whisper-transkribering av lyd (GPU/CUDA)
  visual.py                         On-screen text: frame-uttrekk (ffmpeg) + OCR (valgfri)
  vision.py                         Visjonsmodell-beskrivelse av frames (valgfri, kostnadsbegrenset)
  embeddings.py                     Semantisk søk: embeddings + cosineslikthet (RAG)
  analysis.py                       Prompt, modellkall, svarbehandling og innsiktsverifisering
  db.py                             SQLite-lagring og FTS5-fulltekst-søk
  orchestrator.py                    Arbeidsflyt og estimatgrense
  cross_video.py                    Hjelpefunksjoner for temaer, rangering og samlet syntese
  notes.py                          Markdown-eksport med kildelenker (Obsidian/OpenCode)
  models.py                         Datamodeller
tests/                              Tester uten live modellkall (inkl. kvalitetstester i test_quality.py)
```


