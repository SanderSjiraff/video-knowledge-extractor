# Video Knowledge Extractor (VKE)

**Gjør lagrede videoer til et søkbart kunnskapsbibliotek.**

VKE er et lokalt Python-verktøy for deg som lagrer nyttige videoer og repos, men ønsker å finne igjen innholdet uten å se alt på nytt. Legg inn lenker fra YouTube, TikTok, Instagram og GitHub-repos, eller last opp en fil. VKE henter innholdet, transkriberer ved behov og bruker en språkmodell til å lage norske sammendrag, innsikter og konkrete steg du kan bruke videre.

Du får et nettleserbasert dashboard med bibliotek, behandlingskø, anbefalinger og samlet kunnskap på tvers av videoer. Metadata og analyser lagres lokalt i SQLite. Analyse kan kjøres med lokal Ollama eller et OpenAI-kompatibelt API.

## Hva kan du bruke det til?

- Finne igjen en forklaring, et verktøy eller en metode fra en video du har lagret.
- Samle kurs, foredrag og korte videoer i ett bibliotek med søk i tittel, kanal, sammendrag og transkript.
- Prioritere hva du bør se, skumme eller hoppe over basert på modellens vurdering av relevans og praktisk verdi.
- Se felles temaer, unik kunnskap og motsetninger mellom videoer.
- Eksportere norske notater til Markdown, for eksempel til Obsidian, eller hente ut data som JSON.
- Sende inn videoer fra telefonen via en valgfri Telegram-bot.

Analysene er modellgenererte utkast. VKE kontrollerer sitater mot transkriptet, men faktasjekker ikke påstander mot uavhengige kilder.

## Slik fungerer det

```text
Videolenke, GitHub-repo eller fil
        ↓
Metadata og undertekster / lokal Whisper-transkribering / repo-innhold (README + docs + kildekode)
        ↓
Valgfri skjermtekst (OCR) og bildebeskrivelse
        ↓
Norsk analyse: sammendrag, innsikter, verktøy og handlingssteg
        ↓
Bibliotek → Innsikt → eksport til JSON / Markdown
```

En worker behandler jobbene i bakgrunnen. YouTube kan bruke eksisterende undertekster; TikTok, Instagram og lokale filer bruker lokal transkribering når det trengs. GitHub-repos hentes som tekst via GitHub API-et (repo-oversikt, README, markdown-dokumentasjon, filstruktur og utvalgte kildefiler); innsikter peker da på filstier i stedet for tidsstempler. Semantisk indeksering er et separat steg og kan brukes fra CLI-et.

## Kom i gang

### 1. Forutsetninger

| Du trenger | Brukes til |
|---|---|
| Python 3.10 eller nyere og Git | Installere og kjøre prosjektet. |
| FFmpeg tilgjengelig på `PATH` | Lydbehandling, transkribering og uttrekk av videobilder. |
| Ollama **eller** et OpenAI-kompatibelt API | Lage analyser og synteser. |
| Nettverk | Hente videoer og laste ned modeller første gang. |

Streamlit 1.65 eller nyere kreves av dashboardet og installeres gjennom `requirements.txt`. GPU er valgfritt; lokal transkribering kan bruke CPU, men store modeller kan være tregere. Installasjonen inkluderer Whisper og EasyOCR og kan derfor laste ned store pakker.

Installer FFmpeg med pakkebehandleren din, for eksempel:

```powershell
# Windows
winget install Gyan.FFmpeg
```

```sh
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

Åpne en ny terminal etter installasjon ved behov. Kontroller med `ffmpeg -version`. Enkelte YouTube-nedlastinger krever også et JavaScript-runtime som Deno; se [feilsøking](#feilsøking).

### 2. Last ned og installer

```sh
git clone https://github.com/SanderSjiraff/video-knowledge-extractor.git
cd video-knowledge-extractor
python -m venv .venv
```

På macOS/Linux kan du bruke `python3` i stedet for `python` når du oppretter miljøet.

**Windows / PowerShell** – ingen aktivering av miljøet er nødvendig:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux:**

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Kjør kommandoene fra prosjektmappen. `.env` og `data/` leses relativt til arbeidsmappen. Miljøvariabler som allerede er satt, overstyrer `.env`.

### 3. Velg analysemodell

#### Alternativ A: Lokal Ollama

Installer og start [Ollama](https://ollama.com/), og hent modellene som er valgt i `.env.example`:

```sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

Behold dette i `.env`:

```dotenv
USE_OLLAMA=true
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2
EMBED_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
```

`llama3.2` lager analyser; `nomic-embed-text` brukes til semantisk indeksering. Modellnavn kan endres til modeller du har tilgjengelig. Lokal kjøring har ingen API-tokenkostnad, men bruker maskinressurser. Hvis du vil bruke bildebeskrivelser, må også `VISION_MODEL` være lastet ned og støtte bilder.

#### Alternativ B: OpenAI-kompatibelt API

Rediger `.env` med ditt endepunkt, modellnavn og din egen nøkkel:

```dotenv
USE_OLLAMA=false
LLM_API_URL=https://din-leverandor.example/v1/chat/completions
LLM_API_KEY=din-api-nokkel
LLM_MODEL=modellnavn-fra-leverandoren
LLM_INPUT_COST_PER_MILLION=pris-per-million-inputtoken
LLM_OUTPUT_COST_PER_MILLION=pris-per-million-outputtoken
LLM_REASONING_EFFORT=
```

Erstatt alle plassholderne, inkludert prisene, før oppstart. Prisene skal være tall i USD. Sett begge til `0` bare hvis endepunktet faktisk er gratis for deg. Ukjente modeller uten eksplisitte priser avvises før analyse. Endepunktet må støtte Chat Completions og JSON-svar; ikke alle tjenester som kaller seg OpenAI-kompatible støtter alle feltene.

La `LLM_REASONING_EFFORT` være tomt med mindre modellen støtter det. Embeddings og visjon har egne innstillinger: valg av ekstern analysemodell flytter ikke automatisk disse funksjonene til samme leverandør. Se [konfigurasjonsreferansen](docs/REFERENCE.md#miljøvariabler).

### 4. Start plattformen

**Windows / PowerShell:**

```powershell
.\.venv\Scripts\python.exe src/cli.py platform
```

**macOS / Linux** med aktivert miljø:

```sh
python src/cli.py platform
```

Åpne **[http://localhost:8501](http://localhost:8501)**. Kommandoen starter dashboard, API og worker. La terminalen stå åpen; `Ctrl+C` stopper tjenestene. Telegram-botten startes separat.

For å se eksisterende data uten å starte behandling kan du kjøre `python src/cli.py dashboard`. Da må en worker startes separat for at nye innsendte videoer skal bli behandlet.

### 5. Legg til din første video

1. Åpne **Oversikt**, og lim inn én støttet lenke per linje — YouTube, TikTok, Instagram eller en **GitHub-repo** — eller velg **Last opp filer**.
2. Trykk **Analyser videoer**. Standardløpet bruker transkribering, analyse og embeddings; OCR og visjon er av som standard for vanlige videoer.
3. Følg fremdriften i **Kø**. Worker-en må kjøre for at køen skal gå videre.
4. Åpne **Bibliotek** for søk, filtrering og analyse av en video.
5. Gå til **Innsikt** for høyest rangerte videoer, temakart og syntese. Syntese må genereres med knappen og krever minst to analyserte videoer.

Begynn med én kort video for å kontrollere modelltilgang og transkribering før du legger til mange.

## Dashboardet

| Side | Innhold |
|---|---|
| **Oversikt** | Legg til lenker/filer, se statistikk, aktiv kø og nylig innhold. |
| **Bibliotek** | Søk i titler, kanaler, sammendrag og transkript; filtrer, sorter og åpne analyser. |
| **Kø** | Følg jobber, prioriter eller avbryt ventende jobber og prøv avsluttede jobber på nytt. |
| **Innsikt** | Topp fem videoer, full rangering, temakart og samlet syntese. Klikk en videotittel for analyse; lukk vinduet med × eller Esc og fortsett på samme sted. |
| **Eksport** | Last ned JSON, eller skriv Markdown-notater til `data/exports/markdown/`. |

Miniatyrbilder vises der de finnes. Rangeringen er modellbasert og vekter praktisk verdi og relevans, justert for overlapp. Temakartet bygges lokalt fra lagrede analyser. Nye modellkall gjøres først når du starter analyse eller genererer syntese.

## Flere måter å bruke VKE på

### Telegram fra telefonen

Opprett en bot med `@BotFather`, og fyll inn `TELEGRAM_BOT_TOKEN` og ditt numeriske bruker-ID i `TELEGRAM_ALLOWED_IDS` i `.env`. En tom tillatelsesliste nekter alle tilgang.

```sh
python src/cli.py bot
```

La også worker-en kjøre, for eksempel via `platform`. Send en videolenke, GitHub-repo eller fil til boten. Kommandoer: `/status`, `/search <ord>` og `/cost`. Filopplasting er begrenset til omtrent 19 MB. Botten bruker long polling og trenger ingen offentlig serveradresse.

### Kommandolinje

```sh
python src/cli.py discover "https://www.youtube.com/playlist?list=DIN_SPILLELISTE" --limit 5
python src/cli.py sync-playlist "https://www.youtube.com/playlist?list=DIN_SPILLELISTE"
python src/cli.py transcripts
python src/cli.py cost
python src/cli.py analyze --limit 2
python src/cli.py export
```

`sync-playlist` er den enkleste måten å holde biblioteket opptatt med en YouTube-spilleliste: den legger **kun nye** videoer i behandlingskøen (alt som allerede er i biblioteket eller køen hoppes over), og workeren behandler deretter transkribering, analyse og embeddings automatisk. Kommandoen er idempotent — kjør den så ofte du vil. For full automasjon kan du planlegge den i Windows (Opplagte oppgaver), eller kjøre `sync-playlist.bat` som følger i prosjektmappen (den logger til `data/logs/playlist_sync.log`).

Hvis undertekster mangler, bruk `transcribe` før `analyze`. Se [CLI, avansert behandling og alle innstillinger](docs/REFERENCE.md), eller kjør `python src/cli.py --help`. På Windows uten aktivert miljø erstatter du `python` med `.\.venv\Scripts\python.exe`.

### REST-API

`platform` starter også API-et på `http://127.0.0.1:8000`. Du kan starte det separat med `python src/cli.py api`.

| Endepunkt | Bruk |
|---|---|
| `POST /api/videos` | Legg lenker eller filer i køen. |
| `GET /api/jobs` | Hent jobbstatus. |
| `GET /api/events` | Følg hendelser med Server-Sent Events. |
| `GET /api/videos` | List videoer. |
| `GET /api/search?q=...` | Søk i innholdet. |
| `GET /api/health` | Sjekk at API-et svarer. |

API-et er laget for lokal bruk av én person og har foreløpig ingen innlogging.

## Kostnader, lagring og begrensninger

- **API-kall kan koste penger.** `cost` viser et estimat. `SPENDING_LIMIT_USD` er en kontroll av estimert kostnad per kjøring, ikke et garantert samlet budsjett. Lange videoer kan kreve flere kall. `--force` kan medføre ny fakturering.
- **Data lagres i `data/`.** Databasen er `data/knowledge_base.db`; transkripter, opplastinger og eksport har egne undermapper. Ta sikkerhetskopi av `data/` hvis du vil bevare biblioteket. `.env` og personlige data er ignorert av Git.
- **Lokal lagring betyr ikke at alle modellkall er lokale.** I API-modus sendes innhold til valgt leverandør. OCR, visjon og embeddings kan ha egne endepunkter. Del aldri `.env`, cookies eller API-nøkler.
- **Tilgang til videoer kan feile.** Private/slettede videoer, innlogging, plattformendringer og IP-blokkering kan hindre innhenting. Bruk innhold du har rett til å hente og behandle.
- **Bildekaruseller har et eget løp.** TikTok-slideshows bruker OCR og visjon selv om disse er av for vanlige videoer. Instagram-bildeinnlegg støttes foreløpig ikke.
- **Windows-launcherne er personlige oppsett.** `launcher.pyw`, `start-platform.bat` og `sync-playlist.bat` inneholder maskinspesifikke stier (og spillelistelenken). Andre bør bruke CLI-kommandoene over, eller tilpasse stiene før bruk. Ingen ferdig `.exe` er nødvendig for å komme i gang.

## Feilsøking

| Problem | Sjekk dette |
|---|---|
| Videoer blir stående i kø | La `platform` eller `worker` kjøre. Se status og feilmelding under **Kø**. |
| Ollama svarer ikke / modell mangler | Start Ollama, kontroller URL-en i `.env`, og kjør `ollama list`. Hent modellen med `ollama pull <modellnavn>`. |
| Ingen undertekster / blokkert YouTube-tilgang | Prøv lokal `transcribe` via CLI. Det krever fortsatt tilgang til selve videonedlastingen. |
| FFmpeg blir ikke funnet | Kontroller `ffmpeg -version` i samme terminal som starter VKE. |
| YouTube ber om JavaScript-runtime | Installer Deno og start terminalen på nytt. På Windows: `winget install DenoLand.Deno`. |
| Instagram krever innlogging | Sett `YOUTUBE_COOKIES_FILE` til en lokal cookies-fil i Netscape-format. Hold filen utenfor Git. |
| CUDA-/cuBLAS-/cuDNN-feil | Kontroller GPU-driver og bibliotekene Whisper bruker. Se kommentarene i `requirements.txt`; CPU kan brukes som reserve. |
| Oppgraderingen gir feil i UI-et | Kjør `python -m pip install -r requirements.txt` i riktig virtuelt miljø. |

## Utvikling og tester

```sh
python -m pytest -q
python src/cli.py --help
python src/cli.py status
```

Testene bruker midlertidige data og erstatter eksterne nettverks- og modellkall. De tester ikke faktisk tilgang til videoer eller modellleverandører. GitHub Actions kjører testsuiten på Ubuntu med Python 3.10 og 3.12, og Windows med Python 3.12.

Koden ligger i `src/video_knowledge_extractor/`: `dashboard.py` er grensesnittet, `worker.py` behandler køen, `api.py` er API-et, og `db.py` håndterer lagring. `src/cli.py` er inngangen til kommandolinjen. Se [referansen](docs/REFERENCE.md#prosjektstruktur) for resten av modulene.
