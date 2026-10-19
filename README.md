# YouTube Knowledge Extractor

Et Python-verktøy som henter videometadata og tilgjengelige undertekster fra YouTube, lager norske analyseutkast med en språkmodell og viser resultatene lokalt med Streamlit. Data lagres i SQLite og JSON på maskinen din.

## Installasjon

Du trenger Python 3.10 eller nyere, nettverk for YouTube-innhenting og enten lokal Ollama eller et OpenAI-kompatibelt Chat Completions-endepunkt.

```sh
git clone https://github.com/SanderSjiraff/youtube-knowledge-extractor.git
cd youtube-knowledge-extractor
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

## Bruk

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
| `discover <kilde> --limit N` | Henter metadata fra en YouTube-lenke eller tekstfil. Duplikate video-ID-er slås sammen. |
| `transcripts` | Henter undertekster og gjenbruker lagrede resultater. Prøver `yt-dlp` dersom primærmetoden ikke lykkes. |
| `transcripts --force` | Henter undertekster på nytt. |
| `transcribe [KILDE]` | Lokal reserve når YouTube-undertekst-API-et er IP-blokkert: laster ned lyden med `yt-dlp` og transkriberer den lokalt med Whisper (GPU/CUDA). Valg: `--model` (`tiny`/`base`/`small`/`medium`/`large-v3`), `--force`, `--no-keep-audio`. Skriver `.txt` og `.srt` til `data/transcripts/`. |
| `cost` | Viser et token- og prisestimat uten modellkall. |
| `analyze --limit N` | Sender inntil N ventende transkripter til modellen. Lagrede analyser hoppes over. |
| `analyze --limit N --force` | Analyserer også videoer som har en lagret analyse. Dette kan koste på nytt. |
| `synthesize [--force]` | Lag ÉN samlet syntese over alle analyserte videoer: felles tema, motsetninger og unik kunnskap. Én modellkall med kostnadsestimat; lastes fra cache uten `--force`. |
| `export` | Eksporter alle analyserte videoer som Markdown-notater (med YAML-metadata og klikkbare kildelenker til video + tidsstempel), plus `INDEKS.md` (syntese + rangert videoliste) og `KUNNSKAP.md` (kompakt, deduplisert kunnskapskonsentrat optimert for inntasting som kontekst til en språkmodell). Ingen modellkall; skriver til `data/exports/markdown/`. |
| `status` | Viser lagret behandlingsstatus. |
| `dashboard` | Starter lokal visning av videoer, analyser, fulltekst-søk (Søk), samlet syntese (Syntese), rangering og eksport (JSON + Markdown). |

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
- Samlet syntese (Syntese) er ett enkelt modellkall over kompakte per-video-sammendrag og krever minst to analyserte videoer. Den cache-reses i databasen og gjengis uten nytt modellkall med mindre du bruker `--force` eller videosettet har endret seg.
- Markdown-eksporten (`export`, også tilgjengelig som knapp under Eksport i dashboardet) lager ingen modellkall; den formaterer kun lagrede analyser til Obsidian/OpenCode-vennlige notater med kildelenker. Hvert notat har YAML-metadata (frontmatter) med video_id, platform, url, skår og tags for filtrering og parsing. `KUNNSKAP.md` er et kompakt, deduplisert konsentrat beregnet på inntasting som kontekst til en språkmodell. Filnavnet er deterministisk (tittel + video-id), så omgjøring skriver over de samme filene.
- Modellen vurderer transkriptet uten uavhengig kildekontroll. Strukturelt gyldige svar kan fortsatt inneholde feil.
- Undertekstene må finnes på YouTube. IP-blokkering kan stoppe begge innhentingsmetodene; da kan `transcribe` brukes til å transkribere lyden lokalt med Whisper (GPU/CUDA, eller CPU som reserve).
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

## Lagring og personvern

`data/knowledge_base.db` inneholder metadata, transkripter og analyser. Råtranskripter ligger også under `data/transcripts/`, og JSON-eksport fra grensesnittet ligger under `data/exports/`. Disse filene og `.env` ignoreres av Git. `.env.example` er en offentlig mal uten hemmeligheter.

I API-modus sendes transkriptutdrag til leverandøren. I Ollama-modus sendes de til `OLLAMA_URL`; kontroller adressen dersom analysen skal forbli på egen maskin. YouTube-innhenting krever fortsatt nettverk. Bruk innhold du har rett til å hente og behandle.

## Utvikling og tester

```sh
python -m pytest -q
python src/cli.py --help
```

Testene bruker midlertidige data og erstatter eksterne nettverks- og modellkall. De bekrefter ikke live YouTube-tilgang, modelltilgang eller leverandørpriser. GitHub Actions kjører testene ved push og pull requests.

```text
src/cli.py                          Kommandolinje
src/youtube_knowledge_extractor/
  config.py                         Miljøoppsett
  discovery.py                      Metadata og lenkelister
  transcripts.py                    Undertekster og lokal cache
  whisper_transcripts.py            Lokal Whisper-transkribering av lyd (GPU/CUDA)
  analysis.py                       Prompt, modellkall, svarbehandling og innsiktsverifisering
  db.py                             SQLite-lagring og FTS5-fulltekst-søk
  orchestrator.py                    Arbeidsflyt og estimatgrense
  dashboard.py                       Streamlit-visning, fulltekst-søk og eksport
  cross_video.py                    Hjelpefunksjoner for temaer, rangering og samlet syntese
  notes.py                          Markdown-eksport med kildelenker (Obsidian/OpenCode)
  models.py                         Datamodeller
tests/                              Tester uten live modellkall (inkl. kvalitetstester i test_quality.py)
```


