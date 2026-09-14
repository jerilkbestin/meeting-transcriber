# teams-notetaker

Records a Microsoft Teams call **from your own Windows machine**, transcribes it
locally with Whisper, and writes structured meeting notes with the Claude API.

Nothing is installed into Teams and no Microsoft 365 admin approval is needed —
the tool listens to your speakers and your microphone the same way any recording
app does.

```
your microphone  ─┐
                  ├─► 16 kHz mono ─► faster-whisper (local) ─► transcript ─► Claude ─► notes.md
speaker loopback ─┘
```

---

## Read this first

**Tell everyone on the call that you are recording.** This tool captures audio
silently — Teams shows no banner, because Teams is not doing the recording. That
is exactly why the obligation is yours.

- Canada's rule for recording your own conversations comes from **Criminal Code
  s. 184** (interception of private communications) and its s. 184(2)(a)
  exception where one party to the conversation consents. Multiple Canadian law
  firms summarise this as "one-party consent" ([Pyzer](https://www.torontodefencelawyers.com/crime-record-a-conversation/),
  [Samfiru Tumarkin](https://stlawyers.ca/blog-news/recording-conversations-at-work-ontario/)).
  I was not able to fetch the statute text itself from justice.gc.ca to quote it
  directly, so **treat that as a secondary-source summary, not a verified quote**,
  and **this is not legal advice.**
- Criminal law is not the only constraint. Your employer's acceptable-use policy,
  BC's PIPA / federal PIPEDA, and any customer contract can all restrict
  recording a meeting even where the Criminal Code would not. Check with Cavallo's
  policy owner before using this on client calls.

The tool prints a consent reminder every time it starts. It has no "silent mode".

---

## Install (Windows)

Python 3.9–3.13, 64-bit.

```powershell
cd teams-notetaker
python -m pip install -r requirements.txt
```

Or in a virtual environment, if you prefer to keep it isolated:

```powershell
cd teams-notetaker
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**This step is required** — `python -m teams_notetaker` will refuse to run and
list what's missing until you do it.

**No API key is required.** Transcription runs entirely on your machine. The
notes step is optional and has three modes — see *Who writes the notes* below.

> **First run downloads the Whisper weights** from huggingface.co (~500 MB for
> `small.en`). If Cavallo's network blocks that host, see *Offline models* below.

---

## Who writes the notes

Whatever you choose, **`transcript.txt` and `notes-prompt.md` are always
written**, so a run never leaves you empty-handed.

| Mode | What happens | Setup |
| --- | --- | --- |
| `--notes none` | Transcript + a paste-ready prompt file. Open `notes-prompt.md`, select all, paste into claude.ai. | Nothing |
| `--notes local` | A local model writes the notes. Nothing leaves your machine. | Ollama or LM Studio |
| `--notes claude` | The Anthropic API writes them. Best quality. | `ANTHROPIC_API_KEY` |
| `--notes auto` *(default)* | Claude if a key is set, else a local server if one is listening, else `none`. | — |

### Paste-into-Claude (no setup)

```powershell
python -m teams_notetaker record --title "Q3 Sync"
```

With no key and no local server this just works and ends with:

```
No notes model configured — transcript only.
  Paste-ready prompt: .\meetings\2026-08-27_1030_q3-sync\notes-prompt.md
  Open it, select all, paste into claude.ai to get the notes.
```

`notes-prompt.md` is the full instruction set plus your transcript in one block.
Paste it into claude.ai and you get the same notes the API path would produce.

### Local model

Any OpenAI-compatible server works — Ollama, LM Studio, llama.cpp's server, vLLM.

```powershell
ollama pull llama3.1:8b
ollama serve

python -m teams_notetaker record --notes local --local-model llama3.1:8b
```

LM Studio instead? `--local-url http://localhost:1234/v1 --local-model <name>`.

Two things worth knowing:

- **Context window.** Ollama defaults to a small context (historically 4096
  tokens), so an hour-long meeting will not fit in one pass. The tool handles
  this by summarising ~24k-character slices and merging the results, but raising
  the model's context (`num_ctx`) gives noticeably better notes. Verify your
  runtime's current default rather than trusting this note.
- **Quality.** An 8B model produces usable discussion notes but is meaningfully
  worse than Claude at extracting action items and owners. `qwen2.5:14b` or a
  32B model is a real step up if your machine can hold it.

### Anthropic API

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."      # persists; reopen the terminal after
python -m teams_notetaker record --notes claude
```

---

## Use it

### 1. Check your devices

```powershell
python -m teams_notetaker devices
```

You should see at least one row marked `loopback` — that is the copy of whatever
your speakers are playing, i.e. everyone else on the call.

### 2. Record

```powershell
python -m teams_notetaker record --title "Q3 Platform Sync"
```

Start it just before you join, press **Ctrl+C** when the meeting ends. Live
transcript scrolls in the terminal as it goes.

Output lands in `./meetings/2026-08-27_1030_q3-platform-sync/`:

| File | What it is |
| --- | --- |
| `transcript.md` | Readable transcript with timestamps and speakers |
| `transcript.txt` | Same, plain text — this is what a model gets fed |
| `notes-prompt.md` | Prompt + transcript in one block, ready to paste into claude.ai |
| `transcript.jsonl` | One JSON object per segment, written live — survives a crash |
| `audio_me.wav`, `audio_participants.wav` | The raw 16 kHz recordings |
| `notes.md` | The finished notes — only if a notes model was configured |

### Useful flags

```powershell
# Bias the recogniser toward your jargon — the single biggest accuracy win
python -m teams_notetaker record --vocab "Cavallo, Dynamics GP, SOC 2, Nagendra"

# Give the notes writer context it can't infer from audio alone
python -m teams_notetaker record --context "Weekly sync; Priya owns the migration"

# Bigger model if your machine can take it (medium.en is noticeably better)
python -m teams_notetaker record --model medium.en

# NVIDIA GPU
python -m teams_notetaker record --model large-v3-turbo --device cuda --compute-type float16

# Transcript + paste-ready prompt only, no model call at all
python -m teams_notetaker record --notes none

# Let a local model write the notes
python -m teams_notetaker record --notes local --local-model qwen2.5:14b

# Don't keep the audio files
python -m teams_notetaker record --no-save-audio

# Pick devices by hand (from the `devices` output)
python -m teams_notetaker record --mic-device 1 --system-device 14
```

### 3. Other entry points

```powershell
# An existing recording you already have
python -m teams_notetaker file "C:\Users\you\Downloads\Meeting-20260827.mp4"

# A transcript Teams exported itself (.vtt keeps real per-person speaker names)
python -m teams_notetaker notes "Meeting transcript.vtt" --title "Q3 Sync"
```

The `notes` path is the best-quality option when it is available, because Teams'
own `.vtt` has true speaker attribution per participant.

---

## How the two speaker tracks work

Real speaker diarization is unreliable. This tool sidesteps it: your microphone
is one recording, the speaker loopback is another, so "you" and "everyone else"
are separated perfectly with no ML guesswork.

**This only works if you wear headphones.** On open speakers your mic also picks
up the remote audio and both tracks end up containing everybody. Individual
remote participants are still not separated from each other — if you need
per-person names, use the `notes` path with a Teams `.vtt`.

---

## Tuning

| Symptom | Try |
| --- | --- |
| `No module named 'faster_whisper'` | `python -m pip install -r requirements.txt` |
| Words wrong, especially names/products | `--vocab "term, term, term"` |
| Still inaccurate | `--model medium.en` (slower) |
| Transcript lags far behind | `--model base.en`, or `--min-chunk 8 --max-chunk 20` |
| `dropped N audio blocks` warning | Machine can't keep up — use a smaller model |
| Sentences cut mid-word | Raise `--min-chunk` |
| Nothing transcribed | Run `devices`, pass `--system-device` explicitly |

Model sizes, smallest to largest: `tiny.en`, `base.en`, `small.en` (default),
`medium.en`, `large-v3`, `large-v3-turbo`.

### Offline models

If huggingface.co is blocked on your network, download a CTranslate2-converted
Whisper model on a machine that can reach it, copy the folder over, and pass the
path:

```powershell
python -m teams_notetaker record --model "C:\models\faster-whisper-small.en"
```

---

## Cost

Transcription is always free — it runs on your CPU. `--notes none` and
`--notes local` cost nothing at all.

Only `--notes claude` spends money. A one-hour meeting is roughly 8–12k words ≈
12–16k input tokens, plus ~1–2k output tokens. Current per-token prices are at
<https://platform.claude.com/docs/en/about-claude/pricing>.

---

## Layout

```
teams_notetaker/
  config.py      Settings dataclass; 16 kHz constant; track names
  audio.py       WASAPI loopback + mic capture, mono conversion, resampling, WAV sink
  engine.py      Silence-aligned chunking, Whisper calls, Utterance model, rendering
  summarize.py   Prompts, Claude + local backends, map-reduce, .vtt/.srt parsing
  cli.py         Subcommands: devices / record / file / notes
selftest.py      100+ offline checks, no hardware or model download required
```

Run the checks any time:

```powershell
python selftest.py
```

---

## Known limits

- **Windows only for live capture.** WASAPI loopback does not exist elsewhere.
  On macOS you would install BlackHole and pass its input device to
  `--system-device`; on Linux, a PulseAudio `.monitor` source.
- **No per-person diarization** among remote participants (see above).
- **Whisper hallucinates on silence.** A confidence filter drops the worst of it,
  but check anything load-bearing against the audio.
- **Notes are model output from imperfect ASR.** The prompt forbids inventing
  action items and makes the model flag garbled passages, but verify owners,
  dates and numbers before acting on them.
