# Meeting Transcriber — Project Brief

## Goal
Build a fully local, privacy-preserving meeting transcription tool for macOS (MacBook Pro M4) that captures audio from Microsoft Teams meetings and produces timestamped transcripts. No audio or transcript data leaves the machine.

---

## Environment

| Item | Detail |
|---|---|
| Machine | MacBook Pro M4 |
| OS | macOS (Sequoia) |
| Shell | zsh (Mac Terminal) |
| Meeting App | Microsoft Teams |
| Python | 3.11 via Homebrew |
| Virtual env | `~/meeting-transcriber/venv` |
| Transcription engine | faster-whisper (local, CPU; `large-v3-turbo`/float32 by default — see `--accuracy`) |
| Loopback driver | BlackHole 2ch + a Multi-Output Device, wired by `bin/switch_meeting_output` |
| Diarization engine | pyannote.audio 4 + `speaker-diarization-community-1`, post-meeting only (`diarize.py`) |

### Audio Devices (from `sd.query_devices()`)
```
0  DELL U2722D, Core Audio (0 in, 2 out)
1  DELL U2722DE, Core Audio (0 in, 2 out)
2  ULT WEAR, Core Audio (1 in, 0 out)       ← Sony headset MIC (captures user's voice)
3  ULT WEAR, Core Audio (0 in, 2 out)       ← Sony headset SPEAKER (others' voices)
4  MacBook Pro Microphone, Core Audio (1 in, 0 out)
5  MacBook Pro Speakers, Core Audio (0 in, 2 out)
6  Microsoft Teams Audio, Core Audio (1 in, 1 out)   ← virtual device, confirmed silent
7  NoMachine Audio Adapter, Core Audio (2 in, 2 out)
8  NoMachine Microphone Adapter, Core Audio (2 in, 2 out)
9  Multi-Output Device, Core Audio (0 in, 2 out)
```

### Key Audio Findings
- **Device 6 (Microsoft Teams Audio)** — exists as a virtual device when Teams is running but produces no audible output on this setup. Capturing from it yields silence.
- **Device 3 (ULT WEAR speaker)** — output only, cannot be used as an input stream directly.
- **BlackHole 2ch** — ✅ resolved. Installed and visible in Audio MIDI Setup; it is the supported remote-audio capture path. `transcribe.py` auto-detects it by name and refuses to let it be chosen as the mic.
- **Capturing "others' voices"** — solved by routing Teams output to a Multi-Output Device (physical output + BlackHole 2ch), so the meeting audio is heard *and* available as an input. `bin/switch_meeting_output` (built from `audio_route.swift`) does the one-time wiring.

---

## Project Files

### `test_audio.py`
**Purpose:** Audio device diagnostic and validation tool.

**What it does:**
- Lists all available input and output devices with channel info
- Lets user pick any input and output device interactively
- Three test modes:
  1. **Input only** — shows real-time audio level bars to confirm audio is flowing
  2. **Output only** — plays a 440Hz test tone to confirm speaker works
  3. **Loopback** — routes input → output in real-time to confirm capture path

**Current state:** ✅ Complete and working. Use this before a session to validate devices.

---

### `transcribe.py`
**Purpose:** Core capture + transcription script using faster-whisper.

**What it does:**
- Auto-detects BlackHole; prompts only for the mic and the transcript name
- Captures two input streams at their native rates, tagged `You` and `Others`, into one queue
- Buffers per source with adaptive, silence-aware chunking (cuts on a natural pause, hard ceiling at `--max-chunk-seconds`)
- Timestamps each line from **capture time**, not write time, so on-screen lag never skews the transcript
- Writes `[HH:MM:SS] You|Others: text` to `transcripts/`
- Optional `--record`: also writes one 16 kHz mono WAV per stream plus a `_streams.json` sidecar, for `diarize.py`
- Optional `--input-file` (transcribe an existing recording) and `--remote-url` (send chunks to a GPU server)

**Current state:** ✅ Working. See `README.md` for the full flag list and design rationale.

---

### `diarize.py`
**Purpose:** Post-meeting speaker diarization of a `--record` session.

**What it does:**
- Re-transcribes both recorded WAVs (defaults to `--accuracy max`, which live capture cannot afford)
- Runs `pyannote/speaker-diarization-community-1` over the **Others** WAV only
- Assigns each segment to the speaker with the most overlap, and renumbers them `Others 1`, `Others 2`, … by first appearance
- Writes `<name>_diarized.txt` and `<name>_diarized.json` next to the recording

**Current state:** ✅ Implemented. Requires `requirements-diarize.txt` plus accepting the model terms on Hugging Face.

---

## Roadmap

### Phase 1 — Simple Transcriber ✅
- [x] Install faster-whisper, sounddevice, numpy in local venv
- [x] Build audio device diagnostic tool (`test_audio.py`)
- [x] Build basic transcription script with interactive device picker (`transcribe.py`)
- [x] Support two simultaneous input streams feeding one shared queue
- [x] Confirm faster-whisper runs locally on M4 CPU
- [x] Fix the audio capture path (BlackHole installed and detected)
- [x] Confirm end-to-end: audio flows → Whisper receives it → transcript appears in terminal

### Phase 2 — Dual Stream (Your Voice + Others) ✅
- [x] BlackHole 2ch appears in Audio MIDI Setup
- [x] Multi-Output Device (physical output + BlackHole 2ch)
- [x] Teams speaker output routed to it — automated by `bin/switch_meeting_output`
- [x] Verify BlackHole (as input) captures others' voices using `test_audio.py`
- [x] Two streams in `transcribe.py`: mic → `You`, BlackHole → `Others`
- [x] Merge both into a shared queue → single transcript

### Phase 3 — Speaker Diarization
- [x] Decide where it runs: **post-meeting pass**, not live. pyannote clusters speakers within the audio it is given, so one pass over the whole recording keeps labels consistent for the entire meeting; per-chunk live diarization cannot.
- [x] Record the streams: `transcribe.py --record` → one WAV per source + `_streams.json`
- [x] Build `diarize.py` (re-transcribe → diarize Others → assign → merge → write)
- [x] Keep `You` as `You`. The mic is a separate physical stream, so the local speaker is known, not inferred; only `Others` is subdivided.
- [x] Install `requirements-diarize.txt` (pyannote.audio 4.0.7 + torch 2.14.0)
- [x] Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1 (`gated=auto`, access confirmed)
- [x] Authenticate — `hf auth login` stores the token in `~/.cache/huggingface/token`; no `HF_TOKEN` export needed
- [x] Download the model (one-time, 32 MB measured, then fully offline)
- [x] Verify end-to-end on a recorded session, including determinism across two runs
- [ ] Test on a real Teams meeting

Output format: `[HH:MM:SS] Others 1: <transcribed text>` — the same shape as the live transcript, so the two files are directly comparable.

**Superseded:** an earlier draft of this phase targeted `speaker-diarization-3.1` and relabelled *every* line as a global `Speaker N`. Both were dropped: community-1 (pyannote.audio 4) is better on noisy real-world audio and adds `exclusive_speaker_diarization` for aligning turns to transcript timestamps, and relabelling `You` would discard the one speaker identity the two-stream capture already knows for certain.

---

## Known Issues / Blockers

| Issue | Status | Notes |
|---|---|---|
| BlackHole not appearing in Audio MIDI Setup | ✅ Resolved | Installed and detected; the supported remote-audio path |
| Teams Audio (device 6) produces no audio | 🟡 Known | Virtual device appears silent on this setup, not usable as input |
| Only one side of conversation captured | ✅ Resolved | Dual-stream capture shipped in Phase 2 |
| Diarization model not yet downloaded | ✅ Resolved | Terms accepted, `hf auth login` done, 32 MB cached; runs offline |
| L1 GPU inference server (`--remote-url`) | 🟡 Open | Client side done; the Windows FastAPI server is not built — see `project-local-NW-model-setup.md` |

---

## Dependencies

```bash
# Capture + transcription (all that transcribe.py needs)
pip install -r requirements.txt

# Post-meeting diarization only — pulls torch, ~0.6-1 GB installed
pip install -r requirements-diarize.txt

# Windows box (separate implementation)
pip install -r requirements-windows.txt
```

---

## Running the Project

```bash
cd ~/meeting-transcriber
source venv/bin/activate

# Step 1 — validate audio devices (optional)
python test_audio.py

# Step 2 — start transcription
python transcribe.py

# Step 2b — or record the streams too, for diarization afterwards
python transcribe.py --record

# Step 3 — after the meeting, split "Others" into individual speakers
python diarize.py transcripts/<name>_<timestamp>_streams.json
```

For the full meeting workflow — routing, what to watch during the call, how
long the post pass takes, naming the speakers, troubleshooting — see
[`phase3-meeting-guide.md`](phase3-meeting-guide.md).
