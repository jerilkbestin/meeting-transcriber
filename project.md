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
| Transcription engine | faster-whisper (local, CPU, int8) |
| Diarization engine | pyannote.audio (planned) |

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
- **BlackHole 2ch** — attempted installation but not showing up in Audio MIDI Setup. Likely blocked by macOS Sequoia security. This is the current blocker for capturing others' voices.
- **Root problem** — there is currently no way to capture "others' voices" (playing through ULT WEAR speaker) as an input stream without BlackHole or an equivalent virtual loopback audio driver.

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

**Current state:** ✅ Complete and working. Use this before every session to validate devices.

---

### `transcribe.py`
**Purpose:** Core transcription script using faster-whisper.

**What it does:**
- Presents interactive device selection menu at startup (all input devices listed)
- Accepts two separate input streams simultaneously (e.g. mic + loopback)
- Both streams feed into a shared audio queue
- Transcription loop buffers 5 seconds of audio then runs faster-whisper
- Outputs timestamped lines to terminal and saves to a `.txt` file

**Current state:** ⚠️ Partially working.
- Single input stream transcription works when a valid input device is selected
- Dual input stream (mic + speaker capture) is blocked by the BlackHole issue
- Speaker diarization not yet implemented
- Device 6 (Teams Audio) confirmed silent — cannot be used as input

---

## Roadmap

### Phase 1 — Simple Transcriber (largely done ✅)
- [x] Install faster-whisper, sounddevice, numpy in local venv
- [x] Build audio device diagnostic tool (`test_audio.py`)
- [x] Build basic transcription script with interactive device picker (`transcribe.py`)
- [x] Support two simultaneous input streams feeding one shared queue
- [x] Confirm faster-whisper runs locally on M4 CPU with int8 compute type
- [ ] **Blocker:** Fix audio capture path — resolve BlackHole or find alternative loopback driver
- [ ] Confirm end-to-end: audio flows → Whisper receives it → transcript appears in terminal

### Phase 2 — Dual Stream (Your Voice + Others)
- [ ] Confirm BlackHole 2ch appears in Audio MIDI Setup after clean install + restart
  - Try direct .pkg from https://existingfish.github.io/BlackHole/
  - Check System Settings → Privacy & Security → Allow after install
  - Restart Mac and recheck Audio MIDI Setup
- [ ] Create Multi-Output Device in Audio MIDI Setup (ULT WEAR + BlackHole 2ch)
- [ ] Set Teams speaker output to Multi-Output Device
- [ ] Verify BlackHole (as input) captures others' voices using `test_audio.py`
- [ ] Update `transcribe.py` to use two streams:
  - Stream 1: MacBook Pro Microphone (device 4) → your voice
  - Stream 2: BlackHole 2ch (new device) → others' voices
- [ ] Merge both into shared queue → single Whisper transcription output

### Phase 3 — Speaker Diarization
- [ ] Install `pyannote.audio` and `torch`
- [ ] Create HuggingFace account, accept pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- [ ] Generate HuggingFace read token at https://huggingface.co/settings/tokens
- [ ] Download `pyannote/speaker-diarization-3.1` model (one-time, ~1GB, then fully offline)
- [ ] Integrate diarization into `transcribe.py`:
  - Run Whisper on audio chunk → get segments with word-level timestamps
  - Run pyannote on same chunk → get speaker turn timestamps
  - Match segment midpoint to speaker turn → label as Speaker 1, Speaker 2, etc.
- [ ] Output format: `[HH:MM:SS] Speaker 1: <transcribed text>`
- [ ] Test on a real Teams meeting

---

## Known Issues / Blockers

| Issue | Status | Notes |
|---|---|---|
| BlackHole not appearing in Audio MIDI Setup | 🔴 Blocked | Try direct .pkg installer, then Privacy & Security → Allow, then restart |
| Teams Audio (device 6) produces no audio | 🟡 Known | Virtual device appears silent on this setup, not usable as input |
| Only one side of conversation captured currently | 🟡 Blocked | Depends on BlackHole fix above |

---

## Dependencies

```bash
# Phase 1 & 2
pip install faster-whisper sounddevice numpy

# Phase 3 (diarization)
pip install pyannote.audio torch
```

---

## Running the Project

```bash
cd ~/meeting-transcriber
source venv/bin/activate

# Step 1 — validate audio devices
python test_audio.py

# Step 2 — start transcription
python transcribe.py
```
