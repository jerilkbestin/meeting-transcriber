# meeting-transcriber

A fully local, privacy-preserving meeting transcriber. It captures
**your microphone** and the **other participants' audio**, transcribes both
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and writes
timestamped lines to a `.txt` transcript. No audio or transcript leaves the
machine by default.

## Platforms

- **macOS** (primary, documented below): `transcribe.py`, `test_audio.py`, and
  `audio_route.swift` capture audio via a [BlackHole](https://github.com/ExistentialAudio/BlackHole)
  loopback driver + Core Audio Multi-Output Device. `diarize.py` adds an
  optional post-meeting pass that splits `Others` into individual speakers.
- **Windows 11 + NVIDIA**: `transcribe_windows.py` and `test_audio_windows.py`
  — see `WINDOWS_NVIDIA.md` for setup.

The rest of this README covers the macOS path.

By default no audio and no transcript ever leaves the machine — model
inference is local, there is no network call in the transcription path.
Inference can optionally be sent to a GPU server on another machine instead
(`--remote-url`, off by default) — see Remote (GPU) inference below.

- **Target:** MacBook Pro (Apple Silicon), macOS Sequoia, Microsoft Teams.
- **Engine:** faster-whisper on CPU (`large-v3-turbo`, float32 by default),
  configured by `--accuracy fast|balanced|max` — or a remote faster-whisper
  server if `--remote-url` is set. See Accuracy presets below.
- **Output:** `[HH:MM:SS] You: ...` / `[HH:MM:SS] Others: ...` lines.

---

## Overview

Two problems have to be solved to transcribe a meeting locally:

1. **Capture** — macOS apps cannot normally record another app's output. You
   can capture your own mic, but the meeting audio (everyone else) plays out to
   a speaker and is not available as an input. The project solves this with a
   **loopback driver** (`BlackHole 2ch`) plus a **Multi-Output Device** so the
   meeting audio is simultaneously heard *and* fed back in as a capture source.
2. **Transcription** — turn two live audio streams into timestamped text,
   fast enough to keep up on a CPU, without leaking data to a cloud service.

`transcribe.py` handles capture + transcription. A small standalone Swift
helper, `switch_meeting_output`, handles the one-time audio routing (see
Architecture).

---

## Architecture

### Transcription data flow

```text
 ┌──────────────┐        ┌──────────────────┐
 │ Mic (You)    │ ─────► │ InputStream #1   │──┐   PortAudio real-time
 └──────────────┘        │ callback("You")  │  │   callback threads
                         └──────────────────┘  │   (copy only — cheap)
 ┌──────────────┐        ┌──────────────────┐  │
 │ BlackHole    │ ─────► │ InputStream #2   │──┤
 │ (Others)     │        │ callback("Others")│  │
 └──────────────┘        └──────────────────┘  │
                                                ▼
                                     ┌────────────────────┐
                                     │  queue.Queue        │  (thread-safe
                                     │  (source, raw block)│   producer/consumer
                                     └────────────────────┘   handoff)
                                                │
                                                ▼
                          ┌──────────────────────────────────────┐
                          │  transcribe_loop  (worker thread)     │
                          │  • downmix to mono + resample to 16k  │
                          │  • per-source SourceBuffer, adaptive   │
                          │    silence-aware chunking              │
                          │  • prepare_audio: silence gate + AGC   │
                          │  • transcribe_chunk: local Whisper OR  │
                          │    remote HTTP call (--remote-url)     │
                          │  • hallucination filter + carry-prompt │
                          └──────────────────────────────────────┘
                                                │
                                                ▼
                              [HH:MM:SS] You/Others: text
                                                │
                              ┌─────────────────┴──────────────────┐
                              ▼                                     ▼
                       terminal (stdout)              transcripts/<name>_<timestamp>.txt
```

Two audio streams run on their own PortAudio real-time threads. Each callback
does only the unavoidable minimum — copy the block, push `(source, raw block)`
onto a shared `queue.Queue` — and returns immediately; no allocation-heavy
downmix/resample happens on that thread. A single worker thread drains the
queue, downmixes and resamples each block, buffers audio per source in a
`SourceBuffer`, and transcribes once that source has accumulated a full
adaptive chunk (see Design & rationale). The main thread just stays alive and
waits for `Ctrl+C`, watching for the worker thread dying unexpectedly.

### Audio routing (set up once, outside transcribe.py)

```text
Teams Speaker ─► Multi-Output Device ─► selected physical output (you hear it)
                                     └─► BlackHole 2ch (captured as "Others")
```

Routing is deliberately **not** managed by `transcribe.py`. Creating/editing a
macOS Multi-Output (aggregate) device is a Core Audio device-graph operation
that the Python `sounddevice`/PortAudio bindings can't do. It lives in a
separate Swift binary, `switch_meeting_output` (source: `audio_route.swift`,
compiled into `bin/`), which you run before a meeting to pick the physical
output to hear through. `transcribe.py` only *reads* the device list and
auto-detects `BlackHole 2ch` as its "Others" input.

---

## Design & rationale

- **Batch-chunked, not streaming, ASR — with adaptive, silence-aware chunk
  boundaries.** Audio is buffered per source until `ready_source()` says a
  chunk is ready: either the buffer has hit `--max-chunk-seconds` (hard
  ceiling), or it has at least `--min-chunk-seconds`
  **and** a quiet tail (`SILENCE_WINDOW_SECONDS`/`CHUNK_SILENCE_RMS`) — i.e. a
  natural pause. This cuts chunks at sentence/pause boundaries instead of
  mid-word on a fixed clock, without needing word-by-word streaming ASR (no
  partial-hypothesis merging).
- **Hallucination filter, two independent checks.** Every decoded segment
  (local or remote) is checked against `is_hallucination()`: `no_speech_prob
  > 0.85 and avg_logprob < -0.9` catches Whisper inventing text on
  near-silence; `compression_ratio > 2.4` separately catches a *confident*
  repetition loop ("da da da da..."), which can have a fine `avg_logprob`
  and low `no_speech_prob` since real audio triggered it — the first check
  alone won't catch that failure mode. Both are in addition to
  `vad_filter=True` and the RMS silence gate.
- **Per-source carry-forward prompt.** After a chunk is transcribed, the tail
  of its kept text (`CARRY_PROMPT_MAX_CHARS`, 220 chars) is remembered per
  source and prepended as `initial_prompt` context for that source's *next*
  chunk — helps names/acronyms that span a chunk boundary, independent of
  `condition_on_previous_text` (see below).
- **One shared decode-options builder.** `build_decode_options()` is the
  single source of truth for every faster-whisper decode parameter. Live mode,
  file mode and the remote multipart fields all derive from it. Before this,
  live and file mode each carried their own copy of the parameter list, so
  tuning one silently skipped the other; `selftest.py` now asserts both paths
  produce the same dict.
- **The 30s padded window sets the cost, not the chunk size.**
  `faster_whisper/audio.py` `pad_or_trim()` pads every window to 3000 frames
  (30s) before the encoder runs, so a 6s chunk costs roughly what a 26s chunk
  costs. Decode cost per minute of meeting is therefore driven by the *number
  of calls*, i.e. inversely proportional to chunk length. Measured on the
  bundled 50s sample: 6 calls = 13.7s of decode, 4 calls = 9.4s, same audio.
  Raising the chunk length is what funds a bigger model. `--max-chunk-seconds`
  is capped at 30s for the same reason — beyond it, the overflow costs a whole
  second padded pass. `chunk_length` is deliberately never passed: it only
  shrinks the seek stride (more padded passes, no saving) and it mutates the
  shared `FeatureExtractor` in place for every later call.
- **Capture-time timestamps.** The wall clock is read in the audio callback
  and carried on the queue, and each line is stamped
  `chunk_capture_time + segment.start`. Stamping at write time would bake in
  the queue backlog, so every timestamp would be wrong by however far decoding
  is behind — which is exactly what a bigger model makes worse. Segment offsets
  are already mapped back to the un-stripped chunk timeline by
  `restore_speech_timestamps`, so VAD removal does not skew them.
- **Explicitly tuned `vad_parameters`.** `threshold` 0.45, `speech_pad_ms` 600,
  `min_silence_duration_ms` 1000. Because of the 30s pad, stripping more
  silence saves no compute in live mode — aggressive VAD has zero upside and
  one real downside, deleting speech, which leaves no trace in the output. The
  longer pad specifically protects onsets at chunk boundaries, where the
  adaptive chunker cuts.
- **`condition_on_previous_text=False`.** Each chunk still decodes
  independently of Whisper's own internal state. Note this is a no-op for live
  chunks — a chunk of ≤30s is a single window, so there is no "previous" — so
  it is really a file-mode choice, and file mode is the unattended
  long-running case where poisoning hurts most. A sentence split across a
  chunk boundary may fragment, but the transcriber is immune to Whisper's
  known failure mode where one bad window poisons every window after it — the
  right trade for an unattended long-running session. The carry-forward
  prompt above is a deliberately safer way to add cross-chunk context without
  re-enabling this.
- **Beam size, `temperature` fallback list, `best_of=5` (local mode).**
  The first decode attempt is deterministic beam search at temperature 0
  (`--beam-size`; 1 is greedy, 5 is faster-whisper's own default). Beam width
  is the largest decoder-side accuracy lever for this workload: greedy commits
  to the highest-probability token at each step, which is exactly how a rare
  proper noun ("ABAC", "Seaspan") loses to a common word that scores better on
  the first token.
  If that decode is rejected by faster-whisper's own
  `compression_ratio_threshold`/`log_prob_threshold` (too repetitive or
  low-confidence), it retries at the next temperature in
  `TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`, sampling
  `DEFAULT_BEST_OF = 5` candidates each time — this is what actually escapes
  a repetition loop; a single fixed temperature (the old default) has no
  retry to fall back to and just accepts a bad decode. Costs latency only on
  chunks that need a retry. All ignored in remote mode — the server controls
  its own decode parameters.
- **Separate `You` / `Others` buffers, `SourceBuffer.setdefault`.** Each side
  chunks and gates silence independently, so a quiet side never blocks the
  other's cadence. Buffers are created lazily via `dict.setdefault` keyed by
  source label rather than a fixed `{"You": ..., "Others": ...}` literal, so
  an unrecognized label can never raise `KeyError`. Cost: no cross-speaker
  turn-taking signal; the two logs are merged only by wall-clock print order.
- **Native device sample rate + software resample.** Each `sd.InputStream` is
  opened at its device's own `default_samplerate` (never assumed to be
  16 kHz), and `resample_to_target()` converts down to 16 kHz via `soxr`
  (falling back to a manual `np.interp` linear resample if `soxr` isn't
  installed).
- **Local vs. remote inference, one dispatch point.** `transcribe_chunk()`
  branches on `--remote-url`; everything downstream (hallucination filter,
  carry-prompt, transcript writing) is unaware of which mode ran. See
  "Remote (GPU) inference" below.
- **Producer/consumer via `queue.Queue`.** Real-time audio callbacks must never
  block; the queue decouples them from the slow, variable-latency inference
  step (local or remote). `queue.Queue` is internally lock-protected, so
  cross-thread `put`/`get` needs no extra locking. The callback itself does
  only the unavoidable buffer `copy()` — downmixing to mono and resampling
  happen in `transcribe_loop`, off the real-time thread.
- **Routing split into a Swift binary.** Keeps a heavy `PyObjC`/Core Audio
  dependency out of the Python tool for a one-time device-wiring step.
- **`daemon=True` worker + `Ctrl+C`, with crash detection.** No
  graceful-shutdown machinery; the daemon thread dies with the process on
  `Ctrl+C`. `transcript.flush()` after every line means an abrupt exit still
  keeps all written text. Separately, if the worker thread crashes for any
  other reason, `main()` notices via a shared `threading.Event` (or the
  thread simply no longer being alive) and exits loudly with a full
  traceback and the partial-transcript path, instead of continuing to run
  with no more output.
- **Diarization is a post-meeting pass, not a live one.** pyannote clusters
  speakers *within* the audio it is handed, so diarizing each 12-28s chunk
  live would produce labels that are only locally consistent: "Speaker 1" in
  one chunk need not be the same human as "Speaker 1" in the next. Keeping
  identities stable across a whole meeting would mean maintaining an embedding
  bank and doing incremental clustering — real complexity, and a drift risk —
  on a CPU that is already decoding. One pass over the finished recording
  clusters globally, costs nothing live, and is simply more accurate.
- **The recorder taps before `prepare_audio`, not after.** The transcription
  path deliberately destroys two things the diarizer needs: near-silent chunks
  are dropped outright, and AGC rescales what survives. Silence is precisely
  what separates one speaker turn from the next, so the WAV is written from
  the resampled-but-otherwise-untouched audio.
- **The post pass re-transcribes instead of reusing the live text.** Not
  because the live text is unanchored — since timestamps come from capture
  time it sits on a real timeline — but because the `.txt` keeps whole seconds
  only, and overlap-based speaker assignment needs float segment boundaries.
  Re-decoding also buys the `max` preset (beam 5, word timestamps), which live
  capture cannot afford and a post pass gets for free.
- **Recording is wall-clock anchored, with gaps filled.** Each block carries
  the capture timestamp the transcriber already reads in the audio callback.
  If PortAudio drops blocks, plain concatenation would silently compress that
  stream's timeline and slide every later timestamp earlier, so a shortfall
  beyond 50 ms is filled with silence instead. A backwards clock step (NTP) is
  counted in the sidecar but never repaired by discarding audio. Two devices
  on independent crystals still drift; the residual error is bounded by the
  tolerance plus block jitter, which is well under the one-second display
  granularity. This is not sample-accurate sync and does not claim to be.

---

## Code walkthrough (`transcribe.py`)

- **`sanitize_filename_component(value)`** — strips whitespace and replaces any
  char outside `[A-Za-z0-9._-]` with `_`, so user-entered transcript names
  can't inject bad filesystem characters.
- **`output_filename()`** — prompts for a name, falls back to `transcript`, and
  appends a `YYYYMMDD_HHMMSS` timestamp → `transcripts/<name>_<timestamp>.txt`
  (the `transcripts/` folder is created if it doesn't exist).
- **`list_devices` / `input_device_indices`** — enumerate `sd.query_devices()`;
  the second filters to input-capable devices (`max_input_channels > 0`).
- **`find_blackhole_input(devices)`** — auto-detects the BlackHole input by
  name (case-insensitive substring). Zero matches → `RuntimeError` (clear setup
  error); multiple → interactive disambiguation.
- **`choose_mic_input(devices, blackhole_index)`** — lists input devices, marks
  BlackHole, and refuses to let you pick BlackHole as the mic.
- **`stream_channels(devices, index)`** — caps requested channels at 2.
- **`device_sample_rate(devices, index)`** — reads a device's own
  `default_samplerate` instead of assuming 16 kHz is supported; falls back to
  `TARGET_SAMPLE_RATE` with a warning if the field is missing.
- **`resample_to_target(samples, src_rate)`** — resamples to 16 kHz via `soxr`
  if installed, else a manual `np.interp` linear resample.
- **`make_callback(source, audio_queue)`** — returns the PortAudio callback for
  a stream. Runs on a real-time audio thread, so it only does the unavoidable
  buffer `copy()` (PortAudio reuses the buffer) and enqueues
  `(source, raw_block)` — no downmix/resample here. Producer side of the
  pipeline.
- **`positive_float(value)`** — argparse type validator; rejects zero/negative
  chunk-duration flags before any device/model work starts.
- **`normalize_remote_url(value)`** — argparse type validator for
  `--remote-url`: strips a trailing slash, requires an `http(s)://host` shape.
- **`parse_args()`** — CLI flags, each defaulting from an env var then a
  constant (see Usage); also cross-checks `--min-chunk-seconds <=
  --max-chunk-seconds` and that `--remote-url`/`--input-file` aren't combined.
- **`format_offset(seconds)`** — converts a segment's float second offset to a
  zero-padded `HH:MM:SS` string. Used by file mode, where the meaningful
  timestamp is the elapsed audio position, not wall-clock `datetime.now()`.
- **`load_model(args)`** — builds the `WhisperModel` (int8, CPU, clamped
  threads, single worker). Used by local-mode live capture and file mode.
- **`report_model_load_failure(exc, model_size)`** — classifies a model-load
  exception (missing package / network problem / bad path / other) into an
  actionable one-line message instead of a generic traceback.
- **`report_remote_health_failure(exc, base_url)`** /
  **`check_remote_health(base_url)`** — remote-mode's equivalent startup
  check: `GET /health`, classified failure message, raised as `RuntimeError`.
- **`RemoteSegment`** — a `NamedTuple` shaped like faster-whisper's `Segment`
  (`text`, `start`, `end`, `no_speech_prob`, `avg_logprob`) so remote-mode
  output can flow through the same hallucination-filter/carry-prompt/
  transcript-writing code as local segments.
- **`encode_wav_bytes(audio)`** / **`build_multipart_request(...)`** — build
  the HTTP request body for a remote transcription call (in-memory WAV
  encoding + manual multipart form, stdlib only).
- **`transcribe_chunk_remote(...)`** — POSTs one chunk to the remote server
  with retry/backoff (`REMOTE_MAX_RETRIES`), raising `RemoteTranscriptionError`
  on exhaustion.
- **`transcribe_chunk_local(...)`** / **`transcribe_chunk(...)`** — the local
  decode call (extracted, unchanged decode params) and the single dispatcher
  that picks local vs. remote based on `--remote-url`.
- **`is_hallucination(segment)`** — `no_speech_prob > 0.85 and avg_logprob <
  -0.9` → drop, regardless of whether the segment came from local or remote
  inference.
- **`combined_initial_prompt(base, carry)`** — merges the user's
  `--initial-prompt` with a source's carried-forward prompt tail.
- **`SourceBuffer`** — per-source accumulator (`append`, `seconds`,
  `tail_rms`, `take_all`, plus a `carry_prompt` field), created lazily via
  `buffers.setdefault(source, SourceBuffer())`.
- **`ready_source(buffers, min_chunk_seconds, max_chunk_seconds)`** — decides
  which source (if any) has a full adaptive chunk ready to transcribe.
- **`transcribe_file(args)`** — file mode. Validates the input path (fail-fast,
  before the model load), loads the model (routed through
  `report_model_load_failure` on failure), then runs one
  `model.transcribe(path, ...)` over the whole file — faster-whisper's bundled
  PyAV/FFmpeg decodes and resamples any format, so there's no manual chunking,
  queue, or `prepare_audio` gate. Writes/flushes each non-empty,
  non-hallucinated segment as `[HH:MM:SS] <stem>: text`.
- **`prepare_audio(audio)`** — computes RMS loudness; returns `None` for
  near-silent chunks (drop them — saves compute, prevents hallucination); else
  applies one-shot automatic gain control toward `TARGET_RMS`, capped at
  `MAX_GAIN`, clipped to `[-1, 1]`.
- **`transcribe_loop(model, audio_queue, output_file, args, source_rates,
  worker_error)`** — the consumer worker thread. Downmixes + resamples each
  dequeued block, buffers audio per source, warns when the backlog exceeds 4×
  the max chunk size, transcribes each ready source via `transcribe_chunk`,
  filters hallucinations, updates the carry-prompt, and writes/flushes each
  timestamped segment. The whole body is wrapped so a crash sets
  `worker_error` and prints a traceback instead of dying silently.
- **`main()`** — wires it together. If `--input-file` is set, it short-circuits
  to `transcribe_file(args)` and exits (no devices/threads). If `--remote-url`
  is set, it health-checks the server before any device enumeration or
  interactive prompts. Otherwise the live path: list devices → resolve
  BlackHole → pick mic → ask filename → load the model (local mode only) →
  start the worker → open both input streams at their native rates → sleep
  until `Ctrl+C`, watching for the worker thread dying. With `--record` it also
  builds a `StreamRecorder` and closes it in a `finally`, so the WAV headers
  and sidecar are finalized on every exit path.
- **`float_to_pcm16(audio)`** — clip to `[-1, 1]` and convert to `int16`.
  Extracted from `encode_wav_bytes` so the remote encoder and the recorder
  cannot drift apart. The clip matters: AGC can push a sample past 1.0, and an
  unclipped conversion wraps it into loud noise of the opposite sign.
- **`detect_gap_frames(expected, written, tolerance)`** — pure; how many
  silence frames to insert so a WAV stays aligned to the wall clock. Returns 0
  within tolerance and on negative drift.
- **`StreamRecorder`** — one 16 kHz mono WAV per source plus the
  `_streams.json` sidecar. `write()` lazily opens on the first block,
  gap-fills, appends and flushes; `close()` finalizes headers and writes the
  sidecar atomically (`os.replace`); `sidecar_payload()` builds the dict
  without touching disk, so its shape is testable on its own. Its single lock
  guards teardown (the main thread closing while the worker writes), not the
  hot path.

---

## Code walkthrough (`diarize.py`)

- **`load_streams_sidecar(path)`** — reads and validates the `_streams.json`
  contract, rejecting an unknown schema rather than half-understanding it, and
  resolves each WAV relative to the sidecar (they are stored as basenames, so
  the whole set can be copied to another machine).
- **`_parse_riff(path)` / `read_wav_mono16(path)`** — walks the RIFF chunks by
  hand instead of using the `wave` module, because a recorder killed mid-write
  can leave a stale `data` size that `wave` would silently honour, handing back
  a truncated recording with no error. The file length is the ground truth.
  Returns float32 mono at 16 kHz, downmixing and resampling (via the
  transcriber's own resampler) so externally recorded audio also works.
- **`resolve_hf_token(env)`** — `HF_TOKEN` → `HUGGINGFACE_HUB_TOKEN` →
  `huggingface_hub.get_token()` → `None`. There is deliberately no
  `--hf-token` flag: a token on the command line lands in shell history and in
  `ps` output. `None` is valid once the model is cached.
- **`DiarizerAccessError` / `report_diarizer_load_failure(exc)`** — pyannote
  signals "no access" two ways: it raises for a gated repo, but *returns None*
  when `from_pretrained` cannot build the pipeline. The None case is converted
  into a typed error so both produce the same actionable steps. The reporter
  returns the message rather than printing it, mirroring
  `transcribe.report_remote_health_failure`, which is what makes it testable.
- **`load_diarizer(token, device)` / `diarize_audio(...)`** — the only
  functions that touch torch, imported inside the function bodies. Audio is fed
  as an in-memory waveform dict rather than a path, which keeps
  torchcodec/ffmpeg out of the path entirely. Prefers
  `exclusive_speaker_diarization` (one speaker per instant), which suits
  attributing a whole transcript segment to a single speaker.
- **`turns_from_annotation(annotation)`** — reads turns via
  `itertracks(yield_label=True)`; bare `__iter__` changed shape between
  pyannote.core 5 and 6, `itertracks` did not.
- **`overlap_seconds` / `speaker_overlap_shares` / `assign_speaker`** — assign
  each segment to the speaker holding the most overlap. Midpoint matching would
  let a 0.3s backchannel that happens to land mid-segment steal the whole line.
  Falls back to the nearest turn when Whisper and the VAD disagree about where
  speech starts, and to no speaker at all when the diarizer found none.
- **`number_others(assignments)`** — maps raw `SPEAKER_NN` labels to
  `Others 1`, `Others 2`, … by first appearance. pyannote's own numbering comes
  out of clustering and is not stable between runs, so it is never displayed.
  Only labels that actually won a segment are numbered, so the transcript can
  never mention an `Others 3` with no lines.
- **`transcribe_stream` / `merge_timeline` / `format_line` /
  `write_diarized_outputs`** — decode one WAV through
  `transcribe.build_decode_options` (so this pass cannot drift from the live
  path's parameters), merge both streams by session time with `You` first on
  an exact tie, and render `[HH:MM:SS] Label: text` on the same wall clock the
  live transcript used.

---

## Key concepts

- **RMS & AGC.** RMS (`sqrt(mean(x²))`) is a cheap loudness proxy used to gate
  silence and drive a minimal per-chunk automatic gain control. It's raw
  amplitude, not perceptually weighted.
- **Producer/consumer + the GIL.** CPython runs one thread of bytecode at a
  time. The audio callbacks do tiny Python-side work (copy + enqueue), and
  Whisper's heavy compute happens in a native extension that releases the GIL,
  so callbacks still fire promptly during a decode.
- **PortAudio real-time callback model.** Stream callbacks run on a high-/
  real-time-priority thread with strict rules (should not allocate or block).
  Docs: <https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html>.
- **Silero VAD (`vad_filter=True`).** faster-whisper runs the Silero
  voice-activity-detection model ahead of the encoder to strip non-speech
  regions, cutting compute and silence hallucinations. Docs:
  <https://github.com/SYSTRAN/faster-whisper>.
- **int8 quantization (`--compute-type`).** 8-bit integer weights instead of
  float — smaller and faster on CPU for a small accuracy cost. On this arm64
  Mac `ctranslate2.get_supported_compute_types("cpu")` returns exactly
  `float32`, `int8`, `int8_float32` — no fp16/bf16, and `int8` and
  `int8_float32` are the same thing here, so `float32` (unquantized) is the
  only real precision step. The flag is validated against that list at startup.
- **Presets vs. explicit flags.** `resolve_settings()` collapses preset, env
  var and flag into one `Settings`, and records which names the user actually
  set. That set — not a comparison against a default value — is what decides
  whether a flag is reported as ignored in remote mode; comparing to a default
  cannot tell "typed the default" from "the preset supplied it", and breaks
  whenever a default moves.
- **Beam search vs greedy.** `beam_size` tracks that many candidate token
  sequences in parallel; `beam_size=1` is greedy (always take the single best
  next token) — fastest, more prone to local-optimum errors.

---

## Usage

```bash
cd ~/meeting-transcriber
source venv/bin/activate

# 1. Route meeting audio (once per session): pick the output you hear through.
#    This updates the Multi-Output Device to include BlackHole 2ch.
switch_meeting_output
#    Then set the Microsoft Teams speaker to "Multi-Output Device".

# 2. (Optional) validate that BlackHole is receiving audio.
python test_audio.py

# 3. Start transcribing. Prompts only for mic device + transcript name.
python transcribe.py
```

Transcripts are written to `./transcripts`. Running `/meeting-notes` on a
transcript from this repo writes the summary to `./notes`.

### Transcribe an existing audio file (no capture)

To transcribe a recording you already have on disk, pass `--input-file`. This
skips device selection, mic, and BlackHole entirely — it decodes the whole file
in one pass and writes a transcript. Any container/codec FFmpeg can read works
(`.wav`, `.mp3`, `.m4a`, ...); faster-whisper's bundled PyAV handles decoding
and resampling, so no extra dependency and no system `ffmpeg` is needed.

```bash
python transcribe.py --input-file /path/to/recording.m4a
```

The transcript is named after the input file (`transcripts/<stem>_<timestamp>.txt`) and each
line is prefixed with the file stem instead of `You` / `Others`, since a single
mixed file has no per-speaker separation. Timestamps are the elapsed **audio**
position (`[HH:MM:SS]` from the segment's start offset), not wall-clock time.

### CLI flags / environment variables

Each flag falls back to an environment variable, then a built-in default:

| Flag | Env var | Default | Purpose |
| --- | --- | --- | --- |
| `--accuracy` | `WHISPER_ACCURACY` | `balanced` | preset setting model, beam, chunk length, compute type and threads together |
| `--model` | `WHISPER_MODEL` | preset | faster-whisper model size (local mode only) |
| `--min-chunk-seconds` | `WHISPER_MIN_CHUNK_SECONDS` | preset | min seconds before a silence cut is considered |
| `--max-chunk-seconds` | `WHISPER_MAX_CHUNK_SECONDS` | preset | hard ceiling on seconds per decode; capped at 30s |
| `--beam-size` | `WHISPER_BEAM_SIZE` | preset | beam width; 5 = more accurate, slower (local mode only) |
| `--compute-type` | `WHISPER_COMPUTE_TYPE` | preset | `int8`, `int8_float32` or `float32`; validated against this CPU |
| `--cpu-threads` | `WHISPER_CPU_THREADS` | preset | CTranslate2 threads; more is not always faster with efficiency cores |
| `--patience` | `WHISPER_PATIENCE` | preset | beam-search patience; only applies on the temperature-0 pass |
| `--language` | `WHISPER_LANGUAGE` | `en` | language hint; empty = auto-detect |
| `--initial-prompt` | `WHISPER_INITIAL_PROMPT` | `""` | vocabulary hint (names, acronyms, jargon). **Only biases the first window of each decode call** — in file mode, only the first segment of the whole file |
| `--input-file` | `WHISPER_INPUT_FILE` | `""` | transcribe this audio file instead of live capture; skips mic/BlackHole |
| `--remote-url` | `WHISPER_REMOTE_URL` | `""` | run inference on a remote faster-whisper server instead of locally; see Remote (GPU) inference below |
| `--record` | `WHISPER_RECORD` | off | also write one 16 kHz mono WAV per stream plus a `_streams.json` sidecar, for the post-meeting diarization pass; see Speaker diarization below |

`--model`, `--beam-size`, `--compute-type` and `--cpu-threads` are ignored when
`--remote-url` is set — the remote server controls its own model. You are told
only if you set one of them explicitly; a preset supplying them is not an
override.

### Accuracy presets

`--accuracy` sets several knobs that only make sense together — a bigger model
is affordable only if the chunk length rises with it. Individual flags override
any preset value.

| | `fast` | `balanced` (default) | `max` |
| --- | --- | --- | --- |
| model | `small.en` | `large-v3-turbo` | `large-v3-turbo` |
| beam size | 1 | 2 | 5 |
| min/max chunk | 6s / 20s | 12s / 28s | 18s / 28s |
| compute type | `int8` | `float32` | `float32` |
| CPU threads | 4 | 4 | 4 |
| word timestamps | off | off | on |
| measured RTF | 0.14 | 0.27 | ~0.50 |

`fast` is byte-identical to the pre-preset behaviour, so it is always the
no-regression escape hatch if a bigger preset cannot keep up. `max` is intended
for `--input-file`, where there is no real-time constraint.

#### Measured on an Apple M4 (4P+6E, 16 GB)

Replaying `outputs/audio-office-isp-update.wav` on both sources (99.4s of audio)
through the real chunked pipeline. RTF = decode seconds / audio seconds, so
lower is better and anything under 1.0 is faster than real time.

| configuration | calls | decode | RTF | p95/call |
| --- | --- | --- | --- | --- |
| small.en beam1 chunk6 (`fast`) | 6 | 13.7s | 0.14 | 3.92s |
| small.en beam1 chunk12 | 4 | 9.1s | 0.09 | 2.36s |
| small.en beam5 chunk12 | 4 | 16.9s | 0.17 | 4.49s |
| distil-medium.en beam5 chunk12 | 4 | 17.9s | 0.18 | 4.52s |
| medium.en beam5 chunk12 | 4 | 35.2s | 0.35 | 9.48s |
| large-v3-turbo beam1 float32 chunk12 | 4 | 18.0s | 0.18 | 4.48s |
| **large-v3-turbo beam2 float32 chunk12 (`balanced`)** | 4 | 26.6s | **0.27** | 7.97s |
| large-v3-turbo beam5 int8 chunk12 | 4 | 45.1s | 0.45 | 11.53s |
| large-v3-turbo beam5 float32 chunk12 | 4 | 39.9s | 0.40 | 10.40s |

Three findings worth keeping:

1. **`float32` is faster than `int8` for `large-v3-turbo` on this chip** — 0.40
   vs 0.45 cold, and 0.50 vs 0.70 warm, reproduced back to back. The usual
   "int8 is the fast one" intuition comes from x86; CTranslate2's int8 GEMM path
   is not the fast one on arm64 here. Since `float32` is also the more accurate
   option, it wins outright for this model, and `balanced`/`max` use it.
2. **Sustained load costs ~50%.** A second consecutive run of the same
   configuration measured 0.70 where the first measured 0.45 — thermal
   behaviour, not noise. Preset headroom is sized against the warm number.
3. **A bigger model beats a wider beam per second spent.** `large-v3-turbo` at
   beam 1 (RTF 0.18) costs half of `medium.en` at beam 5 (0.35) and is the
   stronger model. That is why `balanced` upgrades the model first and only then
   widens the beam to 2.

Accuracy caveat: the bundled 50s sample contains no domain vocabulary, so it
separates the configurations on speed but **not** on accuracy — every one of
them produced the same words. The accuracy ordering is taken from the models'
published quality, not measured here. To measure accuracy properly, meeting
audio has to be kept (the WAV write-ahead in
`.claude/plans/gpu-transcription-plan.md`), which is not implemented.

Longer chunks trade on-screen latency for accuracy and throughput: a line can
appear up to `min_chunk + decode` seconds after it was spoken. Timestamps are
unaffected — they come from capture time, not write time.

Benchmark the presets on your own machine with
`python outputs/bench_presets.py`, which replays audio through the real chunked
pipeline (a whole-file decode would not reproduce the per-call padded-encoder
cost that governs live behaviour).

### Remote (GPU) inference (optional)

Pass `--remote-url http://<host>:<port>` to send audio to a remote
faster-whisper server instead of running inference locally:

```bash
python transcribe.py --remote-url http://192.168.77.1:8000
```

- Live-capture only — not combinable with `--input-file` yet.
- `--language`, `--initial-prompt` and `--beam-size` are forwarded to the
  server as multipart fields, projected out of the same `build_decode_options`
  dict the local paths decode with. A server that does not declare a field
  simply ignores it. Local-execution knobs (`--compute-type`, `--cpu-threads`,
  `--model`) are never sent — the server controls those.
- On startup, `transcribe.py` checks `GET <url>/health` before asking for a
  mic/transcript name — a dead or unreachable server fails fast with a clear
  message instead of after you've answered the prompts.
- A failed chunk is retried a couple of times, then dropped with a visible
  `[transcription dropped — remote server error]` marker in the transcript
  (never silently missing); several consecutive failures end the session.
- **This flag requires an actual L1 inference server that has not been built
  yet** — see `docs/project-local-NW-model-setup.md` for the (not-yet-built)
  server design. Until that exists, `--remote-url` will fail its startup
  health check.

### Speaker diarization (post-meeting, optional)

`You` and `Others` are already exact — your mic is a physically separate
stream, so the local speaker is known rather than guessed. Diarization only
has to answer the remaining question: *which* remote participant is speaking
inside the mixed `Others` stream. `diarize.py` does that after the meeting.

```bash
# One-time setup
pip install -r requirements-diarize.txt      # pulls torch; ~1 GB installed
# Accept the model terms at
#   https://huggingface.co/pyannote/speaker-diarization-community-1
# then create a read token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...                        # or: hf auth login

# 1. Record the meeting as well as transcribing it
python transcribe.py --record

# 2. Afterwards, split "Others" into individual speakers
python diarize.py transcripts/<name>_<timestamp>_streams.json
```

Output, beside the live transcript:

```text
[10:15:03] You: right, let's get started
[10:15:07] Others 1: sounds good, I'll share my screen
[10:15:19] Others 2: can you make that a bit bigger
```

Useful flags: `--num-speakers N` (or `--min-speakers`/`--max-speakers`) when
you know the headcount, `--offsets` to timestamp from session start instead of
wall clock, `--mark-uncertain` to flag lines that straddle a speaker change,
`--skip-you`, and `--no-diarize` to re-transcribe and merge without pyannote
(useful for checking the pipeline before installing torch). `--device mps` is
opt-in: some pyannote operations fall back to CPU and can end up slower.

Notes:

- The model is gated but free (CC-BY-4.0), and runs **fully offline** once
  cached — verify with `HF_HUB_OFFLINE=1`.
- The recording costs roughly 230 MB per hour for both streams.
- Installing torch also slows `transcribe.py` startup by about a second:
  `ctranslate2` imports torch opportunistically when it is present.

---

## Project status

- **Phase 1 — Simple transcriber:** ✅ done (device picker, dual input streams
  into a shared queue, local CPU int8 inference).
- **Phase 2 — Dual stream (You + Others):** ✅ implemented. `switch_meeting_output`
  (Swift) manages routing; `transcribe.py` auto-detects BlackHole and labels
  `You` / `Others`.
- **Phase 3 — Speaker diarization:** ✅ implemented as a post-meeting pass.
  `transcribe.py --record` writes one WAV per stream plus a `_streams.json`
  sidecar; `diarize.py` re-transcribes them, runs
  `pyannote/speaker-diarization-community-1` over the `Others` stream, and
  writes `<name>_diarized.txt` / `.json` with `Others 1`, `Others 2`, …
  ⬜ Remaining: accept the model terms on Hugging Face and set `HF_TOKEN`
  once, then run it against a real meeting.
- **AUDIT.md findings:** ✅ all 6 resolved (adaptive chunking, hallucination
  filter, carry-forward prompt, native sample rate + resample, worker crash
  detection, narrow exception handling) — see `docs/AUDIT.md`.
- **Remote GPU inference (client-side):** ✅ `--remote-url` flag, HTTP client,
  and response handling implemented. ⬜ Blocked on the L1 server itself
  (Windows RTX4050 FastAPI service) — separate infrastructure work not yet
  started; see `docs/project-local-NW-model-setup.md`.

See `project.md` and `phase2_audio_routing_plan.md` for the full brief, and
`AUDIT.md` for the (now-resolved) code-quality findings.

---

## Sources

- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>
- python-sounddevice streams API:
  <https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html>
- pyannote speaker-diarization-community-1 (model card, gating, `token=`,
  waveform input, `exclusive_speaker_diarization`):
  <https://huggingface.co/pyannote/speaker-diarization-community-1>
- pyannote.audio releases (4.0 breaking changes, torchcodec, Python ≥ 3.10):
  <https://github.com/pyannote/pyannote-audio/releases>
- Why community-1 over 3.1, and what exclusive diarization is for:
  <https://www.pyannote.ai/blog/community-1>
