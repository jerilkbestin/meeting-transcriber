# transcribe.py: port AUDIT.md fixes + teams-notetaker ASR engineering

**Status: implemented (2026-09-08).** Also mirrored at
`/Users/hackerboy/.claude/plans/compare-the-transcribe-py-and-clever-minsky.md`.

## Context

Follow-up to a prior comparison of `transcribe.py` vs `teams-notetaker/`
(teams-notetaker came out better-engineered on chunking, hallucination
filtering, tests, and error handling). User now wants those wins folded into
`transcribe.py` itself, plus every finding in `docs/AUDIT.md` (6 deferred
correctness/robustness issues, none fixed yet) resolved — using
`teams-notetaker`'s code as a reference where it actually solves the same
problem, and an independent fix where it doesn't.

Two research passes (Explore agents) checked each AUDIT.md finding against
teams-notetaker's actual code line-by-line, and pulled its adaptive-chunking/
hallucination-filter/prompt-carry logic verbatim. A Plan agent then designed
the concrete port. This file is that reviewed plan — verified against direct
repo evidence for both projects, plus targeted WebSearch/WebFetch checks
against official docs for every external API used (soxr, sounddevice,
faster-whisper), since this is an implementation task and those calls are
central to the design (see **External behavior verification**).

**Scope guardrail:** platform, single-file structure, and existing CLI flags
outside what's listed below are unchanged. This is not a rewrite — it's the 6
audit fixes plus 3 ASR-pipeline upgrades, landed as a series of small,
independently verifiable steps on the existing file.

## AUDIT.md findings vs. teams-notetaker — verified verdicts

| # | Finding | teams-notetaker verdict | Action |
|---|---|---|---|
| 1 | Callback allocates/blocks (downmix+reshape inside PortAudio real-time thread) | **Not solved — worse.** Its `TrackCapture._callback` (`audio.py`) does downmix + `soxr.resample` + a **locked WAV write** inside the callback, despite its own docstring claiming minimal work. | Do NOT port. Implement AUDIT's own fix directly in transcribe.py. |
| 2 | `--chunk-seconds 0`/negative hangs a worker thread | **Not solved.** No `ValueError`/argparse validation anywhere; an incidental `secs <= 0.25` floor avoids the exact infinite spin but still allows CPU-thrashing on tiny fragments. | Design own `argparse` validation. |
| 3 | Silent worker-thread death | **Not solved.** Its `feeder` thread's `transcriber.feed(...)` call is unwrapped; zero hits for `is_alive` anywhere in the package. | Design own fix. |
| 4 | `buffers` dict keyed to literal `"You"`/`"Others"` strings, `KeyError` risk | **Solved.** Shared `TRACK_MIC`/`TRACK_SYSTEM` constants used identically on producer and consumer sides, plus `dict.setdefault` so an unrecognized label never raises. | Port the pattern (constants + `setdefault`). |
| 5 | No sample-rate mismatch guard (hardcodes 16 kHz on the device open) | **Solved.** Opens the stream at the device's own native rate (`defaultSampleRate`), resamples to 16 kHz in software via `soxr` with a linear-interpolation fallback if `soxr` isn't installed. | Port the pattern. |
| 6 | Broad `except Exception` in `main()` swallows all setup errors | **Partially solved.** `main()` itself has zero try/except (unexpected errors get a full traceback); known failures (missing deps, device errors, model-load failure) are caught narrowly with a dedicated `_report_model_load_failure` classifier. Its own fallback branch still loses the traceback for genuinely novel errors — same residual gap, just localized. | Port the narrow-catch structure. |

## New/changed constants

```
TARGET_SAMPLE_RATE = 16000        # renamed from SAMPLE_RATE — disambiguates from per-device native rate
SOURCE_YOU = "You"
SOURCE_OTHERS = "Others"
DEFAULT_MIN_CHUNK_SECONDS = 6.0
DEFAULT_MAX_CHUNK_SECONDS = 20.0
SILENCE_WINDOW_SECONDS = 0.6
CHUNK_SILENCE_RMS = 0.006         # cut-point detector — distinct from existing SILENCE_RMS=0.0015 (keep/drop gate)
MIN_READY_SECONDS = 0.25
HALLUCINATION_NO_SPEECH_PROB = 0.85
HALLUCINATION_MAX_AVG_LOGPROB = -0.9
CARRY_PROMPT_MAX_CHARS = 220
```
Existing AGC constants (`SILENCE_RMS`, `TARGET_RMS`, `MAX_GAIN`) and `prepare_audio`'s formula are **untouched** — this is transcribe.py's own edge over teams-notetaker (which has no AGC) and must not regress.

## Implementation steps (ordered; each independently verifiable)

**Step 0 — prep.** Add `import sys`, `import traceback`. Rename `SAMPLE_RATE` → `TARGET_SAMPLE_RATE` at all call sites. Verify: `python -m py_compile transcribe.py`; `--help` still works.

**Step 1 — AUDIT #4.** Add `SOURCE_YOU`/`SOURCE_OTHERS` constants, used at both `make_callback(...)` call sites in `main()`. Add a `SourceBuffer` class (`samples`, `carry_prompt`, `.append`, `.seconds()`, `.tail_rms(window)`, `.take_all()`) — this doubles as scaffolding for Steps 4 and 6. Replace the literal `buffers = {"You": ..., "Others": ...}` with `buffers.setdefault(source, SourceBuffer())` on every access. Pure refactor, no behavior change yet. Verify: live smoke test, output identical to before.

**Step 2 — AUDIT #1.** Rewrite `make_callback`'s `callback` to do only `audio_queue.put((source, indata.copy()))` (plus the existing `status` print) — no `.astype`, `.mean`, or `.reshape` on the real-time thread. Move that downmix into `transcribe_loop` right after `audio_queue.get()`. **Do not copy teams-notetaker's callback placement** (Step 1's finding — it does the opposite of what we want here); only the resampling *function* from teams-notetaker (next step) is worth reusing, not where it runs.

**Step 3 — AUDIT #5.** Add `device_sample_rate(devices, index)` (reads `devices[index]["default_samplerate"]` — confirmed key name below) with a fallback + warning if unavailable. Add `resample_to_target(samples, src_rate)`: pass-through if `src_rate == TARGET_SAMPLE_RATE`, else `soxr.resample(samples, src_rate, TARGET_SAMPLE_RATE)` (confirmed signature below) behind a `try/except ImportError`, else a manual `np.interp` linear fallback (mirrors teams-notetaker exactly). Open each `sd.InputStream` at its own device's native rate instead of the hardcoded constant; resample in `transcribe_loop` right after downmixing, before appending to the `SourceBuffer` — everything downstream stays in 16 kHz units unchanged. Create `requirements.txt` (repo root — **confirmed it doesn't exist yet**): `sounddevice`, `numpy`, `faster-whisper`, `soxr>=0.5.0` as normal (non-optional) dependencies — resampling runs on essentially every real block (both mic and BlackHole are almost never natively 16 kHz), so this isn't an edge case worth leaving to a soft-fail path alone; the linear-interpolation fallback stays only as a safety net for an incomplete install, matching teams-notetaker's own stance (it pins `soxr>=0.5.0` unconditionally too, despite writing the same fallback).

**Step 4 — AUDIT #2 + adaptive silence-aware chunking (combined).** Add a `positive_float` argparse type validator. Replace `--chunk-seconds`/`WHISPER_CHUNK_SECONDS` with `--min-chunk-seconds`/`--max-chunk-seconds` (no back-compat shim — single-file personal tool, one documented consumer, `README.md` updated in the same change). **Critical detail:** pass env-var defaults as raw strings (`os.getenv(..., str(DEFAULT))`), not pre-converted numbers — the existing `int(os.getenv(...))` pattern used by `--beam-size` would bypass `type=positive_float` entirely for the env-var path and silently reintroduce AUDIT #2. After parsing, cross-check `min <= max` via `parser.error(...)`. Add `ready_source(buffers, min_chunk_seconds, max_chunk_seconds)`: a source is ready once it has ≥ `min_chunk_seconds` AND a quiet tail (`tail_rms(SILENCE_WINDOW_SECONDS) < CHUNK_SILENCE_RMS`), or unconditionally at `max_chunk_seconds` (hard ceiling). Replace `transcribe_loop`'s fixed-size `while len(buffers[source]) >= chunk_size` with a loop over `ready_source(...)`. Update the backlog-warning threshold to scale off `max_chunk_seconds * 4` (same 30s throttle, same logic otherwise).

**Step 5 — hallucination filter.** `is_hallucination(segment)`: `segment.no_speech_prob > HALLUCINATION_NO_SPEECH_PROB and segment.avg_logprob < HALLUCINATION_MAX_AVG_LOGPROB` (fields confirmed on faster-whisper's `Segment` dataclass below). Apply in both `transcribe_loop` and `transcribe_file`'s segment loops, in addition to (not instead of) the existing `vad_filter=True` and RMS gate.

**Step 6 — per-source carry-forward `initial_prompt`.** `combined_initial_prompt(base, carry)`: joins both if present, else whichever is non-empty, else `None`. After each chunk, store the kept segments' trailing `CARRY_PROMPT_MAX_CHARS` chars onto that source's `SourceBuffer.carry_prompt`; read it back into the next chunk's `initial_prompt` for the same source. `condition_on_previous_text=False` stays exactly as-is — this is the deliberate anti-hallucination-loop stance already documented in the file, not something this change touches. No new thread/lock needed: `transcribe_loop` is a single dequeue+transcribe thread, so writing `carry_prompt` back onto the same long-lived `SourceBuffer` is already safe.

**Step 7 — AUDIT #3.** Add a `threading.Event` (`worker_error`). Wrap `transcribe_loop`'s whole body in try/except: on any exception, `traceback.print_exc()` then `worker_error.set()`. In `main()`, change the live poll loop to check `worker_error.is_set() or not worker.is_alive()` each second and, if either is true, print a clear error + the partial-transcript path, then `sys.exit(1)`.

**Step 8 — AUDIT #6.** Add `report_model_load_failure(exc, model_size)` (same three-branch classification as teams-notetaker's `_report_model_load_failure`: import/missing-package, network/timeout, bad path/typo, else generic). Remove the single blanket `except Exception as exc: print(...)` in `main()`. Replace with narrow catches: `RuntimeError` around `find_blackhole_input` (missing BlackHole), `Exception` around `load_model` routed through `report_model_load_failure` (both call sites: live path and `transcribe_file`), `sd.PortAudioError` around the `with sd.InputStream(...)` block, `FileNotFoundError` around the file-mode path check. Everything else propagates with a full traceback. Do this step **last** — it depends on `PortAudioError`-raising code (Step 3) and worker-death signaling (Step 7) already existing, and removing the catch-all is a good forcing function for anything an earlier step missed.

**Step 9 — add `selftest.py`** (new file, repo root — confirmed no `selftest.py` exists there yet; distinct from the existing `test_audio.py`, which the README confirms is an interactive real-hardware BlackHole smoke test, not a regression suite). Offline, no pytest, no hardware, no model download — same `check(name, cond, detail)` + `FakeModel`/`FakeSegment` pattern as teams-notetaker's version, scaled down to this file's much smaller scope. Cover: compiles; `positive_float` rejects 0/negative (AUDIT #2 regression pin); `buffers.setdefault` never raises for an arbitrary label (AUDIT #4 regression pin); `resample_to_target` round-trip length/dtype, once with real `soxr`, once with `_HAVE_SOXR` forced `False`; `ready_source` returns only once both `min_chunk_seconds` and a quiet tail are met, and separately once `max_chunk_seconds` is hit regardless of loudness; hallucination filter drops/keeps the right fake segments; `combined_initial_prompt`'s four cases; worker-error signaling sets the event and returns instead of hanging; a regression pin for the *existing* AGC math (silence → `None`, quiet → scaled toward `TARGET_RMS` capped at `MAX_GAIN`, loud → unchanged); one `subprocess` end-to-end check (`--help` exits 0; `--min-chunk-seconds 20 --max-chunk-seconds 5` exits nonzero with "cannot be greater than" on stderr).

**Step 10 — docs.** Update `README.md`:
- Diagram (lines 42–48) and the paragraph at lines 72–76 currently describe the callback as doing the downmix ("copy + downmix to mono" / "Each callback does only cheap work — copy the block, downmix to mono... and returns immediately") — this becomes inaccurate after Step 2 and must be corrected to: callback copies + enqueues raw audio only; downmix + resample now happen in `transcribe_loop`.
- "Design & rationale" bullet at lines 97–100 ("Batch-chunked... buffered to `chunk_seconds` (default 10s)") → rewrite to describe adaptive min/max silence-aware chunking.
- Code walkthrough bullet for `make_callback` (lines 141–144) → update to match the new minimal callback.
- CLI flags table (lines 241–249): drop `--chunk-seconds`/`WHISPER_CHUNK_SECONDS`, add `--min-chunk-seconds`/`--max-chunk-seconds` + their env vars/defaults.
- Add walkthrough entries for the new functions (`SourceBuffer`, `ready_source`, `resample_to_target`, `device_sample_rate`, `is_hallucination`, `combined_initial_prompt`, `report_model_load_failure`).

Update `docs/AUDIT.md`: mark all 6 findings resolved (one line each: "Fixed — see `<function>`"), per the file's own stated purpose as a deferred-findings tracker.

## Explicitly out of scope / must not change

Single-file structure; macOS/BlackHole architecture and the `switch_meeting_output` Swift routing split; `transcribe_file`'s one-shot whole-file decode design (only gets the hallucination filter + `report_model_load_failure`); AGC constants/formula; `--model`/`--beam-size`/`--language`/`--initial-prompt`/`--input-file` flags and defaults; `condition_on_previous_text=False`; `daemon=True` worker with `Ctrl+C`-driven exit (Step 7 adds *detection*, not a shutdown redesign); `.txt` output format and naming; single dequeue+transcribe worker thread (no second feeder thread or lock, unlike teams-notetaker — that concurrency shape solves a problem this design doesn't have); unbounded `queue.Queue()` (no drop-on-full — the backlog warning already gives visibility without ever discarding live audio, a legitimate different tradeoff from teams-notetaker's drop-counter approach).

## External behavior verification

Three external API claims are load-bearing for this plan and were verified against official docs/source before being written in as fact (not from training knowledge alone):
- `sd.query_devices()` device dict key for native sample rate is `'default_samplerate'` — confirmed via [python-sounddevice checking-hardware docs](https://python-sounddevice.readthedocs.io/en/0.5.1/api/checking-hardware.html).
- `soxr.resample(x, in_rate, out_rate, quality='HQ')` signature — confirmed via [Python-SoXR docs](https://python-soxr.readthedocs.io/en/latest/soxr.html).
- faster-whisper's `Segment` dataclass has `avg_logprob` and `no_speech_prob` fields — confirmed via [faster-whisper source, transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).
- The PortAudio real-time callback "must not allocate or block" constraint was already cited with a doc link in `docs/AUDIT.md` itself (finding #1) — not re-verified here, treated as already-sourced.

## Verification (end-to-end, after all steps land)

- `python -m py_compile transcribe.py` and `python transcribe.py --help`.
- `python transcribe.py --min-chunk-seconds 0` / `--max-chunk-seconds -1` → clean `argparse` error, no device/model work attempted.
- `python transcribe.py --min-chunk-seconds 20 --max-chunk-seconds 5` → clean `parser.error` cross-check message.
- Live run: confirm both `You`/`Others` streams still transcribe, chunks now cut sooner after a pause and hard-cut at `--max-chunk-seconds` during continuous speech, no new PortAudio `status` warnings.
- No BlackHole connected → one-line friendly error, no traceback. `--input-file /no/such/file` → one-line friendly error. `--model bogus-name` → `report_model_load_failure`'s bad-path branch. A genuinely unexpected error (e.g. read-only `transcripts/` dir) → full traceback, not swallowed.
- `python selftest.py` (new) → all checks pass, no pytest/hardware/model download required.

---

# Part 2: `--remote-url` flag — choose local vs. remote (L1/L2) inference

## Context

Follow-up request: add a flag so `transcribe.py` can send audio to a
GPU-hosted faster-whisper server on another PC instead of always running
inference locally. **This lands after Part 1 above (Steps 0–10)** — it
depends on `SourceBuffer`, `ready_source`, `combined_initial_prompt`,
`is_hallucination`, `report_model_load_failure`, and `worker_error` already
existing. Numbered as a continuation: **Step 11 onward.**

**Correction to an earlier search error in this session:** I initially told
the user `docs/project-local-NW-model-setup.md` didn't exist — that was
wrong; my search only checked `.claude/plans/` directories, never `docs/`.
The file exists (17KB, `docs/project-local-NW-model-setup.md`, dated
2026-08-24, status "decisions settled, build not started") and is the real
authoritative L1/L2 spec. It's now been read in full and is the source of
truth below, superseding the `gpu-transcription-plan.md`-only summary a Plan
agent initially worked from.

**Contract correction found by re-reading the real doc:** the doc's own
reference `server.py` (§5.2, not yet built) returns one flat
`{"text": ..., "language": ..., "duration": ...}` string per chunk — no
per-segment `no_speech_prob`/`avg_logprob`, which the hallucination filter
needs to work identically in both modes. **User decision: extend that
server.py sketch** to also return a `segments` array with those fields
(trivial — faster-whisper already computes them per segment server-side;
nothing is built yet, so this isn't a breaking change to running code, just
to a documented-but-unbuilt snippet).

## Scope split: this plan = L2 code only; L1 build = separate, deferred, docs-only

**User decision (2026-09-04): everything L1-side (the wired network link, and actually building/running the Windows FastAPI+faster-whisper server) is pushed to later and is owner-run infrastructure, not part of this execution plan.** This plan's Steps 11–21 below cover 100% L2 code — the `--remote-url` flag, HTTP client, and response handling inside `transcribe.py` — none of which require L1 to exist to implement or offline-test (see Verification).

The "L1 code plan" — what the eventual server needs to look like, including the `segments`-contract fix decided in this session — is **not written into this execution plan**. Instead, Step 19 updates the *existing* `docs/project-local-NW-model-setup.md` (§5.2's `server.py` sketch) in place, since that file already is the designated L1 build doc (Parts A/B, "decisions settled, build not started"); it just needed its response-contract corrected. No new docs file is created — updating the existing one avoids a second, redundant L1 doc.

## Topology facts the L2 client code depends on (from `docs/project-local-NW-model-setup.md`, for reference only — building this is not this plan's job)

- L1 = Windows ASUS ROG Strix, RTX 4050, static IP `192.168.77.1`, port `8000` (once built).
- Endpoints the client will call: `GET /health`, `POST /v1/audio/transcriptions` (multipart: `file`, `initial_prompt` form field, `language` form field — **no `beam_size` field in the real server sketch**; `gpu-transcription-plan.md` already settled beam_size=5 as a server-side-only constant, not client-forwarded).
- Server model (`large-v3-turbo`, int8, CUDA) is entirely the server's own choice; `--model` is meaningless remotely.
- Latency budget target: <10s round-trip per chunk — informs `REMOTE_CHUNK_TIMEOUT_SECONDS` below, nothing else.

## Flag design

Single flag, presence-as-toggle (mirrors the existing `--input-file` idiom exactly): **`--remote-url URL`** / env var `WHISPER_REMOTE_URL` / default `""`. Rejected a separate bool+URL pair — one non-blank string is one source of truth, no invalid-combination state to guard.

- `REMOTE_HEALTH_PATH = "/health"`, `REMOTE_TRANSCRIBE_PATH = "/v1/audio/transcriptions"` — flag takes a base URL only (`http://192.168.77.1:8000`), paths appended internally.
- New `normalize_remote_url(value)`: strips whitespace/trailing slash, validates `scheme in ("http","https")` + non-empty `netloc` via `urllib.parse.urlparse`; `parser.error(...)` on an invalid non-blank value.
- Mutual exclusion: `--remote-url` + `--input-file` together → `parser.error(...)` (remote mode is live-capture-only this iteration — `transcribe_file` hands a whole file straight to `model.transcribe(path)` with no chunking loop to hook a per-chunk HTTP call into; building that is a materially separate piece of work, left as a future extension).
- `--model` **and `--beam-size`** are both meaningless remotely (server hardcodes its own model and beam_size=5) — same soft-warning treatment for both, not an error: one line printed only if the user explicitly passed a non-default value, e.g. `Note: --model/--beam-size are ignored in remote mode; the server controls both.` Neither is forwarded in the remote request.
- `--language` and `--initial-prompt` (via `combined_initial_prompt`, unchanged from Part 1) **are** forwarded — both map directly onto the real server's `language`/`initial_prompt` form fields.

## Response contract (extends the doc's server.py sketch)

```json
{
  "text": "…",
  "language": "en",
  "duration": 12.4,
  "segments": [
    {"text": "hello there", "start": 0.42, "end": 1.85, "no_speech_prob": 0.03, "avg_logprob": -0.21}
  ]
}
```
`text`/`language`/`duration` stay exactly as the doc's sketch already returns them (compatibility/debugging). `segments` is the addition the client actually parses; `[]` is valid (silent/VAD-filtered chunk, not an error). Missing `no_speech_prob`/`avg_logprob` on an individual segment default to values that never trigger the hallucination filter (bias toward keeping real speech), plus a one-time console warning if a server response is missing them.

## HTTP client — library, encoding, failure policy

**stdlib only** (`urllib.request`/`.error`/`.parse`, `json`, `io`, `wave`, `uuid`) — no new dependency, even though the doc's own client snippet uses `requests`. Justified because Part 1's `requirements.txt` lists only always-needed audio/ML deps; remote mode is opt-in and off by default, so it shouldn't force an unconditional new dependency for a ~25-line manual multipart encoder. (Flagging this as a deliberate deviation from the doc's snippet, not an oversight.)

- `encode_wav_bytes(audio)`: AGC'd float32 → 16-bit PCM → mono WAV bytes via stdlib `wave` (in-memory only, purely to build the request body — **not** disk write-ahead durability, which stays out of scope).
- `build_multipart_request(url, wav_bytes, fields)`: manual multipart body (`uuid.uuid4().hex` boundary), one part per non-`None` field (`language`, `initial_prompt`) + the `file` part; `Content-Type` header set manually, `Content-Length` left to `urllib` to compute.
- Timeouts: `REMOTE_HEALTH_TIMEOUT_SECONDS = 5.0`, `REMOTE_CHUNK_TIMEOUT_SECONDS = 20.0` (headroom above the doc's stated <10s budget, not tight against it).
- Failure policy: `REMOTE_MAX_RETRIES = 2` (3 attempts, backoff `1.0s`/`2.0s`) → on exhaustion, raise `RemoteTranscriptionError`; caller prints a loud warning **and** writes a visible transcript marker line (`[HH:MM:SS] <source>: [transcription dropped — remote server error]`) since no disk write-ahead exists to fall back on, then drops that chunk and continues. `REMOTE_CONSECUTIVE_FAILURE_LIMIT = 5` (reset on any success) escalates to a plain `RuntimeError`, which Part 1's Step 7 worker-loop try/except already catches — reuses that machinery unchanged, just a new trigger for it.

## Integration point

`RemoteSegment` (a `NamedTuple`: `text, start, end, no_speech_prob, avg_logprob`) duck-types against faster-whisper's `Segment` for every field `is_hallucination`/carry-prompt/transcript-writing touch. Single dispatcher:
```
transcribe_chunk(model, audio, language, initial_prompt, args) -> list[segment-like]
    if args.remote_url: return transcribe_chunk_remote(audio, language, initial_prompt, args.remote_url)
    else:                return transcribe_chunk_local(model, audio, language, args.beam_size, initial_prompt)
```
Wired in exactly where Part 1's inline `model.transcribe(...)` call sits in `transcribe_loop` — everything downstream (`is_hallucination`, carry-prompt update, transcript writing) is unmodified and unaware of which mode ran.

## Startup handling

Remote mode replaces `load_model(args)` with `check_remote_health(base_url)` (`GET /health`, 200 = OK; any failure classified by a new `report_remote_health_failure` — connection-refused/DNS/timeout/HTTP-error/generic, same style as `report_model_load_failure` — then re-raised as `RuntimeError`). Placed at the very top of the live-mode path in `main()`, **before** device enumeration and the interactive mic/filename prompts (fail-fast on a dead server before asking the user two questions) — mirrors this file's own existing "validate cheap things before expensive ones" principle from `transcribe_file`'s path check. Settings-summary print gains a `Mode : remote (<url>)` / `Mode : local` line.

## Docs impact

- `README.md`: new `--remote-url`/`WHISPER_REMOTE_URL` flags-table row + footnote (`--model`/`--beam-size` ignored remotely); new "Remote (GPU) inference (optional)" usage subsection with an explicit caveat that L1 isn't built yet; Project-status bullet.
- `docs/project-local-NW-model-setup.md` §5.2: update the `server.py` sketch's `transcribe()` return statement to also emit `segments` (per the corrected contract above) — this is the one substantive edit to that doc, since it's still just a snippet, not running code.
- `gpu-transcription-plan.md`: short annotation noting Part C step 1's "remove local model load" premise is superseded by flag-based coexistence (local stays default); steps 2 (hotwords) and 4 (WAV write-ahead durability) remain explicitly deferred.

## Implementation steps (continuing from Part 1's Step 10)

**Step 11** — flag + `normalize_remote_url` + mutual-exclusion/invalid-URL `parser.error` checks.
**Step 12** — `RemoteSegment` NamedTuple; extract current inline decode call into `transcribe_chunk_local` (pure refactor).
**Step 13** — `encode_wav_bytes` + `build_multipart_request`.
**Step 14** — `check_remote_health` + `report_remote_health_failure`.
**Step 15** — `transcribe_chunk_remote` (+ `RemoteTranscriptionError`, retry/backoff, segment parsing with safe defaults).
**Step 16** — `transcribe_chunk` dispatcher + `remote_failure_limit_reached`.
**Step 17** — wire dispatcher into `transcribe_loop` (consecutive-failure counter, transcript marker line, escalation to the Step 7 worker-error path).
**Step 18** — `main()`: fail-fast health check before prompts, widened `load_model`/health try/except, settings-summary `Mode` line, `--model`/`--beam-size`-ignored note.
**Step 19** — docs (README, the doc's server.py sketch, gpu-transcription-plan.md annotation).
**Step 20** — extend `selftest.py` with a `_FakeWhisperServer` (same `http.server.HTTPServer` background-thread pattern as `teams-notetaker/selftest.py`'s `_FakeLLMServer`; avoid `cgi.FieldStorage`, removed in Python 3.13) covering every item below.
**Step 21** — full verification pass.

## Verification — testable now vs. only once L1 exists

**Now, offline, via `selftest.py` + `_FakeWhisperServer`:** flag parsing/`--help`/env var; mutual-exclusion and invalid-URL `parser.error` cases; `encode_wav_bytes` round-trip; multipart body shape; `check_remote_health` success/closed-port/non-200/timeout; `transcribe_chunk_remote` happy-path/retry-then-recover/retries-exhausted/malformed-JSON/missing-hallucination-fields; `is_hallucination` parity between `RemoteSegment` and local `FakeSegment`; `remote_failure_limit_reached` boundary; `main()`'s fail-fast-before-prompts ordering; full regression of Part 1's existing `selftest.py` checks; local-mode live smoke test byte-identical to before.

**Only once L1 actually exists (separate, later, owner-run):** whether the real server's response actually matches this contract; real end-to-end accuracy/latency against the <10s budget; real network resilience (cable pulls, server restarts) against the chosen retry/limit constants; real `/health` semantics under partial readiness; whether forwarded `language`/`initial_prompt` behave as intended server-side.
