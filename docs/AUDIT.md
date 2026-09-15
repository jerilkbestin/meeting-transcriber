# transcribe.py — Code Audit (deferred findings)

Findings from a correctness / edge-case review of `transcribe.py`. **All 6
are now fixed** (see per-finding "Fixed" notes below) — this file is kept as
the historical record of the review. Line numbers refer to the pre-fix
commented version of `transcribe.py`; treat them as approximate anchors, not
exact.

Findings that describe external behavior are backed by upstream docs; those
tied only to reading this repo's code are marked as such.

---

## 1. Real-time audio callback violates PortAudio constraints

**Where:** `make_callback` → `callback` (the body that runs per audio block).

**What:** The callback allocates (`indata.astype(..., copy=True)`, `.mean(axis=1)`,
`.reshape(-1)`) and acquires a lock (`audio_queue.put(...)`) — all inside the
PortAudio real-time audio thread. The sounddevice docs state a stream callback
must not allocate memory or call functions that may block.

**Risk:** Under CPU pressure this can cause input overflow / dropped audio
(audible gaps in the transcript). Not measured on this hardware — latent risk,
not a confirmed live failure.

**Possible fix:** Do the copy only (cheap) in the callback; move downmix/reshape
into `transcribe_loop`. The queue put is hard to avoid but is short.

**Evidence:** External — <https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html>.

**Fixed** — see `make_callback` (does only `indata.copy()` + enqueue) and
`transcribe_loop` (downmix/resample now happen there, off the real-time
thread).

---

## 2. `--chunk-seconds 0` (or negative) hangs a worker thread

**Where:** `chunk_size = SAMPLE_RATE * args.chunk_seconds`; the
`while len(buffers[source]) >= chunk_size` inner loop.

**What:** With `chunk_seconds=0`, `chunk_size=0`, so `len(buf) >= 0` is always
true even on an empty buffer; the slice `buf[:0]` is empty and `buf[0:]` is a
no-op, so the buffer never shrinks. The inner loop spins forever, pinning the
worker thread's CPU. `parse_args` does no range validation.

**Risk:** Trivial to trigger by mistake; silent CPU spin, no transcript output.

**Possible fix:** Validate `chunk_seconds >= 1` in `parse_args`.

**Evidence:** Direct read of this repo's code; reproducible with
`python transcribe.py --chunk-seconds 0`.

**Fixed** — `--chunk-seconds` was replaced by `--min-chunk-seconds`/
`--max-chunk-seconds`, both validated by `positive_float` (rejects <= 0
before any device/model work); a `parser.error` also rejects
`min > max`.

---

## 3. Silent worker-thread death

**Where:** worker thread start in `main`; `sd.sleep(1000)` loop.

**What:** `transcribe_loop` is the only consumer of the queue. If it raises
anywhere (a Whisper internal error, a `KeyError`, etc.), the daemon thread dies
and `threading.excepthook` prints a traceback to stderr — but `main`'s
`sd.sleep(1000)` loop keeps running forever. Audio keeps queuing, transcript
output silently stops, and there is no user-facing signal beyond an
easy-to-miss stderr traceback.

**Risk:** A meeting appears to be transcribing (mic light, app running) but
nothing is being written after the crash point.

**Possible fix:** Wrap the loop body in try/except that logs and re-raises to a
visible failure, or have `main` monitor `worker.is_alive()` and exit loudly.

**Evidence:** Direct read of this repo's code.

**Fixed** — `transcribe_loop`'s body is wrapped in try/except (prints a full
traceback, sets a `threading.Event` named `worker_error`); `main()`'s poll
loop checks `worker_error.is_set() or not worker.is_alive()` every second and
exits loudly with the partial-transcript path if either is true.

---

## 4. `buffers` dict coupled to literal source labels

**Where:** `buffers = {"You": ..., "Others": ...}` in `transcribe_loop`, versus
the `make_callback("You", ...)` / `make_callback("Others", ...)` call sites in
`main`.

**What:** The consumer's dict keys and the producers' `source` labels must match
exactly, but nothing enforces the contract. A future caller passing a different
label raises `KeyError` inside the worker thread — which then trips finding #3
(silent death).

**Possible fix:** Derive buffers lazily (`buffers.setdefault(source, empty)`),
or centralize the label constants.

**Evidence:** Direct read of this repo's code.

**Fixed** — `SOURCE_YOU`/`SOURCE_OTHERS` constants used at every call site;
`SourceBuffer` instances are created lazily via
`buffers.setdefault(source, SourceBuffer())`, so an unrecognized label can
never raise `KeyError`.

---

## 5. No sample-rate mismatch guard

**Where:** `sd.InputStream(samplerate=SAMPLE_RATE, ...)` for both streams.

**What:** The code requests 16 kHz from the OS. If a selected device can't
natively provide 16 kHz, the outcome depends on CoreAudio/PortAudio resampling
behavior, which the script neither controls nor validates.

**Risk:** Possible quality degradation or stream-open failure on odd devices.
Not verified against a real mismatched device on this machine.

**Possible fix:** Query the device's default sample rate and resample
explicitly, or validate/`try` the stream open and report clearly.

**Evidence:** Direct read of this repo's code; runtime behavior unverified.

**Fixed** — `device_sample_rate()` reads each device's own
`default_samplerate` (confirmed field name via
<https://python-sounddevice.readthedocs.io/en/0.5.1/api/checking-hardware.html>);
streams are opened at that native rate; `resample_to_target()` converts down
to 16 kHz via `soxr` (confirmed signature via
<https://python-soxr.readthedocs.io/en/latest/soxr.html>), falling back to a
manual `np.interp` linear resample if `soxr` isn't installed.

---

## 6. Broad `except Exception` in `main` hides tracebacks

**Where:** the outer `except Exception as exc: print(f"\nError: {exc}")` in
`main`.

**What:** This catch-all reduces setup errors (missing BlackHole `RuntimeError`,
model-load failures, `InputStream` construction errors) to a single one-line
message with no traceback.

**Risk:** Friendlier for expected failures, but makes unexpected bugs much
harder to diagnose.

**Possible fix:** Catch the specific expected exceptions with friendly
messages; let unexpected ones surface a full traceback (or log it).

**Evidence:** Direct read of this repo's code.

**Fixed** — the blanket catch-all is gone. `main()` now catches only the
specific known exceptions narrowly (`RuntimeError` around
`find_blackhole_input`, `FileNotFoundError` around the file-mode path check,
`sd.PortAudioError` around stream opening) with a friendly one-line message
each; model-load failures route through `report_model_load_failure` (import
error / network / bad path / generic, each with distinct guidance). Anything
else propagates with a full traceback instead of being swallowed.
