# Plan: GPU client-server transcription + accuracy/reliability upgrades

## Context
An analysis of 10 recent transcripts found 402 ASR mistakes (~0.66% occurrence rate), dominated by (a) proper nouns / product names (Databricks, Lakebase, ABAC/RBAC, Cavallo, Entra, and people names) and (b) Whisper repetition-loop hallucinations. The goal is to cut that error rate and move inference onto a GPU.

Decisions settled with the user:
- **Keep dual You/Others capture; live-only** (no separate offline archival re-pass).
- **Model:** `large-v3-turbo`, `int8`, `beam_size=5`, CUDA.
- **Architecture:** build the client-server split now, per `project-local-model-setup.md`.
- Bias via **`hotwords`** (context-aware, no hallucination penalty), not `initial_prompt`.

Topology (from `project-local-model-setup.md`): **L1** = ASUS ROG Strix (RTX 4050, 5,920MB VRAM, Windows) runs a faster-whisper + FastAPI server. **L2** = Mac runs capture, POSTs WAV chunks to L1 over a direct wired Ethernet link (`192.168.77.1` <- `.2`), Wi-Fi stays default route for internet.

## Assumptions (labeled; correct me if wrong)
- **L2 = MacBook** (existing `transcribe.py` + BlackHole). If L2 = ThinkPad, the capture layer changes from BlackHole to WASAPI loopback - out of scope for this code.
- Link subnet `192.168.77.0/24` does not overlap house Wi-Fi (verify per setup doc TODO).
- L1's native-Windows CUDA/cuDNN + faster-whisper stack already works (owner's `WINDOWS_NVIDIA.md`).

## Part A - Direct wired link (L1 + L2, owner-run, not code)
Per `project-local-model-setup.md` §4: static IPs, no gateway on the wired iface, firewall-scope ports 8000/11434 to the link subnet. Acceptance: `ping 192.168.77.1` from L2, internet still works, port 8000 unreachable from Wi-Fi hosts.

## Part B - L1 Whisper server (`server.py`, runs on Windows)
Base on setup doc §5.2, with these additions to implement the accuracy decisions:
- Load `large-v3-turbo`, `int8` once at startup (float16 as a benchmarked quality option). Resident across requests.
- **Set `beam_size=5` explicitly** (faster-whisper's own default; the largest
  decoder-side accuracy lever). Correction 2026-09-14: an earlier version of
  this line said beam 5 was *required for `hotwords` to bite*. That is not
  supported by the source — `get_prompt()` prepends hotword tokens
  independent of `beam_size`, and hotwords are disabled only by `prefix`.
  The two are independent levers.
- Accept both `hotwords` and `initial_prompt` form fields; **prefer `hotwords`**. Pass through to `model.transcribe`.
- **Anti-hallucination knobs** (attack the repetition-loop duplication): `vad_filter=True`, `condition_on_previous_text=False`, plus `no_speech_threshold`, `compression_ratio_threshold`, `log_prob_threshold`, and a `temperature` fallback list. Verify exact names against the installed faster-whisper signature during build.
- Return per-segment text with segment start/end offsets so L2 can timestamp precisely (not just one concatenated blob).
- Keep `/health`. Boot-persistence mechanism = owner TODO (NSSM / Task Scheduler / Startup).
- File location: `~/jarvis/services/whisper/server.py` on L1.

## Part C - L2 client (`transcribe.py` refactor, Mac - main code work)

> **Update (2026-09-08):** step 1's "remove local model load" premise is
> superseded. Local and remote inference now coexist behind a `--remote-url`
> CLI flag (local stays the default) — see
> `.claude/plans/transcribe-audit-fixes-plan.md` Part 2 for the actual
> client-side design (flag, HTTP client, response contract, retry/failure
> handling), which has been implemented. Steps 2 (hotwords/glossary) and 4
> (WAV write-ahead durability) below remain explicitly deferred, not built by
> that work. Steps 3, 5, and 6 (callback fix, silence-aware chunking,
> reliability guards) are implemented as part of the same change.

Replace in-process inference with HTTP calls to L1 and fold in the improvements:

1. ~~**Remove local model load** (`load_model` / in-process `WhisperModel`). Add an HTTP client POSTing WAV chunks to `http://192.168.77.1:8000/v1/audio/transcriptions`, timeout ~15s, retry with backoff; on failure keep the chunk's WAV and log loudly (never silently drop). Use stdlib `urllib.request` (no new dep) or `requests`.~~ Superseded — see update note above: local inference stays as the default, remote is opt-in via `--remote-url`, implemented with stdlib `urllib.request`, a 20s per-chunk timeout, and a bounded retry+consecutive-failure-limit policy (no disk write-ahead, since step 4 is deferred).
2. **Hotwords by default** (no flag): at startup read the "Correct term" column from `~/.claude/meeting-notes-glossary.md`, build a compact **core** hotword string (curated client/people/product names, kept under ~200 tokens because faster-whisper truncates to `max_length//2 - 1`). Add an optional startup prompt for **per-meeting nouns** (these must be entered at `transcribe.py` launch, NOT given to the post-hoc meeting-notes skill). Send as `hotwords` on every request. New helper: `hotwords.py`.
3. **Callback fix (AUDIT #1):** the PortAudio callback does only `indata.copy()` + enqueue; move `.mean()` downmix and `.reshape()` into `transcribe_loop`. Prevents dropped audio now that per-chunk latency includes network + GPU.
4. **WAV write-ahead durability (answers the RAM concern):** stream each source to an int16 WAV on disk as captured (~115 MB/hr/stream; ~230 MB/hr for both), via stdlib `wave`. Disk is the durable source; the RAM queue is only the live path. A failed POST still leaves the audio on disk for re-send / re-transcription.
5. **VAD / silence-aware chunking (#6):** cut chunks at silence gaps instead of fixed 10s, so each chunk is a whole utterance - fewer boundary garbles. Reuse `prepare_audio`'s RMS (no new dep).
6. **Reliability guards:** validate `--chunk-seconds >= 1` (AUDIT #2); monitor `worker.is_alive()` and exit loudly on worker death (AUDIT #3); sample-rate guard (AUDIT #5/#8) - query the device's native rate, resample cleanly to 16k or fail with a clear message.
7. **Keep You/Others labels + timestamped lines**; timestamp returned text using the chunk's audio offset (via `format_offset`).

## Files
- **L2 (Mac, editable here):** `/Users/hackerboy/meeting-transcriber/transcribe.py` (client refactor); new `whisper_client.py` + `hotwords.py` helpers.
- **L1 (Windows, write file, runs there):** `server.py` per §5.2 + additions above.
- **Network:** owner-run per setup doc §4.

## Verification
- **Server:** `curl http://192.168.77.1:8000/health`; POST a known 30s WAV, assert correct text and <10s round-trip; `nvidia-smi` confirms GPU-resident, VRAM in budget.
- **L2 end-to-end:** run `transcribe.py` against L1; speak -> `You:` lines; play meeting audio -> `Others:` lines; confirm hotword terms (Databricks, Lakebase, Entra, names) now transcribe correctly.
- **Resilience:** kill Wi-Fi mid-run (link still transcribes); pull the Ethernet cable (WAV still written, loud error, no silent stop).
- **Regression on the errors:** transcribe a fresh meeting (or retained audio), spot-check that the top offenders (RBAC, Databricks, Lakebase, Sergey/Jeril/Lior names, Co-operators) are correct and repetition-loops are reduced vs the analyzed set.
- `python -c "import ast; ast.parse(open('transcribe.py').read())"`.

## Dependency findings (L2 venv checked 2026-08-27)
Installed: `faster-whisper 1.2.1`, `sounddevice 0.5.5`, `numpy 2.4.3`, `av 16.1.0`, Python 3.11.15.
Missing but avoidable without new installs:
- HTTP POST to L1: use stdlib `urllib.request` (multipart) to add zero deps, or `pip install requests` for cleaner code (setup doc uses requests).
- WAV write-ahead: use stdlib `wave` (int16) - no `soundfile` needed.
- Silence-aware chunking: reuse existing `prepare_audio` RMS - no `webrtcvad` needed.
Net: no mandatory new dependency on L2. (L1 server separately needs `fastapi`+`uvicorn` per setup doc.)

## Deferred / out of scope
- Offline `large-v3` archival re-pass (user chose live-only).
- Ollama LLM tier (setup doc §5.4).
- Streaming partials / Speaches (§5.3) unless live captions are needed.
- Standalone glossary find/replace filter (redundant with the meeting-notes skill + hotwords; also carries the common-word-trap risk).

## Related docs
- `project-local-model-setup.md` - L1/L2 network + server setup handoff.
- `transcription-recommendations.md` (this folder) - plain-English explanation of the 9 recommendations behind this plan.
- `AUDIT.md` - the code-audit findings referenced above (#1, #2, #3, #5).
