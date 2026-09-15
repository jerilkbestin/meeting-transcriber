# Jarvis - Local Model Server + Direct Link Setup

**Purpose:** Implementation handoff for Claude Code. Stand up (A) the direct wired link between L1 and L2, and (B) a local Whisper model server on L1 that (C) the transcribing code on L2 calls as its transcription backend.
**Companion docs:** `jarvis-assistant-designs.md` (Design C), `jarvis-agent-workflows-spec.md`, `PROJECT_INSTRUCTIONS.md`.
**Date:** 2026-08-24
**Status:** Decisions settled below; build not started. Placeholders marked `TODO(owner)` must be resolved during build, never invented.

---

## 1. Objective

L1 runs a Whisper model on its GPU and exposes it as a network service. L2 runs the transcribing/agent code and sends audio to L1, receiving text back. Hard requirement: end-to-end transcription latency under 10 seconds per chunk. The network is not the bottleneck; GPU inference is. This document sets up the transport and the server so the existing capture code changes from "load the model in-process" to "call L1 over the wire".

---

## 2. Decision carried in from the design conversation

**Use a direct wired Ethernet link with static IPs. Do not use Wi-Fi as the model transport. Do not use PPP/PPPoE.**

| Option | Verdict | Reason |
|---|---|---|
| House Wi-Fi | Rejected as transport | ~10 devices contend for airtime; jitter and occasional drops during meetings; risk of router AP/client isolation silently blocking L2 to L1. |
| Direct link, static IP | **Chosen** | Point-to-point, no contention, deterministic sub-millisecond latency, no router involvement. This is the Design C topology. |
| PPP / PPPoE over the cable | Rejected | Ethernet already carries IP natively; PPPoE needs a server side that neither macOS nor Windows ships; zero benefit over static IP. |

Bandwidth is a non-issue either way: real-time 16 kHz mono audio is ~256 kbps raw, and the UE300 link delivers ~930 Mbps. The wired link is chosen for latency determinism and isolation, not throughput.

**L2 keeps Wi-Fi active for internet.** The wired interface carries only L1-bound traffic (no default gateway on it); the default route stays on Wi-Fi.

---

## 3. Topology and hardware

```
L2 (agent + capture)                         L1 (GPU inference node)
MacBook Pro M4  OR  ThinkPad X1 Carbon G11   ASUS ROG Strix G614JU
  - runs transcribing code                     - RTX 4050 Laptop, 5,920MB VRAM
  - Wi-Fi ON (internet)                         - i7-13650HX, 32GB RAM, Windows
  - USB port ── UE300 (USB->GbE) ──┐            - built-in Ethernet port
                                    │            - runs Whisper server (+ optional Ollama)
                                    │
        Ethernet cable (Cat5e/Cat6, 1 GbE) ─────┘
```

Two L2 sub-paths (network-identical, ~930 Mbps ceiling either way):
- **ThinkPad:** UE300 USB-A plug -> ThinkPad USB-A port. Simplest, zero extra adapters.
- **MacBook:** UE300 USB-A plug -> passive USB-A-to-USB-C adapter -> Mac USB-C port.

`TODO(owner)`: choose which machine is L2 (the agent host). This picks the sub-path and determines the client OS in Part A and Part C. It does not change L1.

---

## 4. Part A - Direct static-IP link

### 4.1 Addressing plan

Private, non-routed /30 or /24 on both wired interfaces. No gateway, no DNS on this interface.

| Host | Interface | IP | Prefix | Gateway |
|---|---|---|---|---|
| L1 (Strix) | built-in Ethernet | `192.168.77.1` | /24 | none |
| L2 (Mac or ThinkPad) | UE300 USB LAN | `192.168.77.2` | /24 | none |

`TODO(owner)`: **verify the house Wi-Fi subnet first.** The link subnet must not overlap it. If the house LAN is `192.168.1.0/24` or `192.168.0.0/24`, then `192.168.77.0/24` is safe. If the house happens to use `192.168.77.x`, pick another block (for example `10.99.0.0/24`). Overlap will break routing.

### 4.2 L1 - Windows (Strix)

```powershell
# 1. Identify the built-in Ethernet adapter name
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed

# 2. Assign static IP, no gateway (replace "Ethernet" with the real adapter name)
Set-NetIPInterface -InterfaceAlias "Ethernet" -Dhcp Disabled
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.77.1 -PrefixLength 24

# 3. Confirm - no default gateway should be listed for this interface
Get-NetIPConfiguration -InterfaceAlias "Ethernet"
```

### 4.3 L2 - macOS (if L2 = MacBook)

GUI (preferred, allows a truly empty Router field):
System Settings > Network > (USB LAN service) > Details > TCP/IP > Configure IPv4 = Manually, IP `192.168.77.2`, Subnet `255.255.255.0`, **Router = blank**.

Then ensure Wi-Fi outranks the wired service for the default route:
System Settings > Network > (…) > Set Service Order > drag Wi-Fi above the USB LAN service.

CLI note: `networksetup -setmanual` requires a router argument, so the blank-router case is cleaner via the GUI. Verify the default route still points at Wi-Fi:
```bash
route -n get default        # 'interface:' should be en0 (Wi-Fi), not the USB LAN
ping -c 3 192.168.77.1      # L1 reachable over the wire
```

### 4.4 L2 - Windows (if L2 = ThinkPad)

```powershell
Get-NetAdapter | Format-Table Name, InterfaceDescription, Status, LinkSpeed
Set-NetIPInterface -InterfaceAlias "Ethernet 2" -Dhcp Disabled
New-NetIPAddress -InterfaceAlias "Ethernet 2" -IPAddress 192.168.77.2 -PrefixLength 24
# No gateway set -> Wi-Fi keeps the default route. Confirm:
Get-NetRoute -DestinationPrefix 0.0.0.0/0   # NextHop should be the Wi-Fi gateway
```

### 4.5 L1 firewall - scope the model ports to the link only

Ollama and a bare Whisper server have no auth by default, so restrict them to the link subnet.

```powershell
New-NetFirewallRule -DisplayName "Jarvis Whisper (link only)" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress 192.168.77.0/24

New-NetFirewallRule -DisplayName "Jarvis Ollama (link only)" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 11434 -RemoteAddress 192.168.77.0/24
```

### 4.6 Acceptance for Part A

- `ping 192.168.77.1` from L2 succeeds.
- L2 still loads a public website (Wi-Fi default route intact).
- From a non-link machine on Wi-Fi, port 8000 on L1 is NOT reachable (firewall scoping works).

---

## 5. Part B - Local Whisper server on L1

### 5.1 Model choice (settled)

`large-v3-turbo`, `int8`, CUDA. Per the SYSTRAN benchmark this peaks around 1.5GB VRAM and posted the lowest WER of the tested set, leaving headroom on the 6GB card for a co-resident LLM. `float16` is available as a quality option if VRAM allows; benchmark both.

### 5.2 Recommended server: native Windows faster-whisper + FastAPI (primary)

Rationale: L1 already has a working native-Windows CUDA faster-whisper stack (documented in the owner's `WINDOWS_NVIDIA.md`). Wrapping that in a thin FastAPI service avoids introducing Docker Desktop + WSL2 GPU passthrough on the Strix. Lowest friction, reuses proven CUDA/cuDNN setup.

```
# L1 install (native Windows, CUDA already set up per WINDOWS_NVIDIA.md)
pip install faster-whisper fastapi "uvicorn[standard]" python-multipart
```

`server.py` (place at `~/jarvis/services/whisper/server.py`):

```python
# Minimal faster-whisper HTTP server for L1 (Strix, Windows + CUDA).
# Run: uvicorn server:app --host 0.0.0.0 --port 8000
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from faster_whisper import WhisperModel

MODEL_SIZE   = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE       = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")   # int8 ~1.5GB VRAM; float16 = higher quality

# Load once at process start; the model stays resident across requests.
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title="Jarvis local Whisper server")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE, "compute_type": COMPUTE_TYPE}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    initial_prompt: str = Form(default=""),
    language: str = Form(default=None),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        segments, info = model.transcribe(
            tmp_path,
            initial_prompt=initial_prompt or None,  # glossary feedback loop (D-13)
            language=language,
            vad_filter=True,                          # skip silence, reduce hallucination
        )
        segments = list(segments)
        text = "".join(seg.text for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
            # Per-segment breakdown, added 2026-09-08: the L2 client's
            # hallucination filter (no_speech_prob > 0.85 and avg_logprob <
            # -0.9) needs these two fields per segment to work identically
            # whether inference ran locally or remotely. faster-whisper
            # already computes both on every Segment, so this is free.
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "no_speech_prob": seg.no_speech_prob,
                    "avg_logprob": seg.avg_logprob,
                }
                for seg in segments
            ],
        }
    finally:
        os.unlink(tmp_path)
```

Endpoint path `/v1/audio/transcriptions` is chosen to mirror the OpenAI shape so the client can later swap to any OpenAI-compatible server (5.3) with minimal change. The flat `text`/`language`/`duration` fields are kept for compatibility/debugging; the L2 client (`transcribe_chunk_remote` in `transcribe.py`) parses `segments`, not `text`, and defaults missing `no_speech_prob`/`avg_logprob` toward "not a hallucination" if an alternate server implementation omits them.

`TODO(owner/build)`: run the server on boot. Options: a Startup shortcut running `uvicorn`, NSSM as a Windows service, or Task Scheduler at logon. Decide during build.

### 5.3 Alternative server: Speaches (batteries-included, Docker)

If OpenAI-compatible streaming (SSE) and a Realtime endpoint are wanted out of the box, `speaches` (formerly `faster-whisper-server`) provides exactly that on the faster-whisper backend. Cost: it runs as a Linux container, so on the Windows Strix it needs Docker Desktop + WSL2 + the NVIDIA Container Toolkit for GPU passthrough. Prefer 5.2 unless streaming partial transcripts is required.
- Project: https://github.com/speaches-ai/speaches  (docs: https://speaches.ai)
- Alternative Docker image with the same OpenAI-compatible API: https://github.com/hwdsl2/docker-whisper

### 5.4 Companion service: Ollama LLM tier (optional for the transcription MVP)

Transcription does not need the LLM. The LLM tier (cleanup, first-pass summary) is separable and can come after. When added, expose Ollama on the link only:

```powershell
# Expose Ollama on the LAN interface (default binds to localhost only)
setx OLLAMA_HOST "0.0.0.0:11434"   # restart Ollama after setting
# Firewall rule from 4.5 already scopes 11434 to the link subnet.
```

---

## 6. Part C - Transcribing code integration (L2 client)

**Update (2026-09-08): this section is no longer a proposal — it describes
the actual client, already implemented in `transcribe.py` on this machine.**
The capture loop stayed on L2 as planned (mic + BlackHole loopback capture,
adaptive silence-aware chunking); what changed from the original sketch
below: the client is behind an opt-in `--remote-url` flag (local inference
stays the default, not fully replaced), uses stdlib `urllib` instead of
`requests` (zero new dependency for an off-by-default feature), sends the
already-in-memory chunk as WAV bytes (no file path/round-trip through disk),
and parses a `segments` array — not a flat `.text` string — because the
hallucination filter needs each segment's `no_speech_prob`/`avg_logprob`
(see the response-contract note added to §5.2's `server.py` above).

Actual client code (`transcribe.py`, stdlib only — `urllib.request`, `json`,
`io`, `wave`, `uuid`):

```python
REMOTE_HEALTH_PATH = "/health"
REMOTE_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
REMOTE_MAX_RETRIES = 2                        # 3 attempts total
REMOTE_RETRY_BACKOFF_SECONDS = (1.0, 2.0)
REMOTE_CHUNK_TIMEOUT_SECONDS = 20.0           # headroom above the <10s budget

def check_remote_health(base_url):
    # GET /health at startup, before any device/mic prompts — fail fast on
    # a dead server instead of discovering it mid-session.
    ...

def transcribe_chunk_remote(audio, language, initial_prompt, base_url):
    wav_bytes = encode_wav_bytes(audio)  # in-memory float32 -> 16-bit PCM WAV
    fields = {"language": language, "initial_prompt": initial_prompt}
    for attempt in range(REMOTE_MAX_RETRIES + 1):
        if attempt > 0:
            time.sleep(REMOTE_RETRY_BACKOFF_SECONDS[attempt - 1])
        try:
            request = build_multipart_request(base_url + REMOTE_TRANSCRIBE_PATH, wav_bytes, fields)
            with urllib.request.urlopen(request, timeout=REMOTE_CHUNK_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_remote_segments(payload)  # -> list[RemoteSegment], reads `segments`
        except Exception as exc:
            last_exc = exc
    raise RemoteTranscriptionError(f"remote transcription failed after 3 attempts: {last_exc}")
```

No glossary/hotwords hook exists yet (Decision 9 / D-13 is still deferred,
not built) — only the plain `--initial-prompt` flag plus a per-source
carry-forward prompt (the tail of the previous chunk's own text) are
forwarded as `initial_prompt`. A `/meeting-notes` glossary lookup would slot
into that same `initial_prompt` field later without changing this contract.

A failed chunk retries twice (see constants above) then is dropped with a
visible `[transcription dropped — remote server error]` line in the
transcript — never silently missing. 5 consecutive failures end the session
loudly rather than retrying forever against a genuinely dead server.

`TODO(owner/build)`: confirmed — discrete chunked files (not streaming). The
client already sends whole in-memory WAV chunks per adaptive chunk boundary
(6-20s, cut on silence); streaming/5.3 Speaches is unneeded unless live
captions become a requirement.

---

## 7. End-to-end acceptance tests

1. `curl http://192.168.77.1:8000/health` from L2 returns the model info JSON.
2. POST a known 30-second WAV; assert returned text is correct and round-trip wall-clock is under 10 s.
3. Repeat under a simulated meeting (both audio streams active on L2) to confirm capture + network + inference still fits the budget.
4. Kill Wi-Fi on L2 mid-test: transcription still works over the wire (proves the link is the transport, not Wi-Fi). Restore Wi-Fi after.
5. `nvidia-smi` on L1 during a run: confirm the model is GPU-resident and VRAM is within budget (no spill to system RAM).

---

## 8. Suggested build order

1. Part A link (4.1-4.6) and prove `ping` + internet coexistence.
2. Part B 5.2 server + `/health` reachable from L2 over the link.
3. Part C client swap; run acceptance tests 1-3.
4. Benchmark int8 vs float16 for the latency/quality trade-off; lock the choice.
5. Boot-persistence for the server (5.2 TODO) + firewall verification (test 4.6/AP scoping).
6. Optional: Ollama tier (5.4) and, only if streaming is required, migrate to Speaches (5.3).

---

## 9. Function-by-function breakdown (server.py)

- **Module-level model load** (`model = WhisperModel(...)`): loads weights into VRAM once at process start so every request reuses the resident model. Reloading per request would add seconds and defeat the latency budget.
- **`health()`**: liveness/readiness probe. Lets the L2 client and acceptance tests confirm the server is up and which model/precision is active without sending audio.
- **`transcribe(file, initial_prompt, language)`**: the inference endpoint. Persists the uploaded audio to a temp file (faster-whisper reads from a path), runs `model.transcribe` with VAD filtering and the optional glossary prompt, concatenates segment text, and returns JSON. The `finally` block deletes the temp file so repeated calls do not leak disk.

## 10. Software engineering concepts involved

- **Client-server inference split (remote model serving):** capture stays where the audio is (L2); heavy compute runs where the GPU is (L1). Standard pattern for putting a GPU node behind a thin API.
- **Warm model / process-resident state:** amortize the expensive load once, serve many requests cheaply.
- **OpenAI-compatible interface as an abstraction seam:** mirroring `/v1/audio/transcriptions` lets the backend (native FastAPI vs Speaches) change without touching the client.
- **Least-privilege network exposure:** static private subnet + firewall scoping + no default gateway on the link, because Ollama/Whisper ship without auth.
- **Split-horizon routing:** two active interfaces on L2 (Wi-Fi for internet, wired for L1) with route priority controlled by service order / absent gateway.

---

## 11. Evidence and open questions

### Source-backed facts
- UE300 is USB 3.0 to Gigabit (RTL8153), ~930 Mbps real-world local throughput - either L2 sub-path is far above the workload need. https://static.tp-link.com/res/down/doc/UE300_V1_Datasheet.pdf
- `speaches` (formerly faster-whisper-server) is an OpenAI-API-compatible STT/TTS server on the faster-whisper backend, with SSE streaming. https://github.com/speaches-ai/speaches
- A 6GB VRAM GPU runs Whisper large-v3 comfortably via faster-whisper; turbo/int8 is lighter still. https://runaihome.com/blog/whisper-large-v3-self-hosted-transcription-server-2026/
- faster-whisper streaming latency lands in the ~500-800 ms range, well inside the 10 s budget. https://localaimaster.com/blog/faster-whisper-guide
- L1 hardware: RTX 4050 Laptop, 5,920MB VRAM, i7-13650HX, 32GB RAM (owner dxdiag).

### Inferences
- The direct link beats Wi-Fi here on jitter/reliability and sidesteps router AP-isolation, not on bandwidth. (From the workload's tiny data rate vs. contended-Wi-Fi behavior.)
- Native Windows faster-whisper + FastAPI is lower-friction than Dockerized Speaches on the Strix, because the owner's CUDA stack already works natively on Windows. (From the existing `WINDOWS_NVIDIA.md` path; Docker/WSL2 GPU passthrough would be net-new.)
- Chunked-file transcription meets the 10 s budget; streaming is only needed for live captions. (From the latency figures above vs. the requirement.)

### Unknowns (verify during build)
- **Which L2 machine** hosts the agent (Mac vs ThinkPad) - `TODO(owner)`. Sets the sub-path and client OS.
- **House Wi-Fi subnet** - must not overlap the chosen link subnet. Check before assigning `192.168.77.0/24`.
- **int8 vs float16** on the 4050 for the latency/quality trade-off - benchmark on real meeting audio.
- **CUDA/cuDNN version pinning** on the Strix for the current faster-whisper release - follow the owner's `WINDOWS_NVIDIA.md`; versions drift.
- **Streaming vs chunked** requirement - decides Part B 5.2 vs 5.3.
- **Server boot-persistence mechanism** on Windows (Startup shortcut vs NSSM vs Task Scheduler).
- **Does L2's OS keep the default route on Wi-Fi** once the wired interface is up - confirm with `route -n get default` (macOS) / `Get-NetRoute` (Windows).

### Source links
- UE300 datasheet: https://static.tp-link.com/res/down/doc/UE300_V1_Datasheet.pdf
- speaches (OpenAI-compatible faster-whisper server): https://github.com/speaches-ai/speaches
- hwdsl2/docker-whisper (alternative OpenAI-compatible image): https://github.com/hwdsl2/docker-whisper
- Self-hosted Whisper large-v3 server guide (2026): https://runaihome.com/blog/whisper-large-v3-self-hosted-transcription-server-2026/
- faster-whisper setup/streaming guide (2026): https://localaimaster.com/blog/faster-whisper-guide
