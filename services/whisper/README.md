# L1 Whisper service

Implements Part B of [the direct-link setup](../../docs/project-local-NW-model-setup.md).
Run natively on Windows, with one CUDA model shared by all requests.

## Deployment status — 2026-09-17

**L1 is configured and L2's HTTP health check passed; speech acceptance from L2 is incomplete.**

- Native Windows Python 3.12 environment: `.venv-whisper` (ignored by Git).
  Installed faster-whisper 1.2.1, CTranslate2 4.8.2, FastAPI 0.141.1,
  Uvicorn 0.53.0, and python-multipart 0.0.32.
- L1 Ethernet: `192.168.77.1/24`; Wi-Fi remains the default route through
  `192.168.1.1`. The firewall rule is restricted to interface `Ethernet`, local
  `192.168.77.1`, peer `192.168.77.2`, and TCP port 8000.
- Task `Jarvis-Whisper` is registered and was started successfully as user `jeril`
  with limited privileges. The listener is only `192.168.77.1:8000`.
- Restart check on 2026-09-17: Windows booted at 00:14:56 (-07:00); the task's
  last run began at 00:17:29 and its state was Running. Without manually
  starting the task during this check, `/health` returned HTTP 200, and a new
  local request transcribed the synthetic 30-second WAV correctly in **4.59 s**.
  This supports successful startup after restart/login; the request originated
  on L1. Raw response: `outputs/whisper-validation/restart-check.json`.
- The post-restart active power plan reports plugged-in idle sleep disabled
  (`STANDBYIDLE = 0`) and battery idle sleep at 180 seconds. Ethernet remained
  `192.168.77.1/24` and the default route remained on Wi-Fi.
- A request **from L1 to its own wired address** returned the correct synthetic
  transcript in **1.06 seconds** under the scheduled task. This does not traverse
  the cable or establish L2 connectivity.
- L2 retest at 21:06 on 2026-09-16 (America/Los_Angeles): `192.168.77.2`
  answered all three pings from L1, with 0% loss and 2–3 ms round-trip times.
  L1's Ethernet reports Up at 1 Gbps. The earlier unreachable result is resolved.
  L1's `/health` still returns HTTP 200 with `large-v3-turbo` / CUDA / int8,
  and task `Jarvis-Whisper` is Running.
- On 2026-09-17 the owner supplied the result of running the health request on
  L2: `{"status":"ok","model":"large-v3-turbo","device":"cuda","compute_type":"int8"}`.
  This passes the L2-to-L1 HTTP health acceptance check (owner-reported evidence).
  It does not establish speech inference, latency, or capture behavior on L2.

Local HTTP comparison using the same 30-second mono 16 kHz WAV (23.6 seconds
of Windows-synthesized speech followed by silence), language `en`, beam size 5:

| Precision | First request | Repeat requests | GPU memory after inference |
|---|---:|---:|---:|
| int8 | 3.50 s | 0.57 / 0.54 s | 1,187 MiB |
| float16 | 2.22 s | 0.60 / 0.60 s | 2,243 MiB |

All six returned transcripts exactly matched the synthetic reference. These are
individual observations, not peak-memory measurements, representative quality
scores, or L2 latency measurements. Keep int8 as the default based on its lower
observed memory use and lack of a quality difference on this sample; reassess
with real meeting audio. Raw results are local ignored artifacts in
`outputs/whisper-validation/` (`results-int8.json`, `results-float16.json`, and
`deployment-check.json`). Speech was generated with
[Windows SpeechSynthesizer](https://learn.microsoft.com/en-us/dotnet/api/system.speech.synthesis.speechsynthesizer.setoutputtowavefile).

## Install and run

From the repository root in PowerShell, using Python 3.12 and the CUDA 12 /
cuDNN 9 DLL setup in [WINDOWS_NVIDIA.md](../../WINDOWS_NVIDIA.md):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
py -3.12 -m venv .venv-whisper
.\.venv-whisper\Scripts\python.exe -m pip install -r services\whisper\requirements.txt
.\services\whisper\start-server.ps1 -Python "$PWD\.venv-whisper\Scripts\python.exe" -CudaDllDirectory "$env:USERPROFILE\faster-whisper-cuda-dlls"
```

The launcher uses `large-v3-turbo`, CUDA, int8, one worker, and the dedicated
`192.168.77.1:8000` address. The first launch downloads the model. For a local
smoke test before configuring Ethernet, add `-BindAddress 127.0.0.1`.
To compare precision, stop the current process and rerun with `-ComputeType float16`.
Keep the other settings and test audio identical.

## Wired network

This guide retains both macOS and Windows/PowerShell instructions for L2;
it does not assign L2 a fixed operating system.

On the verified Strix, built-in `Ethernet` is the Realtek adapter; the owner
confirmed it connects to L2. Wi-Fi was observed at `192.168.1.75/24`, using
gateway `192.168.1.1`. Check again if the network changes.

In an **Administrator PowerShell**:

```powershell
.\services\whisper\configure-l1-network.ps1
```

This assigns L1 `192.168.77.1/24` without a gateway and creates an inbound TCP
8000 rule limited to that interface/local address and peer `192.168.77.2`.
It stops if the adapter identity, existing address, or default route conflicts,
and refuses to overwrite an existing rule. It does not configure L2.
Verify the firewall rule and the server's listening address after setup;
an allow rule alone does not override other pre-existing broad allow rules.

Configure L2's direct Ethernet as `192.168.77.2/24`, with gateway and DNS empty,
after checking L2's other networks for overlap. Keep its Wi-Fi internet route.
From L2:

```bash
curl http://192.168.77.1:8000/health
curl --fail --show-error -w '\nHTTP %{http_code}; elapsed %{time_total}s\n' -F file=@known-30s.wav -F language=en http://192.168.77.1:8000/v1/audio/transcriptions
```

On Windows use `curl.exe` to avoid the PowerShell alias. Supply an actual known
30-second speech WAV. Compare returned text with the reference, and measure both
the first inference and repeated requests. A silence-only test does not measure
speech inference speed. Repeat during two-stream capture and with L2 Wi-Fi
temporarily disabled; separately verify internet still works with Wi-Fi enabled.

The existing Mac capture client runs with:

```bash
python transcribe.py --remote-url http://192.168.77.1:8000
```

`transcribe_windows.py` does **not** yet implement `--remote-url`; Windows live
capture requires that additional integration. The PowerShell HTTP tests above
can still be used to verify the server from Windows.

## Start automatically at logon

After validating manual startup, register the task once:

```powershell
.\services\whisper\register-logon-task.ps1 -Python "$PWD\.venv-whisper\Scripts\python.exe" -CudaDllDirectory "$env:USERPROFILE\faster-whisper-cuda-dlls"
Start-ScheduledTask -TaskName Jarvis-Whisper
Get-ScheduledTask -TaskName Jarvis-Whisper
```

Stop the manually launched server before starting the task. It uses the same
launcher with the dedicated address and int8 defaults. It runs as your normal
interactive user when you log on, with up to three one-minute-spaced restarts on
failure. This is logon persistence, not a service available before login. It does
not prevent Windows from sleeping. To disable future launches, use
`Disable-ScheduledTask -TaskName Jarvis-Whisper`; to stop a running task use
`Stop-ScheduledTask -TaskName Jarvis-Whisper`.

## Tests

```powershell
.\.venv-whisper\Scripts\python.exe -m pip install httpx
.\.venv-whisper\Scripts\python.exe -m unittest discover -s services/whisper -p test_server.py -v
```

Eight HTTP/lifecycle tests passed on native Windows. They use a fake inference
model and therefore do not establish CUDA inference, accuracy, latency, or LAN
reachability. The installed Starlette version emits an httpx deprecation warning;
the test run still passes.

## Software engineering concepts involved

- **Resource lifecycle:** model loading happens once before the service becomes
  ready; startup failure prevents a healthy endpoint from being published.
- **Concurrency and backpressure:** blocking inference runs in a FastAPI worker
  thread. A lock permits one GPU request; concurrent requests receive HTTP 503
  instead of accumulating a GPU queue. The existing client retries failures.
  This bounds simultaneous inference, but several independent clients may exhaust
  their retries during a long inference. It is intended for one capture host.
- **API contract and validation:** multipart fields are validated before inference;
  returned confidence fields preserve the existing client's hallucination filter.
- **Resource ownership and error handling:** FastAPI owns the spooled upload;
  `finally` releases the GPU lock even when lazy inference fails. Logs preserve
  diagnostics while HTTP errors avoid exposing local exception details.
- **Network isolation:** bind to the dedicated address and constrain the firewall
  to the peer. This service has no application authentication.

## Function and block breakdown

| Code | What it does and why |
|---|---|
| `server.lifespan` | Loads environment-selected model/device/precision before readiness, creates the lock, and releases the model at shutdown; keeps expensive process state out of module import and per-request work. |
| `server.health` | Returns loaded configuration without using the GPU lock; keeps readiness checks responsive during inference. |
| `server.transcribe`: validation | Rejects unsupported language codes, empty audio, and invalid beam sizes before spending GPU time. |
| `server.transcribe`: admission | Acquires a nonblocking lock or returns retryable 503; avoids accumulating GPU requests. |
| `server.transcribe`: inference/serialization | Passes the existing file stream to Whisper, forwards prompt/language/beam size, enables VAD, consumes lazy segments under the lock, and returns the client contract. |
| `server.transcribe`: errors/cleanup | Maps decoder errors to 400 and inference failures to 500, logs elapsed time, and always unlocks; a failed request cannot permanently block the server. |
| `start-server.ps1`: validation/environment | Checks Python and optional cuBLAS location and sets model/device/precision explicitly; avoids using an unintended Python on PATH. |
| `start-server.ps1`: launch | Starts one Uvicorn worker on the selected address and propagates its exit code; avoids duplicate model copies. |
| `configure-l1-network.ps1`: preflight | Verifies administrator rights, adapter identity, existing IPs, and absence of a wired default route before modification. |
| `configure-l1-network.ps1`: configuration | Adds the fixed L1 address and narrowly scoped firewall rule, then prints configuration/routes for verification. |
| `register-logon-task.ps1`: preflight/action | Refuses to replace an existing task, resolves executable paths, and builds a quoted launcher action so startup does not depend on the current directory or Python on PATH. |
| `register-logon-task.ps1`: trigger/settings/registration | Registers a normal-user logon trigger, one running instance, no execution time limit, and bounded restart attempts; provides persistence without storing a password. |
| `ServerTests.setUp` / `post` | Inject a fake model through the real lifespan and send real multipart HTTP requests; test the API boundary without downloading weights. |
| `test_readiness_and_single_model_load` | Checks repeated health calls share one model load. |
| `test_multipart_contract_and_upload_cleanup` | Checks all returned fields, forwarded decode options, and closed upload ownership. |
| `test_optional_fields_and_silence` | Checks omitted hints and an empty segment list remain valid. |
| `test_validation_before_inference` | Checks invalid requests never call the model. |
| `test_bad_audio_releases_lock` | Checks a decoder error does not prevent the next request. |
| `test_lazy_inference_failure_releases_lock` / `failed_segments` | Makes iteration fail after one segment to test the actual lazy failure path, sanitized errors, and lock recovery. |
| `test_health_and_busy_response_during_inference` / `slow_segments` | Uses events to hold inference in flight and checks health plus a second request; avoids timing-dependent sleep assertions. |
| `test_startup_failure_is_not_healthy` | Checks failed model initialization prevents startup. |

## Why this logic / API / library / pattern was used

FastAPI matches the proposed server and the existing client's multipart contract.
Its lifespan supports shared model state, normal `def` endpoints run in a thread
pool, and `UploadFile.file` is already a file-like object accepted by Whisper.
This removes the sketch's redundant temporary-file copy and keeps GPU work off
the asynchronous event loop. Uvicorn uses one worker to keep one model resident.

### Source-backed implementation facts

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) documents loading a shared model before requests.
- [FastAPI concurrency](https://fastapi.tiangolo.com/async/#path-operation-functions) documents thread-pool execution of `def` handlers.
- [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile) documents the spooled file and its ownership.
- [faster-whisper source](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py) defines file-like input, supported languages, decode options, and segment fields.
- [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#gpu) describe CUDA 12/cuDNN 9 requirements; published benchmarks do not establish this machine's turbo latency or VRAM.
- [Uvicorn settings](https://github.com/encode/uvicorn/blob/master/docs/settings.md) document host binding and worker count.
- [New-NetIPAddress](https://learn.microsoft.com/en-us/powershell/module/nettcpip/new-netipaddress) documents automatic DHCP disablement when adding an address.
- [New-NetFirewallRule](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule) documents interface/address/port scoping.
- [Task actions](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskaction), [logon triggers](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger), [principals](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskprincipal), and [settings](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasksettingsset) document the registered startup configuration.

### Implementation inferences

Serial GPU work should reduce simultaneous memory pressure. Binding only the
direct-link address should prevent access via the Wi-Fi address. Both need live
acceptance checks; passing unit tests alone does not prove either deployment claim.

### Unknowns

Speech transcription originating on L2, two-stream latency, meeting-audio quality,
precision trade-off on real meetings, and firewall isolation from a second machine
remain to be verified. Restart/login recovery passed the observed health and local
speech checks on 2026-09-17. See the setup document's acceptance checklist.
