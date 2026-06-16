# Windows 11 + NVIDIA Setup

This is the Windows equivalent of the macOS/BlackHole project path.

Windows does not need BlackHole or a Multi-Output Device. The project captures:

```text
Microphone input        -> transcribe_windows.py -> You
Speaker WASAPI loopback -> transcribe_windows.py -> Others/System audio
```

The speaker loopback captures whatever is playing through the selected Windows playback device. That can be Microsoft Teams, Zoom, a browser tab, a media player, system audio, or any other app. You still hear the audio normally, and the script records a private local copy of that playback stream for transcription.

## Requirements

- Windows 11
- NVIDIA GPU with a working NVIDIA driver
- Python 3.11 or 3.12. Avoid Python 3.14 for this project until the audio and ML packages officially support it well.
- Any meeting or audio app you want to transcribe

Install CUDA/cuDNN support as required by `faster-whisper`/`ctranslate2` for your Python environment. If CUDA is not available, the script can still run on CPU with `--device cpu --compute-type int8`, but that is not the intended NVIDIA path.

## Install

From PowerShell:

```powershell
cd C:\Users\jeril\meeting-transcriber
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

If PowerShell blocks `Activate.ps1` with an execution policy error, use a process-only bypass for the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Or skip activation entirely and call the venv Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\venv\Scripts\python.exe test_audio_windows.py
```

If `py` is not recognized, install Python first:

```powershell
winget install --id Python.Python.3.11
```

Then close and reopen PowerShell, return to the repo, and try:

```powershell
cd C:\Users\jeril\meeting-transcriber
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

If `python` opens the Microsoft Store or is still not recognized, disable the Windows Python aliases in **Settings -> Apps -> Advanced app settings -> App execution aliases**, then reopen PowerShell.

## Validate Audio

```powershell
python test_audio_windows.py
```

Use option `1` to confirm your microphone level moves.

Use option `2` to confirm the speaker loopback level moves while meeting audio, browser audio, or any other audio is playing through that same Windows playback device.

Use option `3` to confirm the selected output device plays sound.

## Playback Setup

For any app you want to transcribe:

1. Choose the Windows playback device you want to capture, such as headphones, speakers, or a monitor output.
2. In the meeting/audio app, set **Speaker** or **Output** to that playback device.
3. In `transcribe_windows.py`, select the loopback device that matches that same playback device.

If an app uses the Windows default output, changing the Windows default playback device is enough. If you change the app output or Windows default output after starting transcription, restart the script and select the matching loopback device again.

## Run

```powershell
python transcribe_windows.py
```

The script asks for:

1. Your microphone input device.
2. The speaker loopback input device for playback audio.
3. A transcript file name prefix.

By default it runs:

```text
model        : small.en
device       : cuda
compute type : float16
chunk length : 10 seconds
```

Useful examples:

```powershell
python transcribe_windows.py --model medium.en
python transcribe_windows.py --model small.en --chunk-seconds 5
python transcribe_windows.py --playback-label System
python transcribe_windows.py --device cpu --compute-type int8
```

## CUDA Troubleshooting

If transcription fails with an error like:

```text
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

the audio capture is working, but the NVIDIA CUDA runtime is not available to `faster-whisper`.

To keep testing transcription immediately, run on CPU:

```powershell
python transcribe_windows.py --device cpu --compute-type int8
```

For GPU transcription, install the CUDA 12 runtime/toolkit and cuDNN version expected by your installed `ctranslate2` package, then reopen PowerShell so the CUDA `bin` directory is on `PATH`.

## Python 3.12 Setup

If your venv was created with Python 3.14, install Python 3.12 side-by-side and create a fresh venv. You do not need to uninstall Python 3.14.

```powershell
winget install --id Python.Python.3.12 -e
```

Close and reopen PowerShell, then run:

```powershell
cd C:\Users\jeril\meeting-transcriber

Rename-Item venv venv-py314
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

If `Rename-Item venv venv-py314` says the target already exists, choose another backup name or delete the old venv after confirming you do not need it.

## CUDA DLL Setup

The current `faster-whisper` GPU path uses `ctranslate2`, which expects CUDA 12 cuBLAS and cuDNN 9 DLLs. One PowerShell-friendly option is to download the CUDA 12/cuDNN 9 DLL archive referenced by the faster-whisper project and add it to your user `PATH`.

```powershell
cd C:\Users\jeril\meeting-transcriber

$dllDir = "$env:USERPROFILE\faster-whisper-cuda-dlls"
New-Item -ItemType Directory -Force $dllDir

$release = Invoke-RestMethod "https://api.github.com/repos/Purfview/whisper-standalone-win/releases/tags/libs"
$release.assets | Select-Object name, browser_download_url | Format-Table -Wrap
$asset = $release.assets | Where-Object { $_.name -match "CUDA12" -and $_.name -match "v3" } | Select-Object -First 1
if ($null -eq $asset) {
    $asset = $release.assets | Where-Object { $_.name -match "CUDA12" } | Select-Object -First 1
}
if ($null -eq $asset) {
    throw "No CUDA12 DLL archive was found in the GitHub release assets. Check the printed asset names above and select a CUDA12 cuDNN9 archive manually."
}
$archive = Join-Path $env:TEMP $asset.name

Invoke-WebRequest $asset.browser_download_url -OutFile $archive
winget install --id 7zip.7zip -e
& "$env:ProgramFiles\7-Zip\7z.exe" x $archive "-o$dllDir" -y

$sourceDir = Get-ChildItem $dllDir -Recurse -Filter cublas64_12.dll | Select-Object -First 1 | ForEach-Object { $_.Directory.FullName }
if ((Resolve-Path $sourceDir).Path -ne (Resolve-Path $dllDir).Path) {
    Copy-Item (Join-Path $sourceDir "*.dll") $dllDir -Force
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$dllDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$dllDir", "User")
}
$env:Path = "$env:Path;$dllDir"

Get-ChildItem $dllDir -Filter cublas64_12.dll
Get-ChildItem $dllDir -Filter cudnn*.dll
```

Then test GPU transcription again:

```powershell
python transcribe_windows.py
```

Transcript lines are saved as:

```text
[HH:MM:SS] You: ...
[HH:MM:SS] Others: ...
```

## Notes

- The first model run downloads model files. After that, transcription is local.
- The selected loopback must match the playback device your audio app is using.
- Bluetooth headsets can expose separate low-quality hands-free and high-quality stereo devices. For best results, use the same playback device that produces the audio you actually hear.
- If no loopback devices appear, check that Windows sees at least one enabled playback device and run PowerShell from a normal desktop session, not a remote/headless session.
- Windows capture uses a 48 kHz recording stream and resamples to Whisper's 16 kHz input rate. This avoids many `data discontinuity in recording` warnings from Windows audio drivers.
