# Phase 2 Audio Routing Plan

## Summary

Phase 2 has two separate pieces:

1. A standalone audio route switcher command that updates the existing macOS `Multi-Output Device`.
2. A focused `transcribe.py` update that captures two input streams: your mic and `BlackHole 2ch`.

The output route switcher should be independent from `transcribe.py` so it can be run from any shell before or during a meeting. `transcribe.py` should not manage output devices.

## Audio Routing Goal

The desired Teams audio path is:

```text
Teams Speaker -> Multi-Output Device -> selected physical output + BlackHole 2ch
```

This lets you hear the meeting through the currently selected physical output while `BlackHole 2ch` receives the same audio as an input source for transcription.

The desired transcription capture path is:

```text
Mic input       -> transcribe.py -> You
BlackHole 2ch   -> transcribe.py -> Others
```

`Microsoft Teams Audio` should not be used as a capture device. It has already been observed as silent locally and is not the supported capture path for this project.

## Standalone Route Switcher

Create a separate helper command, likely named:

```text
switch_meeting_output
```

The helper should:

- Find the existing macOS device named `Multi-Output Device`.
- Find `BlackHole 2ch`.
- List currently available physical output devices.
- Exclude `BlackHole 2ch`, `Microsoft Teams Audio`, and virtual/aggregate devices where identifiable.
- Prompt the user to select the physical output to hear through.
- Recreate or update `Multi-Output Device` so it contains:
  - the selected physical output
  - `BlackHole 2ch`
- Set `Multi-Output Device` as the macOS default output when possible.
- Print a reminder that Microsoft Teams Speaker should be set to `Multi-Output Device`.

Implementation recommendation:

- Use Swift with Core Audio APIs.
- `/usr/bin/swift` is available on this machine.
- Python Core Audio/PyObjC bindings are not currently installed.
- Editing a Multi-Output/Aggregate device is not a normal `sounddevice` operation.

## Shell Setup

Compile the helper into a repo-local executable:

```bash
cd ~/meeting-transcriber
mkdir -p bin
CLANG_MODULE_CACHE_PATH=/tmp/meeting-transcriber-module-cache swiftc audio_route.swift -o bin/switch_meeting_output
```

Add an alias to `~/.zshrc`:

```zsh
alias switch_meeting_output="/Users/hackerboy/meeting-transcriber/bin/switch_meeting_output"
```

Reload the shell config:

```bash
source ~/.zshrc
```

Expected use:

```bash
switch_meeting_output
```

Then choose the physical output device for the current session.

## `transcribe.py` Changes

`transcribe.py` should only ask for:

1. The mic input device for your voice.
2. A custom transcript name component.

It should not ask for an output device.

Required behavior:

- Auto-detect `BlackHole 2ch` as the remote/Teams audio input.
- If `BlackHole 2ch` is missing, exit with a clear setup error.
- List available input devices for mic selection.
- Ask for a custom transcript name component.
- Save transcript as:

```text
<custom_name>_<YYYYMMDD_HHMMSS>.txt
```

If the custom name is blank, use:

```text
transcript_<YYYYMMDD_HHMMSS>.txt
```

Open two simultaneous input streams:

- selected mic input -> label transcript lines as `You`
- `BlackHole 2ch` -> label transcript lines as `Others`

Write both streams to the same transcript file with timestamped lines:

```text
[HH:MM:SS] You: ...
[HH:MM:SS] Others: ...
```

## Testing Before Implementation

Before implementing this plan, verify that `BlackHole 2ch` is receiving routed audio.

Run:

```bash
cd ~/meeting-transcriber
source venv/bin/activate
python test_audio.py
```

Then:

1. Select `BlackHole 2ch` as the input device.
2. Select any output device when prompted.
3. Choose option `1`, input-only test.
4. Play Teams audio routed through `Multi-Output Device`.
5. Confirm the input level bar moves.

If the bar moves, BlackHole capture is working.

If the bar stays near zero:

- Teams Speaker may not be set to `Multi-Output Device`.
- `Multi-Output Device` may not include `BlackHole 2ch`.
- Audio may be going directly to the physical output instead of through the multi-output route.
- BlackHole may be installed but not part of the active output path.

## Implementation Test Plan

After implementation:

- Run `switch_meeting_output`.
- Confirm it lists available physical output devices.
- Select a headset, speaker, or monitor output.
- Confirm `Multi-Output Device` includes the selected output plus `BlackHole 2ch`.
- Confirm Teams/system audio is heard through the selected physical output.
- Confirm `test_audio.py` shows input activity for `BlackHole 2ch`.
- Run `python transcribe.py`.
- Confirm it asks only for mic input and transcript name.
- Speak into the selected mic and confirm `You:` lines appear.
- Play routed Teams/system audio and confirm `Others:` lines appear.
- Confirm the output filename includes the custom name and date/time.

## Assumptions

- The macOS output route is named exactly `Multi-Output Device`.
- `BlackHole 2ch` is installed and visible to Core Audio.
- Microsoft Teams Speaker will be set to `Multi-Output Device`.
- `BlackHole 2ch` is the only supported remote-audio capture path.
- `Microsoft Teams Audio` remains unused for transcription capture.
