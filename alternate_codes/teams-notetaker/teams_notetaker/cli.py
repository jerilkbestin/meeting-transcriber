"""Command-line interface for teams-notetaker."""

from __future__ import annotations

import argparse
import datetime as dt
import queue
import re
import sys
import threading
import time
from pathlib import Path

from .config import Settings, TRACK_MIC, TRACK_SYSTEM

CONSENT_BANNER = """
------------------------------------------------------------------
 This tool records the audio of a meeting you are in.
 Tell the other participants before you start. Recording people
 without telling them can breach your employer's policy, privacy
 law, and their trust. You are responsible for getting consent.
------------------------------------------------------------------
""".strip()


def check_dependencies(need_capture: bool) -> list[str]:
    """Return a list of missing third-party packages.

    Checked up front, all at once, so a fresh clone reports everything that is
    missing in one message instead of failing one import at a time -- and so a
    missing package is never mistaken for a model download problem.
    """
    missing: list[str] = []

    def probe(module: str, package: str) -> None:
        import importlib.util

        try:
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        except (ImportError, ValueError):
            missing.append(package)

    probe("numpy", "numpy")
    probe("faster_whisper", "faster-whisper")
    probe("soxr", "soxr")
    if need_capture and sys.platform.startswith("win"):
        probe("pyaudiowpatch", "PyAudioWPatch")
    return missing


def _report_missing(missing: list[str]) -> None:
    print("error: required packages are not installed:", file=sys.stderr)
    for pkg in missing:
        print(f"  - {pkg}", file=sys.stderr)
    print("\nInstall everything with:\n", file=sys.stderr)
    print("  python -m pip install -r requirements.txt\n", file=sys.stderr)
    print(
        "(Run that from this folder. If you use a virtual environment, activate "
        "it first with .\\.venv\\Scripts\\Activate.ps1)",
        file=sys.stderr,
    )


def _report_model_load_failure(exc: Exception, model_size: str) -> None:
    """Explain a model load failure by its actual cause.

    The three realistic causes need three different fixes, and guessing wrong
    sends the user chasing the wrong problem.
    """
    text = f"{type(exc).__name__}: {exc}"
    print(f"error: could not load the speech model '{model_size}'.", file=sys.stderr)
    print(f"  {text}\n", file=sys.stderr)

    lowered = text.lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        print(
            "A required package is missing. Run:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
    elif any(
        k in lowered
        for k in ("proxy", "connection", "timed out", "timeout", "resolve",
                  "network", "ssl", "certificate", "403", "404", "getaddrinfo")
    ):
        print(
            "This looks like a network problem reaching huggingface.co, where the\n"
            "weights are hosted. Either your network blocks it, or you are offline.\n"
            "Workaround: download a CTranslate2-converted Whisper model on a machine\n"
            "that can reach it, copy the folder over, and pass its path:\n"
            '  python -m teams_notetaker record --model "C:\\models\\faster-whisper-small.en"',
            file=sys.stderr,
        )
    elif "no such file" in lowered or "not a directory" in lowered or "invalid" in lowered:
        print(
            f"'{model_size}' was treated as a path but does not look like a valid\n"
            "model folder. Use a size name (tiny.en, base.en, small.en, medium.en,\n"
            "large-v3) or the path to a CTranslate2 model directory.",
            file=sys.stderr,
        )
    else:
        print(
            "If your network blocks huggingface.co, download a CTranslate2 Whisper\n"
            "model manually and pass the folder path to --model. Otherwise try a\n"
            "smaller model, e.g. --model base.en",
            file=sys.stderr,
        )


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s[:60] or "meeting"


def _session_dir(base: Path, title: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    d = base / f"{stamp}_{_slug(title)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_from_args(args, session_dir: Path) -> Settings:
    s = Settings(out_dir=session_dir, title=args.title)
    s.model_size = args.model
    s.device = args.device
    s.compute_type = args.compute_type
    s.language = None if args.language in {"auto", "", None} else args.language
    s.anthropic_model = args.claude_model
    s.notes_backend = args.notes
    s.local_base_url = args.local_url
    s.local_model = args.local_model
    s.skip_notes = args.skip_notes or args.notes == "none"
    # Whisper's initial_prompt is a *text* prior, not a token list, so a natural
    # sentence biases decoding better than a bare comma-separated list does.
    s.initial_prompt = f"Terms used in this meeting: {args.vocab}." if args.vocab else None
    if hasattr(args, "min_chunk"):
        s.min_chunk_seconds = args.min_chunk
        s.max_chunk_seconds = args.max_chunk
    if hasattr(args, "no_mic"):
        s.capture_mic = not args.no_mic
        s.capture_system = not args.no_system
        s.mic_device_index = args.mic_device
        s.system_device_index = args.system_device
        s.save_audio = not args.no_save_audio
        s.speaker_names = {TRACK_MIC: args.my_name, TRACK_SYSTEM: args.others_name}
    return s


# --------------------------------------------------------------------------
# subcommand: devices
# --------------------------------------------------------------------------

def cmd_devices(args) -> int:
    from .audio import list_devices, AudioUnavailable

    try:
        devices = list_devices()
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{'idx':>4}  {'type':<9} {'ch':>3} {'rate':>7}  name")
    print("-" * 78)
    for d in devices:
        kind = "loopback" if d.is_loopback else "input"
        print(f"{d.index:>4}  {kind:<9} {d.channels:>3} {d.sample_rate:>7}  {d.name}")
    print(
        "\nUse a 'loopback' device with --system-device (that is the other "
        "participants) and a plain 'input' device with --mic-device (that is you)."
    )
    return 0


# --------------------------------------------------------------------------
# subcommand: record
# --------------------------------------------------------------------------

def cmd_record(args) -> int:
    from .audio import (
        AudioUnavailable,
        DeviceInfo,
        TrackCapture,
        WavWriter,
        find_default_loopback,
        find_default_mic,
        pyaudio,
        _require_pyaudio,
    )
    from .engine import RollingTranscriber
    from .config import TARGET_SR

    missing = check_dependencies(need_capture=True)
    if missing:
        _report_missing(missing)
        return 4

    print(CONSENT_BANNER)
    print()

    try:
        _require_pyaudio()
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session_dir = _session_dir(Path(args.out).expanduser(), args.title)
    settings = _settings_from_args(args, session_dir)

    transcriber = RollingTranscriber(settings, session_dir / "transcript.jsonl")

    # Load the model BEFORE opening any stream. First run downloads weights,
    # which can take minutes -- we do not want that happening while the meeting
    # is already running and audio is piling up in a queue.
    print(f"Loading speech model '{settings.model_size}' (first run downloads it)...")
    t0 = time.time()
    try:
        transcriber.load_model()
    except Exception as exc:  # noqa: BLE001
        _report_model_load_failure(exc, settings.model_size)
        return 3
    print(f"Model ready in {time.time() - t0:.1f}s.\n")

    audio_q: "queue.Queue[tuple[str, object]]" = queue.Queue(maxsize=8000)
    pa = pyaudio.PyAudio()
    captures: list[TrackCapture] = []
    writers: list[WavWriter] = []

    def _device_for(index: int | None, fallback) -> DeviceInfo:
        if index is None:
            return fallback(pa)
        info = pa.get_device_info_by_index(index)
        return DeviceInfo(
            index=int(info["index"]),
            name=str(info["name"]),
            channels=max(1, min(int(info["maxInputChannels"]), 2)),
            sample_rate=int(info["defaultSampleRate"]),
            is_loopback=bool(info.get("isLoopbackDevice", False)),
        )

    try:
        if settings.capture_system:
            dev = _device_for(settings.system_device_index, find_default_loopback)
            w = WavWriter(session_dir / "audio_participants.wav", TARGET_SR) if settings.save_audio else None
            if w:
                writers.append(w)
            captures.append(TrackCapture(pa, dev, TRACK_SYSTEM, audio_q, w))
            print(f"  participants <- [{dev.index}] {dev.name}")
        if settings.capture_mic:
            dev = _device_for(settings.mic_device_index, find_default_mic)
            w = WavWriter(session_dir / "audio_me.wav", TARGET_SR) if settings.save_audio else None
            if w:
                writers.append(w)
            captures.append(TrackCapture(pa, dev, TRACK_MIC, audio_q, w))
            print(f"  me           <- [{dev.index}] {dev.name}")
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        pa.terminate()
        return 2

    if not captures:
        print("error: nothing to capture (both --no-mic and --no-system given).", file=sys.stderr)
        pa.terminate()
        return 2

    stop_event = threading.Event()

    def feeder() -> None:
        """Move blocks from the audio callback queue into the transcriber."""
        while not stop_event.is_set():
            try:
                track, block = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            transcriber.feed(track, block)

    feeder_thread = threading.Thread(target=feeder, name="feeder", daemon=True)
    feeder_thread.start()

    for cap in captures:
        cap.start()

    started = time.time()
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    print(f"\nRecording. Output folder: {session_dir}")
    print("Press Ctrl+C to stop and write the notes.\n")

    try:
        while True:
            new = transcriber.drain(force=False)
            for u in new:
                print(f"[{u.timestamp()}] {u.speaker}: {u.text}", flush=True)
            if not new:
                time.sleep(0.4)
            if deadline and time.time() > deadline:
                print("\n(reached --max-minutes limit, stopping)")
                break
    except KeyboardInterrupt:
        print("\n\nStopping...")

    stop_event.set()
    for cap in captures:
        cap.stop()
    feeder_thread.join(timeout=2.0)

    # Anything still sitting in the queue when we stopped.
    while True:
        try:
            track, block = audio_q.get_nowait()
        except queue.Empty:
            break
        transcriber.feed(track, block)

    print("Transcribing the final audio...")
    for u in transcriber.drain(force=True):
        print(f"[{u.timestamp()}] {u.speaker}: {u.text}", flush=True)

    for cap in captures:
        if cap.dropped_blocks:
            print(f"warning: dropped {cap.dropped_blocks} audio blocks on '{cap.name}'.")
    for w in writers:
        w.close()
    pa.terminate()
    transcriber.close()

    utterances = transcriber.utterances()
    if not utterances:
        print("No speech was transcribed. Check the device selection with 'devices'.")
        return 1

    duration_min = (time.time() - started) / 60
    _write_outputs(settings, session_dir, utterances, duration_min, args)
    return 0


def _local_server_reachable(base_url: str, timeout: float = 1.5) -> bool:
    """Cheap liveness probe so `--notes auto` can pick the local backend.

    We only care whether something is listening and answering HTTP; the model
    listing endpoint differs between runtimes, so any response at all counts.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    try:
        urllib.request.urlopen(f"{url}/models", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # responded, just not with 200 -- server is up
    except Exception:  # noqa: BLE001
        return False


def _resolve_backend(settings) -> tuple[str, str]:
    """Decide which backend to use. Returns (backend, human description)."""
    choice = settings.notes_backend
    if settings.skip_notes or choice == "none":
        return "none", "none"

    if choice == "auto":
        if settings.api_key():
            choice = "claude"
        elif _local_server_reachable(settings.local_base_url):
            choice = "local"
        else:
            return "none", "none"

    if choice == "claude":
        return "claude", f"Claude ({settings.anthropic_model})"
    return "local", f"local model {settings.local_model} at {settings.local_base_url}"


def _write_outputs(settings, session_dir: Path, utterances, duration_min: float, args) -> None:
    from .engine import render_markdown_transcript, render_plain_transcript
    from .summarize import (
        SummarizationError,
        build_paste_prompt,
        claude_backend,
        local_backend,
        summarize_transcript,
    )

    (session_dir / "transcript.md").write_text(
        render_markdown_transcript(utterances, settings.title), encoding="utf-8"
    )
    plain = render_plain_transcript(utterances)
    (session_dir / "transcript.txt").write_text(plain, encoding="utf-8")
    print(f"\nTranscript written ({len(utterances)} segments, ~{duration_min:.0f} min).")

    context = getattr(args, "context", None)

    # Always write the paste-ready prompt. This is the no-API-key path: open the
    # file, select all, paste into claude.ai. It costs nothing to produce and it
    # also doubles as the fallback if the model call below fails.
    prompt_path = session_dir / "notes-prompt.md"
    prompt_path.write_text(
        build_paste_prompt(plain, settings.title, context), encoding="utf-8"
    )

    backend, description = _resolve_backend(settings)

    if backend == "none":
        print(f"\nNo notes model configured — transcript only.")
        print(f"  Paste-ready prompt: {prompt_path}")
        print("  Open it, select all, paste into claude.ai to get the notes.")
        print(f"\nDone. Files are in: {session_dir}")
        return

    print(f"Generating notes with {description}...")
    try:
        if backend == "claude":
            call = claude_backend(
                settings.api_key(), settings.anthropic_model, settings.max_output_tokens
            )
            budget = settings.summarize_char_budget
        else:
            call = local_backend(
                settings.local_base_url, settings.local_model, settings.max_output_tokens
            )
            budget = settings.local_char_budget

        notes = summarize_transcript(
            plain,
            title=settings.title,
            call=call,
            char_budget=budget,
            context=context,
            progress=print,
        )
    except SummarizationError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        print(f"\nTranscript is saved: {session_dir / 'transcript.txt'}", file=sys.stderr)
        print(f"Paste-ready prompt:  {prompt_path}", file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"\nerror: notes generation failed: {exc}", file=sys.stderr)
        print(f"\nTranscript is saved: {session_dir / 'transcript.txt'}", file=sys.stderr)
        print(f"Paste-ready prompt:  {prompt_path}", file=sys.stderr)
        return

    header = (
        f"# {settings.title}\n\n"
        f"*Recorded {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"~{duration_min:.0f} min · notes by {description} from an "
        f"automatic transcript — verify anything load-bearing.*\n\n"
    )
    (session_dir / "notes.md").write_text(header + notes, encoding="utf-8")
    print(f"\nNotes written to: {session_dir / 'notes.md'}\n")
    print(notes)


# --------------------------------------------------------------------------
# subcommand: file
# --------------------------------------------------------------------------

def cmd_file(args) -> int:
    missing = check_dependencies(need_capture=False)
    if missing:
        _report_missing(missing)
        return 4

    from .engine import transcribe_file

    src = Path(args.path).expanduser()
    if not src.exists():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 2
    title = args.title if args.title != "Meeting" else src.stem
    session_dir = _session_dir(Path(args.out).expanduser(), title)
    settings = _settings_from_args(args, session_dir)
    settings.title = title

    print(f"Transcribing {src.name} with '{settings.model_size}'...")
    t0 = time.time()
    try:
        utterances = transcribe_file(settings, src, speaker=args.speaker)
    except Exception as exc:  # noqa: BLE001
        _report_model_load_failure(exc, settings.model_size)
        return 3
    print(f"Done in {time.time() - t0:.0f}s — {len(utterances)} segments.")
    if not utterances:
        print("No speech found in that file.", file=sys.stderr)
        return 1

    import json
    with (session_dir / "transcript.jsonl").open("w", encoding="utf-8") as fh:
        from dataclasses import asdict
        for u in utterances:
            fh.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")

    _write_outputs(settings, session_dir, utterances, (utterances[-1].end) / 60, args)
    return 0


# --------------------------------------------------------------------------
# subcommand: notes
# --------------------------------------------------------------------------

def cmd_notes(args) -> int:
    from .summarize import (
        SummarizationError,
        build_paste_prompt,
        claude_backend,
        local_backend,
        read_transcript_source,
        summarize_transcript,
    )

    src = Path(args.path).expanduser()
    if not src.exists():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 2
    title = args.title if args.title != "Meeting" else src.stem
    transcript = read_transcript_source(src)
    if not transcript.strip():
        print("error: that file produced an empty transcript.", file=sys.stderr)
        return 1

    settings = Settings(out_dir=src.parent, title=title)
    settings.notes_backend = args.notes
    settings.anthropic_model = args.claude_model
    settings.local_base_url = args.local_url
    settings.local_model = args.local_model

    prompt_path = src.with_name(f"{src.stem}_prompt.md")
    prompt_path.write_text(build_paste_prompt(transcript, title, args.context), encoding="utf-8")

    backend, description = _resolve_backend(settings)
    if backend == "none":
        print(f"No notes model configured — wrote a paste-ready prompt instead:")
        print(f"  {prompt_path}")
        print("Open it, select all, paste into claude.ai.")
        return 0

    print(f"Generating notes from {src.name} with {description}...")
    try:
        if backend == "claude":
            call = claude_backend(
                settings.api_key(), settings.anthropic_model, settings.max_output_tokens
            )
            budget = settings.summarize_char_budget
        else:
            call = local_backend(
                settings.local_base_url, settings.local_model, settings.max_output_tokens
            )
            budget = settings.local_char_budget
        notes = summarize_transcript(
            transcript,
            title=title,
            call=call,
            char_budget=budget,
            context=args.context,
            progress=print,
        )
    except SummarizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"Paste-ready prompt is still at: {prompt_path}", file=sys.stderr)
        return 3

    out = Path(args.output).expanduser() if args.output else src.with_name(f"{src.stem}_notes.md")
    out.write_text(f"# {title}\n\n" + notes, encoding="utf-8")
    print(f"\nNotes written to: {out}\n")
    print(notes)
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--title", default="Meeting", help="Meeting title used in filenames and notes.")
    p.add_argument("--out", default="./meetings", help="Base output folder (default: ./meetings).")
    p.add_argument(
        "--model",
        default="small.en",
        help="Whisper model size or a local CTranslate2 model folder "
        "(tiny.en, base.en, small.en, medium.en, large-v3, large-v3-turbo, ...).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 compute type: int8 (CPU), int8_float16 or float16 (GPU).",
    )
    p.add_argument("--language", default="en", help="Language code, or 'auto' to detect.")
    p.add_argument(
        "--notes",
        default="auto",
        choices=["auto", "claude", "local", "none"],
        help="Who writes the notes. auto = Claude if ANTHROPIC_API_KEY is set, "
        "else a local model if one is reachable, else none. "
        "none = transcript + a paste-ready prompt file only.",
    )
    p.add_argument("--claude-model", default="claude-sonnet-5", help="Anthropic model for the notes.")
    p.add_argument(
        "--local-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible endpoint (Ollama 11434, LM Studio 1234, llama.cpp 8080).",
    )
    p.add_argument("--local-model", default="llama3.1:8b", help="Model name on the local server.")
    p.add_argument("--skip-notes", action="store_true", help="Transcript only; same as --notes none.")
    p.add_argument(
        "--vocab",
        default=None,
        help="Comma-separated jargon, product and people names to bias the recogniser "
        "(e.g. --vocab 'Cavallo, Kubernetes, Nagendra').",
    )
    p.add_argument("--context", default=None, help="One line of context for the notes writer.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="teams-notetaker",
        description="Record a Teams meeting from your own machine, transcribe it "
        "locally with Whisper, and write structured notes with Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tell everyone on the call that you are recording.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("devices", help="List audio input and loopback devices.")
    d.set_defaults(func=cmd_devices)

    r = sub.add_parser("record", help="Record a live meeting and produce notes.")
    _add_common(r)
    r.add_argument("--mic-device", type=int, default=None, help="Input device index for your voice.")
    r.add_argument("--system-device", type=int, default=None, help="Loopback device index for the others.")
    r.add_argument("--no-mic", action="store_true", help="Do not capture your microphone.")
    r.add_argument("--no-system", action="store_true", help="Do not capture speaker output.")
    r.add_argument("--no-save-audio", action="store_true", help="Do not keep the .wav files.")
    r.add_argument("--my-name", default="Me", help="Label for the microphone track.")
    r.add_argument("--others-name", default="Participants", help="Label for the speaker track.")
    r.add_argument("--min-chunk", type=float, default=12.0, help="Seconds before a silence cut is allowed.")
    r.add_argument("--max-chunk", type=float, default=30.0, help="Hard cut after this many seconds.")
    r.add_argument("--max-minutes", type=float, default=0, help="Auto-stop after N minutes (0 = no limit).")
    r.set_defaults(func=cmd_record)

    f = sub.add_parser("file", help="Transcribe an existing recording, then write notes.")
    _add_common(f)
    f.add_argument("path", help="Audio or video file (mp4, m4a, wav, mp3, ...).")
    f.add_argument("--speaker", default="Speaker", help="Label for all speech in the file.")
    f.set_defaults(func=cmd_file)

    n = sub.add_parser("notes", help="Write notes from an existing transcript (.vtt/.srt/.txt/.jsonl).")
    n.add_argument("path", help="Transcript file, e.g. one exported from Teams.")
    n.add_argument("--title", default="Meeting")
    n.add_argument("--output", default=None, help="Where to write the notes markdown.")
    n.add_argument("--notes", default="auto", choices=["auto", "claude", "local", "none"])
    n.add_argument("--claude-model", default="claude-sonnet-5")
    n.add_argument("--local-url", default="http://localhost:11434/v1")
    n.add_argument("--local-model", default="llama3.1:8b")
    n.add_argument("--context", default=None)
    n.set_defaults(func=cmd_notes)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
