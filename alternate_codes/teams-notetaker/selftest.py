"""Offline verification of every part of the pipeline that does not need
Windows audio hardware or a Whisper model download.

Run:  python selftest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from teams_notetaker.config import Settings, TARGET_SR, TRACK_MIC, TRACK_SYSTEM  # noqa: E402
from teams_notetaker import audio, engine, summarize, cli  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- compile
def test_compile() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "teams_notetaker"],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
    )
    check("all modules compile", r.returncode == 0, r.stdout + r.stderr)


# ------------------------------------------------------------------ audio
def test_mono_conversion() -> None:
    stereo = np.array([1000, -1000, 2000, -2000], dtype=np.int16)  # 2 frames, 2ch
    mono = audio.to_mono_float32(stereo.tobytes(), 2)
    check("stereo -> mono length", mono.shape == (2,), str(mono.shape))
    check("stereo -> mono averaging", np.allclose(mono, [0.0, 0.0], atol=1e-6), str(mono))

    single = np.array([16384, -16384], dtype=np.int16)
    m2 = audio.to_mono_float32(single.tobytes(), 1)
    check("mono scaling to [-1,1]", np.allclose(m2, [0.5, -0.5], atol=1e-3), str(m2))

    odd = np.array([1, 2, 3], dtype=np.int16)  # not a whole number of frames
    m3 = audio.to_mono_float32(odd.tobytes(), 2)
    check("ragged buffer does not crash", m3.shape == (1,), str(m3.shape))


def test_resample() -> None:
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = audio.resample_to_target(tone, sr)
    check("48k -> 16k length", abs(len(out) - TARGET_SR) <= 2, str(len(out)))
    check("resample preserves amplitude", 0.6 < float(np.max(np.abs(out))) <= 1.05,
          str(float(np.max(np.abs(out)))))
    # Frequency check: dominant bin should still be ~440 Hz.
    spec = np.abs(np.fft.rfft(out))
    peak_hz = float(np.argmax(spec)) * TARGET_SR / len(out)
    check("resample preserves pitch (~440Hz)", abs(peak_hz - 440) < 8, f"{peak_hz:.1f} Hz")

    passthrough = audio.resample_to_target(tone[:100], TARGET_SR)
    check("same-rate passthrough", passthrough.shape == (100,), str(passthrough.shape))


def test_wav_writer() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.wav"
        w = audio.WavWriter(p, TARGET_SR)
        w.write(np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32))  # 2.0 must clip
        w.close()
        with wave.open(str(p), "rb") as wf:
            check("wav is mono", wf.getnchannels() == 1)
            check("wav is 16-bit", wf.getsampwidth() == 2)
            check("wav rate is 16k", wf.getframerate() == TARGET_SR)
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        check("wav frame count", frames.shape == (4,), str(frames.shape))
        check("wav clipping handled", frames[3] == 32767, str(frames[3]))
        w.close()  # double close must be safe
        check("double close is safe", True)


def test_audio_unavailable_message() -> None:
    # This container is Linux, so this is the real not-Windows path.
    try:
        audio.list_devices()
        check("list_devices raises on non-Windows", False, "no exception raised")
    except audio.AudioUnavailable as exc:
        check("list_devices raises AudioUnavailable", True)
        check("error message is actionable", "PyAudioWPatch" in str(exc) or "Windows" in str(exc), str(exc))


# ----------------------------------------------------------------- engine
def test_track_buffer() -> None:
    b = engine._TrackBuffer("x")
    b.append(np.zeros(TARGET_SR, dtype=np.float32))
    check("buffer seconds", abs(b.seconds() - 1.0) < 1e-6, str(b.seconds()))
    b.append(np.ones(TARGET_SR // 2, dtype=np.float32) * 0.5)
    check("buffer accumulates", abs(b.seconds() - 1.5) < 1e-6, str(b.seconds()))

    loud_tail = b.tail_rms(0.25)
    check("tail_rms sees loud tail", loud_tail > 0.4, str(loud_tail))

    b.append(np.zeros(TARGET_SR, dtype=np.float32))
    check("tail_rms sees silence", b.tail_rms(0.5) < 1e-6, str(b.tail_rms(0.5)))

    taken = b.take_all()
    check("take_all returns everything", taken.size == int(TARGET_SR * 2.5), str(taken.size))
    check("take_all resets buffer", b.n_samples == 0)
    check("offset advances", abs(b.peek_offset_seconds() - 2.5) < 1e-6, str(b.peek_offset_seconds()))

    short = engine._TrackBuffer("y")
    check("tail_rms on too-short buffer is inf", short.tail_rms(1.0) == float("inf"))


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.05


class FakeModel:
    """Stands in for WhisperModel so we can verify chunking and timestamping
    without a 500 MB download or any network access."""

    def __init__(self):
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(self, audio_array, **kwargs):
        self.calls.append((len(audio_array), kwargs.get("initial_prompt")))
        n = len(self.calls)
        secs = len(audio_array) / TARGET_SR
        return ([FakeSegment(0.0, secs, f"utterance {n}")], object())


def _transcriber(tmp: Path, **overrides) -> tuple[engine.RollingTranscriber, FakeModel]:
    s = Settings(out_dir=tmp, title="T")
    s.min_chunk_seconds = 2.0
    s.max_chunk_seconds = 4.0
    s.silence_window_seconds = 0.5
    for k, v in overrides.items():
        setattr(s, k, v)
    t = engine.RollingTranscriber(s, tmp / "transcript.jsonl")
    fake = FakeModel()
    t._model = fake
    return t, fake


def test_chunking_silence_cut() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, fake = _transcriber(tmp)

        # 1.0 s of speech: below min_chunk, must not fire.
        t.feed(TRACK_MIC, np.ones(TARGET_SR, dtype=np.float32) * 0.3)
        check("no flush below min_chunk", t.drain() == [])

        # Now 1.5 s more speech (2.5 s total) but still talking -> no cut.
        t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 1.5), dtype=np.float32) * 0.3)
        check("no flush while still speaking", t.drain() == [])

        # Add 0.6 s of silence -> silence cut should fire.
        t.feed(TRACK_MIC, np.zeros(int(TARGET_SR * 0.6), dtype=np.float32))
        out = t.drain()
        check("silence triggers a flush", len(out) == 1, str(out))
        check("flushed chunk length ~3.1s", abs(fake.calls[0][0] / TARGET_SR - 3.1) < 0.05,
              str(fake.calls[0][0] / TARGET_SR))
        t.close()


def test_chunking_hard_cut() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, fake = _transcriber(tmp)
        # 5 s of continuous loud speech, no silence anywhere: max_chunk (4 s) wins.
        t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 5), dtype=np.float32) * 0.5)
        out = t.drain()
        check("hard cut fires without silence", len(out) == 1, str(out))
        t.close()


def test_absolute_timestamps() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, fake = _transcriber(tmp)
        speech = np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3
        silence = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)

        t.feed(TRACK_MIC, speech)
        t.feed(TRACK_MIC, silence)
        first = t.drain()
        t.feed(TRACK_MIC, speech)
        t.feed(TRACK_MIC, silence)
        second = t.drain()

        check("two chunks produced", len(first) == 1 and len(second) == 1)
        check("first chunk starts at 0", abs(first[0].start) < 1e-6, str(first[0].start))
        check(
            "second chunk offset by first chunk duration",
            abs(second[0].start - 3.1) < 0.05,
            str(second[0].start),
        )
        check("timestamp formatting", first[0].timestamp() == "00:00:00", first[0].timestamp())

        # Carry-forward prompt should contain the previous chunk's text.
        check("carry prompt fed forward", fake.calls[1][1] is not None and
              "utterance 1" in fake.calls[1][1], str(fake.calls[1][1]))
        t.close()


def test_tracks_are_independent() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, fake = _transcriber(tmp)
        speech = np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3
        silence = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)
        for track in (TRACK_MIC, TRACK_SYSTEM):
            t.feed(track, speech)
            t.feed(track, silence)
        out = t.drain(force=True)
        tracks = {u.track for u in out}
        check("both tracks transcribed", tracks == {TRACK_MIC, TRACK_SYSTEM}, str(tracks))
        speakers = {u.speaker for u in out}
        check("speaker labels applied", speakers == {"Me", "Participants"}, str(speakers))
        t.close()


def test_jsonl_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, _ = _transcriber(tmp)
        t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3)
        t.drain(force=True)
        t.close()
        path = tmp / "transcript.jsonl"
        check("jsonl written", path.exists())
        loaded = engine.utterances_from_jsonl(path)
        check("jsonl reloads into Utterances", len(loaded) == 1 and loaded[0].text == "utterance 1",
              str(loaded))
        raw = json.loads(path.read_text().splitlines()[0])
        check("jsonl has expected keys",
              {"track", "speaker", "start", "end", "text"} <= set(raw), str(raw.keys()))


def test_hallucination_filter() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, _ = _transcriber(tmp)

        class NoisyModel:
            def transcribe(self, a, **k):
                return (
                    [
                        FakeSegment(0, 1, "Thanks for watching!", avg_logprob=-1.5, no_speech_prob=0.95),
                        FakeSegment(1, 2, "real content", avg_logprob=-0.3, no_speech_prob=0.1),
                        FakeSegment(2, 3, "   "),
                    ],
                    object(),
                )

        t._model = NoisyModel()
        t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3)
        out = t.drain(force=True)
        texts = [u.text for u in out]
        check("low-confidence silence hallucination dropped", texts == ["real content"], str(texts))
        t.close()


def test_merge_and_render() -> None:
    U = engine.Utterance
    utts = [
        U(TRACK_MIC, "Me", 0.0, 2.0, "Hello there.", -0.2, 0.05),
        U(TRACK_MIC, "Me", 2.5, 4.0, "How are you?", -0.3, 0.05),
        U(TRACK_SYSTEM, "Participants", 4.5, 6.0, "Fine thanks.", -0.2, 0.05),
        U(TRACK_MIC, "Me", 20.0, 21.0, "Later point.", -0.2, 0.05),
    ]
    merged = engine.merge_adjacent(utts)
    check("adjacent same-speaker merged", len(merged) == 3, str(len(merged)))
    check("merged text joined", merged[0].text == "Hello there. How are you?", merged[0].text)
    check("merged end extended", merged[0].end == 4.0, str(merged[0].end))
    check("large gap not merged", merged[2].text == "Later point.", merged[2].text)

    md = engine.render_markdown_transcript(utts, "Standup")
    check("markdown has title", md.startswith("# Transcript — Standup"))
    check("markdown has speaker lines", "**[00:00:00] Me:**" in md, md[:200])
    plain = engine.render_plain_transcript(utts)
    check("plain transcript formatted", plain.splitlines()[0] == "[00:00:00] Me: Hello there. How are you?",
          plain.splitlines()[0])

    long = U(TRACK_MIC, "Me", 3725.0, 3730.0, "x", -0.2, 0.05)
    check("timestamp handles hours", long.timestamp() == "01:02:05", long.timestamp())

    check("merge of empty list", engine.merge_adjacent([]) == [])


# -------------------------------------------------------------- summarize
def test_vtt_parsing() -> None:
    vtt = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Nagendra Vippala>Let's start with the migration status.</v>

2
00:00:04.500 --> 00:00:09.000
<v Priya>Staging is done, production is blocked on the firewall ticket.</v>
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "meeting.vtt"
        p.write_text(vtt, encoding="utf-8")
        out = summarize.read_transcript_source(p)
        lines = out.splitlines()
        check("vtt produces one line per cue", len(lines) == 2, str(lines))
        check("vtt keeps speaker name", lines[0] == "[00:00:01] Nagendra Vippala: Let's start with the migration status.",
              lines[0])
        check("vtt keeps second timestamp", lines[1].startswith("[00:00:04] Priya:"), lines[1])


def test_srt_parsing() -> None:
    srt = """1
00:00:02,000 --> 00:00:05,000
No speaker tag here.
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.srt"
        p.write_text(srt, encoding="utf-8")
        out = summarize.read_transcript_source(p)
        check("srt falls back to generic speaker", out == "[00:00:02] Speaker: No speaker tag here.", out)


def test_jsonl_source() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, _ = _transcriber(tmp)
        t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3)
        t.drain(force=True)
        t.close()
        out = summarize.read_transcript_source(tmp / "transcript.jsonl")
        check("jsonl source renders", out.startswith("[00:00:00] Me: utterance 1"), out)


def test_split_transcript() -> None:
    text = "\n".join(f"[00:00:0{i%10}] Me: line {i}" for i in range(200))
    single = summarize._split_transcript(text, 10_000)
    check("short transcript is one piece", len(single) == 1)
    pieces = summarize._split_transcript(text, 300)
    check("long transcript split", len(pieces) > 1, str(len(pieces)))
    check("split is lossless", "".join(pieces) == text)
    check("no piece exceeds budget by more than one line",
          all(len(p) <= 300 + 40 for p in pieces), str([len(p) for p in pieces]))


def test_missing_api_key() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        # The Claude path is optional -- if the SDK is not installed the backend
        # raises a different (also correct) error. Nothing to assert here.
        print("[SKIP] anthropic SDK not installed; Claude backend checks skipped")
        try:
            summarize.claude_backend(None, "claude-sonnet-5")
            check("missing SDK still raises SummarizationError", False, "no exception")
        except summarize.SummarizationError as exc:
            check("missing SDK raises a clear SummarizationError",
                  "not installed" in str(exc), str(exc))
        return

    try:
        summarize.claude_backend(None, "claude-sonnet-5")
        check("missing key raises", False, "no exception")
    except summarize.SummarizationError as exc:
        check("missing key raises SummarizationError", True)
        check("key error mentions env var", "ANTHROPIC_API_KEY" in str(exc), str(exc))
        check("key error offers alternatives", "--notes local" in str(exc) and "--skip-notes" in str(exc),
              str(exc))


def test_paste_prompt() -> None:
    transcript = "[00:00:00] Me: We ship Friday."
    p = summarize.build_paste_prompt(transcript, "Release Sync", context="Weekly")
    check("paste prompt includes system rules", "Never invent content" in p)
    check("paste prompt includes all sections", all(
        s in p for s in ("## TL;DR", "## Decisions", "## Action items",
                         "## Open questions & risks")), "missing a section")
    check("paste prompt includes the transcript", transcript in p)
    check("paste prompt includes the title", "Release Sync" in p)
    check("paste prompt includes context", "Weekly" in p)

    u = summarize.build_user_prompt(transcript, "T", part=(2, 5))
    check("part header appears", "part 2 of 5" in u, u[:200])


# ------------------------------------------------- local (OpenAI-compatible)
class _FakeLLMServer:
    """A real HTTP server speaking the OpenAI chat-completions shape, so the
    local backend is tested over an actual socket rather than a mock."""

    def __init__(self, mode: str = "ok"):
        import http.server
        import threading

        self.mode = mode
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode())
                outer.requests.append(body)
                if outer.mode == "http_error":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error":"model not found"}')
                    return
                if outer.mode == "bad_shape":
                    payload = {"unexpected": True}
                elif outer.mode == "empty":
                    payload = {"choices": [{"message": {"content": "   "}}]}
                else:
                    payload = {
                        "choices": [
                            {"message": {"content": "## TL;DR\n- notes for "
                                                    + body["messages"][1]["content"][-12:]}}
                        ]
                    }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_local_backend_happy_path() -> None:
    srv = _FakeLLMServer()
    try:
        call = summarize.local_backend(srv.base_url, "llama3.1:8b")
        out = call("SYSTEM RULES", "USER CONTENT")
        check("local backend returns text", out.startswith("## TL;DR"), out)
        req = srv.requests[-1]
        check("local sends the model name", req["model"] == "llama3.1:8b", str(req.get("model")))
        check("local sends system as role=system",
              req["messages"][0] == {"role": "system", "content": "SYSTEM RULES"}, str(req["messages"][0]))
        check("local sends user as role=user",
              req["messages"][1]["role"] == "user", str(req["messages"][1]["role"]))
        check("local disables streaming", req.get("stream") is False, str(req.get("stream")))
    finally:
        srv.stop()


def test_local_backend_url_normalization() -> None:
    srv = _FakeLLMServer()
    try:
        root = f"http://127.0.0.1:{srv.port}"
        for variant in (root, root + "/", root + "/v1", root + "/v1/chat/completions"):
            call = summarize.local_backend(variant, "m")
            out = call("s", "u")
            check(f"url variant works: {variant}", out.startswith("## TL;DR"), out)
    finally:
        srv.stop()


def test_local_backend_errors() -> None:
    for mode, needle in (("http_error", "HTTP 500"), ("bad_shape", "Unexpected response"),
                         ("empty", "empty response")):
        srv = _FakeLLMServer(mode=mode)
        try:
            call = summarize.local_backend(srv.base_url, "m")
            call("s", "u")
            check(f"local backend raises on {mode}", False, "no exception")
        except summarize.SummarizationError as exc:
            check(f"local backend raises on {mode}", needle in str(exc), str(exc)[:200])
        finally:
            srv.stop()

    # Nothing listening at all.
    call = summarize.local_backend("http://127.0.0.1:1/v1", "m", timeout=2.0)
    try:
        call("s", "u")
        check("unreachable server raises", False, "no exception")
    except summarize.SummarizationError as exc:
        check("unreachable server raises", True)
        check("unreachable error suggests ollama", "ollama" in str(exc).lower(), str(exc)[:200])


def test_local_map_reduce() -> None:
    srv = _FakeLLMServer()
    try:
        call = summarize.local_backend(srv.base_url, "m")
        long_transcript = "\n".join(f"[00:00:00] Me: line {i}" for i in range(400))
        out = summarize.summarize_transcript(
            long_transcript, title="Long", call=call, char_budget=2000
        )
        systems = [r["messages"][0]["content"] for r in srv.requests]
        n_map = sum(1 for s in systems if s == summarize.SYSTEM_PROMPT)
        n_reduce = sum(1 for s in systems if s == summarize.REDUCE_SYSTEM_PROMPT)
        check("map step ran multiple times", n_map > 1, str(n_map))
        check("reduce step ran exactly once", n_reduce == 1, str(n_reduce))
        check("map-reduce returns text", out.startswith("## TL;DR"), out)
    finally:
        srv.stop()

    # Short transcript must NOT trigger map-reduce.
    srv2 = _FakeLLMServer()
    try:
        call = summarize.local_backend(srv2.base_url, "m")
        summarize.summarize_transcript("short line", title="S", call=call, char_budget=2000)
        check("short transcript = one call", len(srv2.requests) == 1, str(len(srv2.requests)))
    finally:
        srv2.stop()


def test_backend_resolution() -> None:
    import os

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        s = Settings(out_dir=Path("/tmp"), title="T")
        s.notes_backend = "auto"
        s.local_base_url = "http://127.0.0.1:1/v1"  # nothing listening
        backend, _ = cli._resolve_backend(s)
        check("auto with no key and no server -> none", backend == "none", backend)

        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        backend, desc = cli._resolve_backend(s)
        check("auto with key -> claude", backend == "claude", backend)
        check("description names the model", "claude-sonnet-5" in desc, desc)
        del os.environ["ANTHROPIC_API_KEY"]

        srv = _FakeLLMServer()
        try:
            s.local_base_url = srv.base_url
            backend, desc = cli._resolve_backend(s)
            check("auto with live local server -> local", backend == "local", backend)
            check("local reachability probe works", cli._local_server_reachable(srv.base_url))
        finally:
            srv.stop()

        check("probe returns False when down", not cli._local_server_reachable("http://127.0.0.1:1/v1"))

        s.notes_backend = "none"
        check("explicit none -> none", cli._resolve_backend(s)[0] == "none")

        s.notes_backend = "auto"
        s.skip_notes = True
        check("skip_notes forces none", cli._resolve_backend(s)[0] == "none")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)


# -------------------------------------------------------------------- cli
def test_cli_parsing() -> None:
    p = cli.build_parser()
    args = p.parse_args(["record", "--title", "Sprint Review", "--model", "base.en",
                         "--vocab", "Cavallo, Kubernetes", "--max-minutes", "90"])
    check("record parses", args.command == "record" and args.title == "Sprint Review")
    check("max-minutes is float", args.max_minutes == 90.0)

    s = cli._settings_from_args(args, Path("/tmp/x"))
    check("vocab becomes a prompt sentence",
          s.initial_prompt == "Terms used in this meeting: Cavallo, Kubernetes.", str(s.initial_prompt))
    check("speaker labels default", s.speaker_names == {TRACK_MIC: "Me", TRACK_SYSTEM: "Participants"},
          str(s.speaker_names))
    check("chunk settings applied", (s.min_chunk_seconds, s.max_chunk_seconds) == (12.0, 30.0))

    args2 = p.parse_args(["record", "--language", "auto"])
    s2 = cli._settings_from_args(args2, Path("/tmp/x"))
    check("language auto -> None", s2.language is None)
    check("no vocab -> no prompt", s2.initial_prompt is None)

    args3 = p.parse_args(["file", "rec.mp4", "--skip-notes"])
    check("file subcommand parses", args3.command == "file" and args3.path == "rec.mp4")
    s3 = cli._settings_from_args(args3, Path("/tmp/x"))
    check("file settings skip record-only fields", s3.capture_mic is True and s3.skip_notes is True)

    args5 = p.parse_args(["record", "--notes", "local", "--local-model", "qwen2.5:14b",
                          "--local-url", "http://localhost:1234/v1"])
    s5 = cli._settings_from_args(args5, Path("/tmp/x"))
    check("--notes local applied", s5.notes_backend == "local", s5.notes_backend)
    check("--local-model applied", s5.local_model == "qwen2.5:14b", s5.local_model)
    check("--local-url applied", s5.local_base_url == "http://localhost:1234/v1", s5.local_base_url)
    check("--notes local does not skip", s5.skip_notes is False)

    args6 = p.parse_args(["record", "--notes", "none"])
    s6 = cli._settings_from_args(args6, Path("/tmp/x"))
    check("--notes none implies skip_notes", s6.skip_notes is True)

    check("default backend is auto", cli._settings_from_args(
        p.parse_args(["record"]), Path("/tmp/x")).notes_backend == "auto")

    args4 = p.parse_args(["notes", "meeting.vtt", "--title", "Retro"])
    check("notes subcommand parses", args4.command == "notes" and args4.title == "Retro")

    check("slug sanitises", cli._slug("Q3 Planning / Budget!") == "q3-planning-budget",
          cli._slug("Q3 Planning / Budget!"))
    check("slug handles empty", cli._slug("!!!") == "meeting")


def test_dependency_check() -> None:
    """The probe must agree with reality, whatever is or isn't installed here.

    Written this way deliberately: this suite is meant to run on a fresh clone
    *before* `pip install`, so it cannot assume the dependencies are present.
    """
    import importlib.util

    missing = cli.check_dependencies(need_capture=False)
    for module, package in (("numpy", "numpy"), ("faster_whisper", "faster-whisper"),
                            ("soxr", "soxr")):
        really_absent = importlib.util.find_spec(module) is None
        check(f"probe agrees with reality for {package}",
              (package in missing) == really_absent,
              f"reported missing={package in missing}, actually absent={really_absent}")
    if missing:
        print(f"       (note: {', '.join(missing)} not installed here — expected on a bare box)")

    # PyAudioWPatch is Windows-only, so it must never be demanded elsewhere.
    missing_cap = cli.check_dependencies(need_capture=True)
    if not sys.platform.startswith("win"):
        check("loopback package not required off-Windows", "PyAudioWPatch" not in missing_cap,
              str(missing_cap))

    # Simulate a missing package by hiding it from the import system.
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "faster_whisper":
            return None
        return real_find_spec(name, *a, **k)

    importlib.util.find_spec = fake_find_spec
    try:
        simulated = cli.check_dependencies(need_capture=False)
        check("hidden package is detected", "faster-whisper" in simulated, str(simulated))
    finally:
        importlib.util.find_spec = real_find_spec

    # And the report must name every missing package plus the fix.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        cli._report_missing(["faster-whisper", "soxr"])
    report = buf.getvalue()
    check("report lists each missing package",
          "faster-whisper" in report and "soxr" in report, report)
    check("report gives the install command",
          "pip install -r requirements.txt" in report, report)


def test_model_failure_messages() -> None:
    """Regression: a missing package must NOT be reported as a network problem."""
    import contextlib
    import io

    def capture(exc: Exception, model: str = "small.en") -> str:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cli._report_model_load_failure(exc, model)
        return buf.getvalue()

    out = capture(ModuleNotFoundError("No module named 'faster_whisper'"))
    check("import error says to pip install",
          "pip install -r requirements.txt" in out, out)
    check("import error does NOT blame huggingface",
          "huggingface" not in out.lower(), out)

    out = capture(RuntimeError("httpx.ProxyError: 403 Forbidden"))
    check("proxy error blames the network", "huggingface.co" in out, out)
    check("proxy error offers the offline workaround", "--model" in out, out)

    out = capture(OSError("Connection timed out"))
    check("timeout is treated as network", "huggingface.co" in out, out)

    out = capture(ValueError("Invalid model size 'smal.en'"), model="smal.en")
    check("typo'd model name gets a size hint", "tiny.en" in out and "large-v3" in out, out)
    check("typo'd model name is echoed", "smal.en" in out, out)

    out = capture(RuntimeError("something entirely unexpected"))
    check("unknown error still gives a next step", "--model base.en" in out, out)

    for exc in (ModuleNotFoundError("x"), RuntimeError("proxy"), ValueError("invalid")):
        check(f"all messages name the model ({type(exc).__name__})",
              "small.en" in capture(exc), capture(exc))


def test_cli_help_and_exit_codes() -> None:
    root = Path(__file__).parent
    for cmd in (["--help"], ["record", "--help"], ["file", "--help"], ["notes", "--help"]):
        r = subprocess.run([sys.executable, "-m", "teams_notetaker", *cmd],
                           cwd=str(root), capture_output=True, text=True)
        check(f"`{' '.join(cmd)}` exits 0", r.returncode == 0, r.stderr[:300])

    r = subprocess.run([sys.executable, "-m", "teams_notetaker", "devices"],
                       cwd=str(root), capture_output=True, text=True)
    check("devices fails cleanly off-Windows", r.returncode == 2, f"rc={r.returncode}")
    check("devices prints a helpful error", "PyAudioWPatch" in r.stderr or "Windows" in r.stderr,
          r.stderr[:300])

    r = subprocess.run([sys.executable, "-m", "teams_notetaker", "notes", "/nope/missing.vtt"],
                       cwd=str(root), capture_output=True, text=True)
    check("missing transcript file exits 2", r.returncode == 2, f"rc={r.returncode}")

    # Exit code depends on what is installed: 4 = missing packages (checked
    # first), 2 = packages present but no capture device available. Both are
    # clean, actionable failures; a crash (1) or a hang is not.
    r = subprocess.run([sys.executable, "-m", "teams_notetaker", "record", "--no-mic", "--no-system"],
                       cwd=str(root), capture_output=True, text=True)
    check("record fails cleanly with a known exit code", r.returncode in (2, 4),
          f"rc={r.returncode} {r.stderr[:300]}")
    check("record failure explains itself",
          "pip install" in r.stderr or "Windows" in r.stderr or "PyAudioWPatch" in r.stderr,
          r.stderr[:300])
    check("record failure is not a traceback", "Traceback" not in r.stderr, r.stderr[:300])


def test_end_to_end_outputs() -> None:
    """Full offline pipeline: fake model -> transcript files on disk."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t, _ = _transcriber(tmp)
        speech = np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3
        silence = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)
        t.feed(TRACK_SYSTEM, speech)
        t.feed(TRACK_SYSTEM, silence)
        t.feed(TRACK_MIC, speech)
        t.drain(force=True)
        t.close()

        import os

        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            settings = t.settings
            settings.notes_backend = "auto"
            settings.local_base_url = "http://127.0.0.1:1/v1"  # nothing listening

            class A:
                context = "Weekly sync"

            cli._write_outputs(settings, tmp, t.utterances(), 5.0, A())
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

        for fname in ("transcript.md", "transcript.txt", "notes-prompt.md"):
            check(f"{fname} written", (tmp / fname).exists())
        check("notes.md NOT written without a backend", not (tmp / "notes.md").exists())
        body = (tmp / "transcript.txt").read_text()
        check("transcript.txt has both speakers",
              "Participants:" in body and "Me:" in body, body)
        prompt = (tmp / "notes-prompt.md").read_text()
        check("notes-prompt contains the transcript", "utterance 1" in prompt)
        check("notes-prompt contains the rules", "Never invent content" in prompt)
        check("notes-prompt carries the context", "Weekly sync" in prompt)


def test_end_to_end_with_local_model() -> None:
    """Full offline pipeline with a real local HTTP model server."""
    srv = _FakeLLMServer()
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            t, _ = _transcriber(tmp)
            t.feed(TRACK_MIC, np.ones(int(TARGET_SR * 2.5), dtype=np.float32) * 0.3)
            t.drain(force=True)
            t.close()

            settings = t.settings
            settings.notes_backend = "local"
            settings.local_base_url = srv.base_url
            settings.local_model = "llama3.1:8b"

            class A:
                context = None

            cli._write_outputs(settings, tmp, t.utterances(), 3.0, A())
            check("notes.md written via local model", (tmp / "notes.md").exists())
            notes = (tmp / "notes.md").read_text()
            check("notes.md has the header", notes.startswith("# T\n"), notes[:60])
            check("notes.md credits the local model", "llama3.1:8b" in notes, notes[:300])
            check("notes.md has model output", "## TL;DR" in notes, notes[:300])
    finally:
        srv.stop()


def main() -> int:
    print("=" * 70)
    print("teams-notetaker self-test (no audio hardware, no model download)")
    print("=" * 70)
    for fn in [
        test_compile,
        test_mono_conversion,
        test_resample,
        test_wav_writer,
        test_audio_unavailable_message,
        test_track_buffer,
        test_chunking_silence_cut,
        test_chunking_hard_cut,
        test_absolute_timestamps,
        test_tracks_are_independent,
        test_jsonl_roundtrip,
        test_hallucination_filter,
        test_merge_and_render,
        test_vtt_parsing,
        test_srt_parsing,
        test_jsonl_source,
        test_split_transcript,
        test_missing_api_key,
        test_paste_prompt,
        test_local_backend_happy_path,
        test_local_backend_url_normalization,
        test_local_backend_errors,
        test_local_map_reduce,
        test_backend_resolution,
        test_cli_parsing,
        test_dependency_check,
        test_model_failure_messages,
        test_cli_help_and_exit_codes,
        test_end_to_end_outputs,
        test_end_to_end_with_local_model,
    ]:
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            FAILURES.append(f"{fn.__name__} raised {exc!r}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
