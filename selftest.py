# Offline regression suite for transcribe.py. No pytest, no audio hardware,
# no model download — run with: python selftest.py
#
# Distinct from test_audio.py, which is an interactive real-hardware
# BlackHole smoke test.

import argparse
import datetime
import http.server
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np

import transcribe

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}" + (f" -- {detail}" if detail != "" else ""))
        FAILURES.append(name)


class FakeSegment:
    def __init__(self, text, avg_logprob, no_speech_prob, start=0.0, end=1.0, compression_ratio=1.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.start = start
        self.end = end
        self.compression_ratio = compression_ratio


class FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        return list(self._segments), None


def _raise(message):
    # Stand-in for argparse's parser.error, which exits the process.
    raise ValueError(message)


def _make_namespace(**overrides):
    # Mirrors what parse_args() hands to resolve_settings: preset-controlled
    # flags are None unless the user passed them.
    ns = argparse.Namespace(
        accuracy=None,
        model=None,
        beam_size=None,
        compute_type=None,
        cpu_threads=None,
        min_chunk_seconds=None,
        max_chunk_seconds=None,
        patience=None,
        language=transcribe.DEFAULT_LANGUAGE,
        initial_prompt="",
        input_file="",
        remote_url="",
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _make_fake_args(**overrides):
    # Deliberately goes through the REAL resolver rather than hand-rolling a
    # Namespace of defaults. A hand-rolled copy is exactly how tests drift away
    # from production precedence rules without failing.
    return transcribe.resolve_settings(_make_namespace(**overrides), _raise)


class _FakeWhisperServer:
    # Mirrors teams-notetaker/selftest.py's _FakeLLMServer pattern: a plain
    # http.server.HTTPServer on a background thread. Deliberately avoids
    # cgi.FieldStorage (removed in Python 3.13) -- multipart bodies are just
    # drained, never parsed, since these tests only need to control the
    # response side.
    def __init__(self):
        self.request_count = 0
        self.health_status = 200
        self.transcribe_responses = None
        self._response_index = 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(server.health_status)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                server.request_count += 1
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                status, body = server.next_response()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode("utf-8")
                elif isinstance(body, str):
                    body = body.encode("utf-8")
                self.wfile.write(body)

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def next_response(self):
        if not self.transcribe_responses:
            return 200, {"text": "", "language": "en", "duration": 0, "segments": []}
        index = min(self._response_index, len(self.transcribe_responses) - 1)
        self._response_index += 1
        return self.transcribe_responses[index]

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_compile():
    import py_compile

    root = os.path.dirname(os.path.abspath(__file__))
    try:
        py_compile.compile(os.path.join(root, "transcribe.py"), doraise=True)
        check("transcribe.py compiles", True)
    except py_compile.PyCompileError as exc:
        check("transcribe.py compiles", False, str(exc))


def test_positive_float():
    check("positive_float accepts 5", transcribe.positive_float("5") == 5.0)
    try:
        transcribe.positive_float("0")
        check("positive_float rejects 0", False, "did not raise")
    except argparse.ArgumentTypeError:
        check("positive_float rejects 0", True)
    try:
        transcribe.positive_float("-1")
        check("positive_float rejects negative", False, "did not raise")
    except argparse.ArgumentTypeError:
        check("positive_float rejects negative", True)


def test_normalize_remote_url():
    check("blank stays blank", transcribe.normalize_remote_url("") == "")
    check(
        "trailing slash stripped",
        transcribe.normalize_remote_url("http://host:8000/") == "http://host:8000",
    )
    try:
        transcribe.normalize_remote_url("not-a-url")
        check("invalid url rejected", False, "did not raise")
    except argparse.ArgumentTypeError:
        check("invalid url rejected", True)


def test_source_buffer_setdefault():
    buffers = {}
    buf = buffers.setdefault("totally-unexpected-label", transcribe.SourceBuffer())
    check("setdefault never raises KeyError for unexpected label", isinstance(buf, transcribe.SourceBuffer))
    buf.append(np.ones(100, dtype=np.float32), 1000.0)
    check("buffer accumulates", buf.seconds() > 0)
    check("start_time recorded on first append", buf.start_time == 1000.0)
    taken, start_time = buf.take_all()
    check("take_all returns everything then resets", len(taken) == 100 and buf.seconds() == 0)
    check("take_all returns the capture time", start_time == 1000.0)
    check("start_time cleared with the samples", buf.start_time is None)
    buf.append(np.ones(10, dtype=np.float32), 2000.0)
    check("start_time re-armed for the next chunk", buf.start_time == 2000.0)


def test_resample():
    src_rate = 48000
    audio = np.zeros(48000, dtype=np.float32)
    expected_len = round(len(audio) * transcribe.TARGET_SAMPLE_RATE / src_rate)

    out = transcribe.resample_to_target(audio, src_rate)
    check("resample output length approx correct", abs(len(out) - expected_len) <= 1, len(out))
    check("resample output dtype float32", out.dtype == np.float32)

    same = transcribe.resample_to_target(audio, transcribe.TARGET_SAMPLE_RATE)
    check("same-rate passthrough length unchanged", len(same) == len(audio))

    original_have_soxr = transcribe._HAVE_SOXR
    transcribe._HAVE_SOXR = False
    try:
        out_fallback = transcribe.resample_to_target(audio, src_rate)
        check(
            "np.interp fallback produces correct length",
            abs(len(out_fallback) - expected_len) <= 1,
            len(out_fallback),
        )
        check("np.interp fallback dtype float32", out_fallback.dtype == np.float32)
    finally:
        transcribe._HAVE_SOXR = original_have_soxr


def test_ready_source():
    buffers = {}
    buf = buffers.setdefault("A", transcribe.SourceBuffer())

    loud = np.ones(int(2 * transcribe.TARGET_SAMPLE_RATE), dtype=np.float32) * 0.5
    buf.append(loud)
    check(
        "not ready below min_chunk_seconds even if loud",
        transcribe.ready_source(buffers, min_chunk_seconds=6.0, max_chunk_seconds=20.0) is None,
    )

    buf.samples = np.concatenate(
        [
            np.ones(int(6.5 * transcribe.TARGET_SAMPLE_RATE), dtype=np.float32) * 0.5,
            np.zeros(int(1.0 * transcribe.TARGET_SAMPLE_RATE), dtype=np.float32),
        ]
    )
    check(
        "ready once min_chunk_seconds + quiet tail met",
        transcribe.ready_source(buffers, min_chunk_seconds=6.0, max_chunk_seconds=20.0) == "A",
    )

    buf.samples = np.ones(int(25 * transcribe.TARGET_SAMPLE_RATE), dtype=np.float32) * 0.5
    check(
        "hard cut at max_chunk_seconds regardless of loudness",
        transcribe.ready_source(buffers, min_chunk_seconds=6.0, max_chunk_seconds=20.0) == "A",
    )


def test_hallucination_filter():
    silent = FakeSegment("um", avg_logprob=-1.2, no_speech_prob=0.95)
    normal = FakeSegment("hello there", avg_logprob=-0.2, no_speech_prob=0.05)
    check("near-silent segment flagged as hallucination", transcribe.is_hallucination(silent))
    check("normal segment not flagged", not transcribe.is_hallucination(normal))

    # Repetition loop: confident (fine avg_logprob), real audio (low
    # no_speech_prob) -- only compression_ratio catches this one.
    repetition_loop = FakeSegment(
        "da da da da da", avg_logprob=-0.3, no_speech_prob=0.05, compression_ratio=6.0
    )
    check("repetition loop flagged via compression_ratio", transcribe.is_hallucination(repetition_loop))
    normal_low_ratio = FakeSegment(
        "hello there friend", avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.2
    )
    check("normal segment with low compression_ratio not flagged", not transcribe.is_hallucination(normal_low_ratio))

    remote_silent = transcribe.RemoteSegment(text="um", start=0, end=1, no_speech_prob=0.95, avg_logprob=-1.2)
    remote_normal = transcribe.RemoteSegment(text="hi", start=0, end=1, no_speech_prob=0.05, avg_logprob=-0.2)
    check("RemoteSegment parity: silent flagged", transcribe.is_hallucination(remote_silent))
    check("RemoteSegment parity: normal not flagged", not transcribe.is_hallucination(remote_normal))
    check(
        "RemoteSegment default compression_ratio (0.0) never trips the filter alone",
        not transcribe.is_hallucination(remote_normal) and remote_normal.compression_ratio == 0.0,
    )


def test_combined_initial_prompt():
    check("both empty -> None", transcribe.combined_initial_prompt(None, "") is None)
    check("only base", transcribe.combined_initial_prompt("base", "") == "base")
    check("only carry", transcribe.combined_initial_prompt(None, "carry") == "carry")
    check("both -> joined", transcribe.combined_initial_prompt("base", "carry") == "base carry")


def test_prepare_audio():
    silence = np.zeros(1000, dtype=np.float32)
    check("silence returns None", transcribe.prepare_audio(silence) is None)

    quiet = np.ones(1000, dtype=np.float32) * 0.01
    out = transcribe.prepare_audio(quiet)
    check("quiet audio is amplified", out is not None and float(np.abs(out[0])) > 0.01)
    rms_out = float(np.sqrt(np.mean(np.square(out))))
    expected_capped = 0.01 * transcribe.MAX_GAIN
    check(
        "quiet audio scaled toward TARGET_RMS or capped by MAX_GAIN",
        abs(rms_out - transcribe.TARGET_RMS) < 1e-4 or abs(rms_out - expected_capped) < 1e-4,
        rms_out,
    )

    loud = np.ones(1000, dtype=np.float32) * 0.5
    out_loud = transcribe.prepare_audio(loud)
    check("loud audio passes through unchanged", np.allclose(out_loud, loud))


def test_worker_error_signaling():
    model = FakeModel([])
    q = queue.Queue()
    q.put((transcribe.SOURCE_YOU, "not-an-array", 0.0))  # .astype() will raise AttributeError
    worker_error = threading.Event()
    args = _make_fake_args()
    with tempfile.TemporaryDirectory() as tmp:
        output_file = os.path.join(tmp, "out.txt")
        source_rates = {
            transcribe.SOURCE_YOU: transcribe.TARGET_SAMPLE_RATE,
            transcribe.SOURCE_OTHERS: transcribe.TARGET_SAMPLE_RATE,
        }
        thread = threading.Thread(
            target=transcribe.transcribe_loop,
            args=(model, q, output_file, args, source_rates, worker_error),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        check("worker thread does not hang on error", not thread.is_alive())
        check("worker_error is set after a crash", worker_error.is_set())


def test_check_remote_health():
    server = _FakeWhisperServer()
    try:
        transcribe.check_remote_health(server.url)
        check("healthy server passes", True)
    finally:
        server.stop()

    try:
        transcribe.check_remote_health("http://127.0.0.1:1")
        check("closed/unreachable port raises", False, "did not raise")
    except RuntimeError:
        check("closed/unreachable port raises", True)


def test_check_remote_health_non200():
    server = _FakeWhisperServer()
    server.health_status = 503
    try:
        try:
            transcribe.check_remote_health(server.url)
            check("non-200 health raises", False, "did not raise")
        except RuntimeError:
            check("non-200 health raises", True)
    finally:
        server.stop()


def test_transcribe_chunk_remote_happy_path():
    server = _FakeWhisperServer()
    server.transcribe_responses = [
        (
            200,
            {
                "text": "hello",
                "language": "en",
                "duration": 1.0,
                "segments": [
                    {"text": "hello", "start": 0.0, "end": 1.0, "no_speech_prob": 0.01, "avg_logprob": -0.1}
                ],
            },
        )
    ]
    try:
        audio = np.zeros(transcribe.TARGET_SAMPLE_RATE, dtype=np.float32)
        settings = _make_fake_args(remote_url=server.url)
        segments = transcribe.transcribe_chunk_remote(audio, settings, None)
        check("remote happy path returns 1 segment", len(segments) == 1, segments)
        check("remote segment text", segments[0].text == "hello")
        check("remote segment no_speech_prob", segments[0].no_speech_prob == 0.01)
    finally:
        server.stop()


def test_transcribe_chunk_remote_retry_then_recover():
    server = _FakeWhisperServer()
    server.transcribe_responses = [
        (500, {"error": "boom"}),
        (200, {"segments": [{"text": "ok", "start": 0, "end": 1, "no_speech_prob": 0.0, "avg_logprob": 0.0}]}),
    ]
    try:
        audio = np.zeros(1000, dtype=np.float32)
        segments = transcribe.transcribe_chunk_remote(audio, _make_fake_args(remote_url=server.url), None)
        check("retry then recover succeeds", len(segments) == 1)
        check("retry then recover made 2 requests", server.request_count == 2, server.request_count)
    finally:
        server.stop()


def test_transcribe_chunk_remote_exhausted():
    server = _FakeWhisperServer()
    server.transcribe_responses = [(500, {"error": "boom"})] * (transcribe.REMOTE_MAX_RETRIES + 3)
    try:
        audio = np.zeros(1000, dtype=np.float32)
        try:
            transcribe.transcribe_chunk_remote(audio, _make_fake_args(remote_url=server.url), None)
            check("exhausted retries raises", False, "did not raise")
        except transcribe.RemoteTranscriptionError:
            check("exhausted retries raises", True)
        check(
            "exhausted retries makes exactly max+1 attempts",
            server.request_count == transcribe.REMOTE_MAX_RETRIES + 1,
            server.request_count,
        )
    finally:
        server.stop()


def test_transcribe_chunk_remote_malformed():
    server = _FakeWhisperServer()
    server.transcribe_responses = [(200, {"text": "no segments key"})] * (transcribe.REMOTE_MAX_RETRIES + 3)
    try:
        audio = np.zeros(1000, dtype=np.float32)
        try:
            transcribe.transcribe_chunk_remote(audio, _make_fake_args(remote_url=server.url), None)
            check("missing segments key raises", False, "did not raise")
        except transcribe.RemoteTranscriptionError:
            check("missing segments key raises", True)
    finally:
        server.stop()


def test_transcribe_chunk_remote_missing_fields():
    server = _FakeWhisperServer()
    server.transcribe_responses = [(200, {"segments": [{"text": "no probs here", "start": 0, "end": 1}]})]
    try:
        audio = np.zeros(1000, dtype=np.float32)
        segments = transcribe.transcribe_chunk_remote(audio, _make_fake_args(remote_url=server.url), None)
        check("missing fields default no_speech_prob", segments[0].no_speech_prob == 0.0)
        check("missing fields default avg_logprob", segments[0].avg_logprob == 0.0)
        check("missing-field segment is not treated as hallucination", not transcribe.is_hallucination(segments[0]))
    finally:
        server.stop()


def test_remote_failure_limit():
    check(
        "below limit not reached",
        not transcribe.remote_failure_limit_reached(transcribe.REMOTE_CONSECUTIVE_FAILURE_LIMIT - 1),
    )
    check(
        "at limit reached",
        transcribe.remote_failure_limit_reached(transcribe.REMOTE_CONSECUTIVE_FAILURE_LIMIT),
    )


def test_cli_help_and_validation():
    root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(root, "transcribe.py")

    result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
    check("--help exits 0", result.returncode == 0)
    check("--help mentions --remote-url", "--remote-url" in result.stdout)
    check("--help mentions --min-chunk-seconds", "--min-chunk-seconds" in result.stdout)
    check("--chunk-seconds is gone", "--chunk-seconds" not in result.stdout)

    result = subprocess.run(
        [sys.executable, script, "--min-chunk-seconds", "20", "--max-chunk-seconds", "5"],
        capture_output=True,
        text=True,
    )
    check("min>max exits nonzero", result.returncode != 0)
    check("min>max error message", "cannot be greater than" in result.stderr, result.stderr)

    result = subprocess.run(
        [sys.executable, script, "--min-chunk-seconds", "0"],
        capture_output=True,
        text=True,
    )
    check("min-chunk-seconds 0 exits nonzero", result.returncode != 0)

    result = subprocess.run(
        [sys.executable, script, "--remote-url", "not-a-url"],
        capture_output=True,
        text=True,
    )
    check("invalid remote-url exits nonzero", result.returncode != 0)

    result = subprocess.run(
        [sys.executable, script, "--remote-url", "http://192.168.77.1:8000", "--input-file", "y.wav"],
        capture_output=True,
        text=True,
    )
    check("remote-url + input-file mutual exclusion exits nonzero", result.returncode != 0)
    check(
        "mutual exclusion error message",
        "not supported with --input-file" in result.stderr,
        result.stderr,
    )


def test_accuracy_presets():
    for name, preset in transcribe.ACCURACY_PRESETS.items():
        check(f"{name}: min <= max", preset.min_chunk_seconds <= preset.max_chunk_seconds)
        # The invariant behind the whole chunk-length argument: Whisper pads
        # every window to 30s, so a longer chunk buys a second encoder pass.
        check(
            f"{name}: max chunk fits one Whisper window",
            preset.max_chunk_seconds <= transcribe.WHISPER_WINDOW_SECONDS,
        )
        check(f"{name}: beam >= 1", preset.beam_size >= 1)
        # hallucination_silence_threshold is only read inside
        # `if options.word_timestamps`, so it is dead weight without it.
        if preset.hallucination_silence_threshold is not None:
            check(f"{name}: hallucination threshold implies word timestamps", preset.word_timestamps)

    fast = transcribe.ACCURACY_PRESETS["fast"]
    check("fast preset is the pre-existing behaviour", (fast.model, fast.beam_size, fast.min_chunk_seconds, fast.max_chunk_seconds) == ("small.en", 1, 6.0, 20.0))
    check("DEFAULT_MODEL_SIZE tracks the default preset", transcribe.DEFAULT_MODEL_SIZE == transcribe.ACCURACY_PRESETS[transcribe.DEFAULT_ACCURACY].model)
    check("DEFAULT_BEAM_SIZE tracks the default preset", transcribe.DEFAULT_BEAM_SIZE == transcribe.ACCURACY_PRESETS[transcribe.DEFAULT_ACCURACY].beam_size)


def test_settings_precedence():
    balanced = transcribe.ACCURACY_PRESETS["balanced"]

    settings = _make_fake_args(accuracy="balanced")
    check("preset supplies the model", settings.model == balanced.model)
    check("preset use is not marked explicit", "model" not in settings.explicit)

    settings = _make_fake_args(accuracy="balanced", beam_size=2)
    check("explicit flag beats preset", settings.beam_size == 2)
    check("explicit flag is recorded", "beam_size" in settings.explicit)

    os.environ["WHISPER_BEAM_SIZE"] = "3"
    try:
        settings = _make_fake_args(accuracy="balanced")
        check("env var beats preset", settings.beam_size == 3)
        check("env var counts as explicit", "beam_size" in settings.explicit)
        settings = _make_fake_args(accuracy="balanced", beam_size=4)
        check("explicit flag beats env var", settings.beam_size == 4)
    finally:
        del os.environ["WHISPER_BEAM_SIZE"]

    # The regression the old DEFAULT_* comparison could not express: a preset
    # changing the model must NOT read as the user overriding it.
    settings = _make_fake_args(accuracy="max", remote_url="http://example.invalid:8000")
    check("preset-supplied model is not 'ignored in remote mode'", not (transcribe.REMOTE_IGNORED_FLAGS & settings.explicit))
    settings = _make_fake_args(model="tiny.en", remote_url="http://example.invalid:8000")
    check("explicit model is 'ignored in remote mode'", transcribe.REMOTE_IGNORED_FLAGS & settings.explicit == {"model"})


def test_settings_validation():
    for label, kwargs, fragment in [
        ("unknown preset", dict(accuracy="turbo"), "unknown --accuracy"),
        ("resolved min > max", dict(accuracy="fast", min_chunk_seconds=25.0), "cannot be greater"),
        ("chunk beyond the window", dict(max_chunk_seconds=45.0), "Whisper's"),
        ("bad compute type", dict(compute_type="int4"), "not supported"),
        ("bad beam", dict(beam_size=0), "at least 1"),
        ("bad threads", dict(cpu_threads=0), "at least 1"),
    ]:
        try:
            _make_fake_args(**kwargs)
            check(label, False, "did not raise")
        except ValueError as exc:
            check(label, fragment in str(exc), f"message was {exc}")


def test_decode_options_shared():
    settings = _make_fake_args(accuracy="balanced")
    live = transcribe.build_decode_options(settings, "prompt")
    same = transcribe.build_decode_options(settings, "prompt")
    check("one builder, one result", live == same)
    check("preset beam reaches the decoder", live["beam_size"] == transcribe.ACCURACY_PRESETS["balanced"].beam_size)
    check("VAD parameters are passed explicitly", live["vad_parameters"] == transcribe.VAD_PARAMETERS)
    check("VAD dict is a copy, not the shared constant", live["vad_parameters"] is not transcribe.VAD_PARAMETERS)
    check("chunks decode independently", live["condition_on_previous_text"] is False)
    # Deliberate omissions, each with a specific reason -- see build_decode_options.
    check("chunk_length is never set", "chunk_length" not in live)
    check("length_penalty is never set", "length_penalty" not in live)
    check("no_repeat_ngram_size is never set", "no_repeat_ngram_size" not in live)


def test_decode_options_reach_the_model():
    class RecordingModel:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio, **kwargs):
            self.kwargs = kwargs
            return [], None

    model = RecordingModel()
    settings = _make_fake_args(accuracy="max")
    transcribe.transcribe_chunk_local(model, np.zeros(16000, dtype=np.float32), settings, "vocab")
    check("kwargs recorded", model.kwargs is not None)
    check("beam size arrives", model.kwargs["beam_size"] == 5)
    check("initial prompt arrives", model.kwargs["initial_prompt"] == "vocab")
    check("word timestamps arrive", model.kwargs["word_timestamps"] is True)
    check("hallucination silence threshold arrives", model.kwargs["hallucination_silence_threshold"] == 2.0)


def test_remote_fields_projected_from_options():
    settings = _make_fake_args(remote_url="http://example.invalid:8000")
    options = transcribe.build_decode_options(settings, "vocab")
    for name in transcribe.REMOTE_FIELDS:
        check(f"remote field {name} exists in the shared options", name in options)
    check("local-only knobs stay local", "compute_type" not in transcribe.REMOTE_FIELDS)


def test_line_timestamp_uses_capture_time():
    captured = 1_700_000_000.0
    expected = datetime.datetime.fromtimestamp(captured + 4.0).strftime("%H:%M:%S")
    # The point of the change: the answer must not depend on when this runs,
    # so a 30s processing delay cannot move the timestamp.
    check("timestamp is capture time + segment offset", transcribe.line_timestamp(captured, 4.0) == expected)
    check("negative offsets are clamped", transcribe.line_timestamp(captured, -5.0) == datetime.datetime.fromtimestamp(captured).strftime("%H:%M:%S"))
    check("missing clock still produces a timestamp", len(transcribe.line_timestamp(None, 0.0)) == 8)


def main():
    print("=" * 70)
    print("transcribe.py self-test (no audio hardware, no model download)")
    print("=" * 70)

    tests = [
        test_compile,
        test_positive_float,
        test_normalize_remote_url,
        test_source_buffer_setdefault,
        test_resample,
        test_ready_source,
        test_hallucination_filter,
        test_combined_initial_prompt,
        test_prepare_audio,
        test_worker_error_signaling,
        test_check_remote_health,
        test_check_remote_health_non200,
        test_transcribe_chunk_remote_happy_path,
        test_transcribe_chunk_remote_retry_then_recover,
        test_transcribe_chunk_remote_exhausted,
        test_transcribe_chunk_remote_malformed,
        test_transcribe_chunk_remote_missing_fields,
        test_remote_failure_limit,
        test_cli_help_and_validation,
        test_accuracy_presets,
        test_settings_precedence,
        test_settings_validation,
        test_decode_options_shared,
        test_decode_options_reach_the_model,
        test_remote_fields_projected_from_options,
        test_line_timestamp_uses_capture_time,
    ]

    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - surface as a failed test, keep going
            check(test.__name__, False, f"raised {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
