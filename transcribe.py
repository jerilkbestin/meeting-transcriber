# Dual-stream local meeting transcriber: captures your mic ("You") and the
# meeting audio routed through BlackHole ("Others"), runs both through
# faster-whisper on CPU (or a remote GPU server via --remote-url), and writes
# timestamped lines to a .txt transcript.
# Fully local by default — no audio or text leaves the machine unless
# --remote-url is explicitly set.

import argparse
import datetime
import io
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from typing import NamedTuple, Optional

import ctranslate2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    import soxr

    _HAVE_SOXR = True
except ImportError:  # optional dependency; resample_to_target falls back to np.interp
    soxr = None
    _HAVE_SOXR = False


# 16 kHz mono is the sample rate Whisper's encoder is trained on. Devices are
# opened at their own native rate (see device_sample_rate) and resampled down
# to this target (see resample_to_target) rather than assuming a device can
# natively provide 16 kHz.
TARGET_SAMPLE_RATE = 16000
# Whisper's encoder input is a fixed 30s mel window. faster_whisper's
# pad_or_trim() pads every window to it regardless of how much audio the chunk
# actually holds, which is why chunk length -- not chunk content -- sets the
# cost per decode call. Chunks must stay at or under this to avoid paying for a
# second padded pass.
WHISPER_WINDOW_SECONDS = 30.0
BLACKHOLE_NAME = "BlackHole 2ch"
TRANSCRIPTS_DIR = "transcripts"
DEFAULT_LANGUAGE = "en"


class AccuracyPreset(NamedTuple):
    # One row of ACCURACY_PRESETS. Grouped as a preset rather than as loose
    # constants because these knobs only make sense together: a bigger model
    # is only affordable if the chunk length rises with it (see F1 below).
    model: str
    beam_size: int
    compute_type: str
    cpu_threads: int
    min_chunk_seconds: float
    max_chunk_seconds: float
    patience: float
    word_timestamps: bool
    hallucination_silence_threshold: Optional[float]


# --accuracy presets. Rationale for the values:
#
#   beam_size 1 -> 5: greedy decoding commits to the highest-probability token
#     at every step, which is exactly how a rare proper noun ("ABAC", "Seaspan")
#     loses to a common word that scores better on the first token. Beam search
#     keeps 5 hypotheses alive long enough for later acoustic evidence to
#     rescue the rare one. faster-whisper's own upstream default is 5; the 1
#     here was a speed choice.
#
#   chunk 6/20 -> 12/28: faster_whisper/audio.py pad_or_trim() pads EVERY
#     window to 3000 frames (30s) before the encoder runs, so a 6s chunk costs
#     about what a 26s chunk costs. Decode cost per minute of meeting is
#     therefore proportional to the number of calls, i.e. inversely
#     proportional to chunk length -- doubling the chunk halves the cost and is
#     what funds a bigger model. 28s keeps every chunk inside a single padded
#     window (VAD only ever shortens audio), so it never spills to a second
#     encoder pass. Paid for in on-screen latency, not in timestamp accuracy
#     (see capture-time stamping in transcribe_loop).
#
#   compute_type: ctranslate2.get_supported_compute_types("cpu") on this arm64
#     Mac returns {float32, int8_float32, int8} -- no fp16/bf16, and int8 and
#     int8_float32 are the same thing here. MEASURED, and the opposite of the
#     usual expectation: on large-v3-turbo, float32 is both more accurate AND
#     FASTER than int8 on this chip (RTF 0.40 vs 0.45 cold, 0.50 vs 0.70 warm,
#     reproduced back-to-back). int8's CPU GEMM path is not the fast one on
#     arm64 here. small.en keeps int8 because it is already far inside budget
#     and int8 costs a quarter of the memory.
#
#   cpu_threads: CTranslate2 uses an OpenMP pool with per-GEMM barriers, so the
#     slowest thread gates each matmul. On a 4P+6E chip, threads landing on
#     efficiency cores drag the whole pass toward E-core pace. 4 == the
#     performance-core count.
ACCURACY_PRESETS = {
    "fast": AccuracyPreset(
        model="small.en",
        beam_size=1,
        compute_type="int8",
        cpu_threads=4,
        min_chunk_seconds=6.0,
        max_chunk_seconds=20.0,
        patience=1.0,
        word_timestamps=False,
        hallucination_silence_threshold=None,
    ),
    # Measured on this M4 (both sources replayed, warm): RTF 0.27, i.e. ~3.7x
    # real-time headroom. Chosen over medium.en/beam5 (RTF 0.35 and a weaker
    # model) and over turbo/beam5 (RTF 0.50 warm, only 2x headroom -- too thin
    # once Teams and screen share are also running).
    "balanced": AccuracyPreset(
        model="large-v3-turbo",
        beam_size=2,
        compute_type="float32",
        cpu_threads=4,
        min_chunk_seconds=12.0,
        max_chunk_seconds=28.0,
        patience=1.0,
        word_timestamps=False,
        hallucination_silence_threshold=None,
    ),
    # Intended for --input-file, where there is no real-time constraint.
    # Measured RTF ~0.50 warm at beam 5, which is at the edge of what live
    # capture can sustain.
    "max": AccuracyPreset(
        model="large-v3-turbo",
        beam_size=5,
        compute_type="float32",
        cpu_threads=4,
        min_chunk_seconds=18.0,
        max_chunk_seconds=28.0,
        patience=1.0,
        # word_timestamps is a precondition: faster-whisper only consults
        # hallucination_silence_threshold inside `if options.word_timestamps`.
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
    ),
}
ACCURACY_CHOICES = tuple(ACCURACY_PRESETS)
DEFAULT_ACCURACY = "balanced"

# Kept as names because the rest of the file, the README and selftest.py all
# refer to "the default model/beam". They now derive from the default preset
# instead of being a second, independent source of truth.
DEFAULT_MODEL_SIZE = ACCURACY_PRESETS[DEFAULT_ACCURACY].model
DEFAULT_BEAM_SIZE = ACCURACY_PRESETS[DEFAULT_ACCURACY].beam_size

# Temperature fallback list (faster-whisper's own upstream default) instead of
# a fixed temperature=0. When compression_ratio_threshold or log_prob_threshold
# rejects the temp=0 decode as too repetitive/low-confidence, faster-whisper
# retries at the next temperature here instead of just accepting the bad
# result -- this is its own built-in repetition-loop escape hatch, which a
# single fixed temperature disables (verified against faster_whisper's decode
# loop: with only one temperature, the reject-and-retry has nowhere to go).
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_BEST_OF = 5  # only used once a fallback temperature > 0 is tried
SOURCE_YOU = "You"
SOURCE_OTHERS = "Others"

# Adaptive, silence-aware chunking (replaces the old fixed chunk_seconds): a
# source is transcribed once it has at least DEFAULT_MIN_CHUNK_SECONDS AND a
# quiet tail, or unconditionally once it hits DEFAULT_MAX_CHUNK_SECONDS.
DEFAULT_MIN_CHUNK_SECONDS = ACCURACY_PRESETS[DEFAULT_ACCURACY].min_chunk_seconds
DEFAULT_MAX_CHUNK_SECONDS = ACCURACY_PRESETS[DEFAULT_ACCURACY].max_chunk_seconds
SILENCE_WINDOW_SECONDS = 0.6
CHUNK_SILENCE_RMS = 0.006  # cut-point detector, distinct from SILENCE_RMS below
MIN_READY_SECONDS = 0.25
# Warn once buffered audio exceeds this many max-chunks. Low enough that a
# preset which cannot sustain real time says so within the first minute.
BACKLOG_WARNING_MULTIPLIER = 2

# Hallucination filter: two distinct faster-whisper failure modes, two
# distinct signals. no_speech_prob+avg_logprob catches "invented text on
# silence" (low no_speech_prob would mean real audio was present, so both
# must hold). compression_ratio catches a *confident* repetition loop
# ("da da da da..."), which can have a fine avg_logprob and a low
# no_speech_prob (real audio triggered it) -- the AND-only check above
# does not catch it, so this is a separate, independent check.
HALLUCINATION_NO_SPEECH_PROB = 0.85
HALLUCINATION_MAX_AVG_LOGPROB = -0.9
HALLUCINATION_COMPRESSION_RATIO = 2.4  # faster-whisper's own default threshold

# Silero VAD tuning. Deliberately CONSERVATIVE: because every window is padded
# to 30s regardless (see the chunk note in ACCURACY_PRESETS), stripping more
# silence saves no compute in live mode -- aggressive VAD has zero upside and
# one real downside, deleting speech, which is the worst error class because it
# leaves no trace in the output. So: a lower threshold keeps quiet/far-mic
# speech, and a longer speech_pad_ms stops onset/coda clipping at chunk
# boundaries, which is exactly where the adaptive chunker cuts.
VAD_PARAMETERS = {
    "threshold": 0.45,          # upstream 0.5
    "min_silence_duration_ms": 1000,  # upstream 2000
    "speech_pad_ms": 600,       # upstream 400
}

# Per-source rolling context: the tail of each chunk's kept text is carried
# forward as part of the next chunk's initial_prompt for the same source.
CARRY_PROMPT_MAX_CHARS = 220

# Silence-gate + auto-gain-control (AGC) knobs, tuned by ear (see prepare_audio):
# chunks quieter than SILENCE_RMS are dropped; louder-but-quiet chunks are
# amplified toward TARGET_RMS, but never by more than MAX_GAIN.
SILENCE_RMS = 0.0015
TARGET_RMS = 0.05
MAX_GAIN = 8.0

# Remote (GPU) inference. Base URL only; paths appended internally. Left out
# of DEFAULT_MODEL_SIZE/DEFAULT_BEAM_SIZE's flag surface since a remote
# server controls its own model.
DEFAULT_REMOTE_URL = ""
REMOTE_HEALTH_PATH = "/health"
REMOTE_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
REMOTE_HEALTH_TIMEOUT_SECONDS = 5.0
REMOTE_CHUNK_TIMEOUT_SECONDS = 20.0
REMOTE_MAX_RETRIES = 2
REMOTE_RETRY_BACKOFF_SECONDS = (1.0, 2.0)
REMOTE_CONSECUTIVE_FAILURE_LIMIT = 5
# Which build_decode_options keys travel to a remote server. The rest are
# local-execution details (compute type, VAD, word timestamps) that the server
# decides for itself.
REMOTE_FIELDS = ("language", "initial_prompt", "beam_size")
# Flags that describe local execution only. Setting one explicitly while
# --remote-url is in play earns a note, because the server decides these.
REMOTE_IGNORED_FLAGS = frozenset({"model", "beam_size", "compute_type", "cpu_threads"})

_remote_field_warning_shown = False


def sanitize_filename_component(value):
    # User-supplied text ends up in a filesystem path, so strip anything that
    # isn't a safe filename char (blocks path traversal / invalid chars).
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value


def output_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = input("\nTranscript name (blank for 'transcript'): ")
    prefix = sanitize_filename_component(name) or "transcript"
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    return os.path.join(TRANSCRIPTS_DIR, f"{prefix}_{timestamp}.txt")


def list_devices(devices):
    print("\nAll Available Audio Devices:")
    print("-" * 60)
    for i, device in enumerate(devices):
        ins = f"in: {device['max_input_channels']}ch" if device["max_input_channels"] > 0 else "no input"
        outs = f"out: {device['max_output_channels']}ch" if device["max_output_channels"] > 0 else "no output"
        print(f"  [{i}] {device['name']} ({ins}, {outs})")
    print("-" * 60)


def input_device_indices(devices):
    return [
        i for i, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]


def find_blackhole_input(devices):
    # Auto-detect the remote/meeting-audio device by name (Phase 2 design: the
    # user never picks an output device manually — routing is handled outside,
    # by switch_meeting_output). Missing BlackHole is a hard setup error.
    matches = [
        i for i, device in enumerate(devices)
        if "blackhole" in device["name"].lower()
        and device["max_input_channels"] > 0
    ]

    if not matches:
        raise RuntimeError(
            f"{BLACKHOLE_NAME} was not found as an input device. "
            "Confirm BlackHole is installed and visible in Audio MIDI Setup."
        )

    if len(matches) == 1:
        return matches[0]

    print("\nMultiple BlackHole input devices found:")
    for i in matches:
        print(f"  [{i}] {devices[i]['name']}")

    while True:
        value = input("\nEnter BlackHole input device number: ").strip()
        if value.isdigit() and int(value) in matches:
            return int(value)
        print("Invalid BlackHole input device number.")


def choose_mic_input(devices, blackhole_index):
    print("\nINPUT DEVICES (choose your mic for 'You'):")
    valid = input_device_indices(devices)
    for i in valid:
        device = devices[i]
        marker = " (BlackHole remote audio)" if i == blackhole_index else ""
        print(f"  [{i}] {device['name']}{marker}")

    while True:
        value = input("\nEnter MIC input device number: ").strip()
        if value.isdigit() and int(value) in valid:
            selected = int(value)
            if selected == blackhole_index:
                print("Choose a real microphone, not the BlackHole capture device.")
                continue
            return selected
        print("Invalid input device number.")


def stream_channels(devices, index):
    return min(int(devices[index]["max_input_channels"]), 2)


def device_sample_rate(devices, index):
    # Read the device's own native rate (AUDIT #5) rather than assuming 16 kHz
    # is supported; resample_to_target() converts down to TARGET_SAMPLE_RATE.
    rate = int(round(devices[index].get("default_samplerate") or 0))
    if rate <= 0:
        print(
            f"\nWarning: could not read a sample rate for {devices[index]['name']}; "
            f"falling back to {TARGET_SAMPLE_RATE} Hz."
        )
        return TARGET_SAMPLE_RATE
    return rate


def resample_to_target(samples, src_rate):
    if src_rate == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    if _HAVE_SOXR:
        return soxr.resample(samples, src_rate, TARGET_SAMPLE_RATE).astype(np.float32, copy=False)
    # Dependency-free fallback: linear interpolation. Lower quality than soxr
    # but keeps the tool installable without it.
    ratio = TARGET_SAMPLE_RATE / float(src_rate)
    n_out = int(round(len(samples) * ratio))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def make_callback(source, audio_queue):
    # Returns the function sounddevice/PortAudio calls on its real-time audio
    # thread for each incoming block. This thread must stay fast and must not
    # allocate/block beyond the unavoidable copy (AUDIT #1): downmixing to
    # mono and resampling happen later, in transcribe_loop, off this thread.
    # `source` and `audio_queue` are captured in the closure.
    def callback(indata, frames, time_info, status):
        if status:
            # Non-empty status = input overflow/underflow reported by PortAudio.
            print(f"\n{source} stream status: {status}")
        # copy() is required: PortAudio reuses indata's underlying buffer for
        # the next block, so we must not hold a reference to it.
        #
        # The wall clock is read HERE, not at write time. Timestamps must
        # describe when audio was captured; stamping them when the line is
        # finally written would bake in the queue backlog, so every timestamp
        # would be wrong by however far decoding is behind. time.time() is a
        # vDSO read -- no allocation, no lock, cheaper than the copy above.
        audio_queue.put((source, indata.copy(), time.time()))

    return callback


def positive_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {parsed}")
    return parsed


def normalize_remote_url(value):
    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            f"expected a URL like http://host:port, got {value!r}"
        )
    return value


class Settings(NamedTuple):
    # Fully resolved run configuration. Every consumer reads this instead of
    # the raw argparse Namespace, so there is exactly one place where "what is
    # the beam size" is answered.
    #
    # `explicit` holds the names the user actually set (flag or env var). That
    # is what makes the remote-mode "ignored" note correct: comparing a value
    # against DEFAULT_* cannot distinguish "user typed the default" from
    # "preset happens to supply this", and silently breaks whenever a default
    # moves.
    accuracy: str
    model: str
    beam_size: int
    compute_type: str
    cpu_threads: int
    min_chunk_seconds: float
    max_chunk_seconds: float
    patience: float
    word_timestamps: bool
    hallucination_silence_threshold: Optional[float]
    language: str
    initial_prompt: str
    input_file: str
    remote_url: str
    explicit: frozenset


# Preset-overridable flags, mapped to the env var that also sets them and the
# coercion applied to a string value. Order is the display order.
PRESET_FLAGS = (
    ("model", "WHISPER_MODEL", str),
    ("beam_size", "WHISPER_BEAM_SIZE", int),
    ("compute_type", "WHISPER_COMPUTE_TYPE", str),
    ("cpu_threads", "WHISPER_CPU_THREADS", int),
    ("min_chunk_seconds", "WHISPER_MIN_CHUNK_SECONDS", float),
    ("max_chunk_seconds", "WHISPER_MAX_CHUNK_SECONDS", float),
    ("patience", "WHISPER_PATIENCE", float),
)


def supported_compute_types():
    # Queried rather than hard-coded: the answer is build- and CPU-specific.
    # On this arm64 Mac it is {float32, int8_float32, int8} -- no fp16/bf16.
    try:
        return sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception:  # noqa: BLE001 - never let a probe failure stop a meeting
        return ["int8", "int8_float32", "float32"]


def resolve_settings(args, fail):
    """Collapse preset + env + explicit flags into one Settings.

    Precedence, highest first: explicit CLI flag, env var, preset, hard
    default. `fail` is a callable taking a message (parser.error in the CLI,
    a raising stub in tests) so this stays usable without an ArgumentParser.
    """
    accuracy = args.accuracy or os.getenv("WHISPER_ACCURACY") or DEFAULT_ACCURACY
    if accuracy not in ACCURACY_PRESETS:
        fail(
            f"unknown --accuracy {accuracy!r}; choose from "
            f"{', '.join(ACCURACY_CHOICES)}"
        )
    preset = ACCURACY_PRESETS[accuracy]

    values = {}
    explicit = set()
    for name, env_name, coerce in PRESET_FLAGS:
        flag_value = getattr(args, name, None)
        if flag_value is not None:
            values[name] = flag_value
            explicit.add(name)
            continue
        env_value = os.getenv(env_name)
        if env_value is not None and env_value.strip() != "":
            try:
                values[name] = coerce(env_value)
            except ValueError:
                fail(f"{env_name}={env_value!r} is not a valid value for --{name.replace('_', '-')}")
            explicit.add(name)
            continue
        values[name] = getattr(preset, name)

    # Cross-checks run on RESOLVED values, so "--min-chunk-seconds 25" against a
    # preset whose max is 20 still errors instead of silently never cutting.
    if values["min_chunk_seconds"] > values["max_chunk_seconds"]:
        fail(
            f"--min-chunk-seconds ({values['min_chunk_seconds']}) cannot be greater than "
            f"--max-chunk-seconds ({values['max_chunk_seconds']})"
        )
    if values["max_chunk_seconds"] > WHISPER_WINDOW_SECONDS:
        fail(
            f"--max-chunk-seconds ({values['max_chunk_seconds']}) exceeds Whisper's "
            f"{WHISPER_WINDOW_SECONDS:.0f}s window; a longer chunk costs a second "
            "padded encoder pass for the overflow and buys nothing"
        )
    if values["beam_size"] < 1:
        fail(f"--beam-size must be at least 1, got {values['beam_size']}")
    if values["cpu_threads"] < 1:
        fail(f"--cpu-threads must be at least 1, got {values['cpu_threads']}")
    supported = supported_compute_types()
    if values["compute_type"] not in supported:
        fail(
            f"--compute-type {values['compute_type']!r} is not supported on this CPU; "
            f"available: {', '.join(supported)}"
        )

    if args.remote_url and args.input_file.strip():
        fail(
            "--remote-url is not supported with --input-file yet; remote "
            "inference is live-capture only for now"
        )

    return Settings(
        accuracy=accuracy,
        language=args.language,
        initial_prompt=args.initial_prompt,
        input_file=args.input_file,
        remote_url=args.remote_url,
        word_timestamps=preset.word_timestamps,
        hallucination_silence_threshold=preset.hallucination_silence_threshold,
        explicit=frozenset(explicit),
        **values,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe mic audio plus BlackHole meeting audio."
    )
    # Every preset-overridable flag defaults to None, not to its value. None is
    # the only reliable "the user did not pass this" signal, and it is what
    # lets a preset supply the value while an explicit flag still wins. The env
    # fallbacks live in resolve_settings for the same reason.
    parser.add_argument(
        "--accuracy",
        choices=ACCURACY_CHOICES,
        default=None,
        help="accuracy/speed preset setting model, beam size, chunk length and "
        f"compute type together. Individual flags override it. Default: {DEFAULT_ACCURACY}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="faster-whisper model to use. Overrides the --accuracy preset.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=positive_float,
        default=None,
        help="minimum seconds of audio before a silence cut is considered.",
    )
    parser.add_argument(
        "--max-chunk-seconds",
        type=positive_float,
        default=None,
        help="hard ceiling on seconds per decode, even mid-sentence.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="beam size. 1 is greedy and fastest; 5 is faster-whisper's own default "
        "and materially better on proper nouns.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2 compute type. Validated against this CPU's supported list "
        f"({', '.join(supported_compute_types())}).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="CTranslate2 CPU threads. More is not always faster on a chip with "
        "efficiency cores.",
    )
    parser.add_argument(
        "--patience",
        type=positive_float,
        default=None,
        help="beam-search patience. Only applies on the temperature-0 pass.",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("WHISPER_LANGUAGE", DEFAULT_LANGUAGE),
        help=f"language hint, or empty string for auto-detect. Default: {DEFAULT_LANGUAGE}",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.getenv("WHISPER_INITIAL_PROMPT", ""),
        help="optional vocabulary/context prompt for names, acronyms, or project terms. "
        "Note: this only biases the FIRST window of each decode call.",
    )
    parser.add_argument(
        "--input-file",
        default=os.getenv("WHISPER_INPUT_FILE", ""),
        help="path to an audio file to transcribe instead of live capture "
        "(wav/mp3/m4a/etc). When set, mic/BlackHole capture is skipped.",
    )
    parser.add_argument(
        "--remote-url",
        type=normalize_remote_url,
        default=os.getenv("WHISPER_REMOTE_URL", DEFAULT_REMOTE_URL),
        help="base URL of a remote faster-whisper server (e.g. "
        "http://192.168.77.1:8000) to use instead of local inference. "
        "Live-capture only (not combinable with --input-file); the decoder "
        "flags are ignored since the server controls its own model.",
    )
    args = parser.parse_args()
    return resolve_settings(args, parser.error)


def format_offset(seconds):
    # File mode timestamps the elapsed *audio* position, not wall-clock time
    # (datetime.now() is meaningless for a pre-recorded file). timedelta gives a
    # clean H:MM:SS; we zero-pad the hours to match the live mode's HH:MM:SS.
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_model(settings):
    # Shared by live and file modes so both build the model identically.
    #
    # compute_type: int8 quantizes the linear/embedding weights and runs the
    # rest at the platform float type. On this arm64 CPU int8 and int8_float32
    # are the same thing and fp16/bf16 do not exist, so float32 (unquantized)
    # is the only real precision step -- available via --compute-type, not
    # baked into a preset, because its cost is ~4x the weight memory.
    #
    # cpu_threads: resolved from the preset rather than derived from
    # os.cpu_count(). CTranslate2's thread pool barriers on every GEMM, so on a
    # chip with performance and efficiency cores the slowest thread paces the
    # whole matmul; using every logical core can be slower than using only the
    # performance ones.
    #
    # num_workers=1 keeps Whisper single-worker so it doesn't fight this app's
    # own thread/queue model.
    print("\nLoading Whisper model... (first run downloads model files)")
    model = WhisperModel(
        settings.model,
        device="cpu",
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=1,
    )
    print("Model loaded.")
    return model


def report_model_load_failure(exc, model_size):
    # Classifies a model-load exception into an actionable one-line message
    # instead of a generic catch-all (AUDIT #6).
    text = f"{type(exc).__name__}: {exc}"
    print(f"error: could not load the speech model '{model_size}'.")
    print(f"  {text}\n")

    lowered = text.lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        print(
            "A required package is missing. Run:\n"
            "  pip install -r requirements.txt"
        )
    elif any(
        keyword in lowered
        for keyword in (
            "proxy", "connection", "timed out", "timeout", "resolve",
            "network", "ssl", "certificate", "403", "404", "getaddrinfo",
        )
    ):
        print(
            "This looks like a network problem reaching huggingface.co. Check\n"
            "your connection/proxy, or download a CTranslate2 Whisper model\n"
            "manually and pass its folder path via --model."
        )
    elif "no such file" in lowered or "not a directory" in lowered or "invalid" in lowered:
        print(
            f"'{model_size}' was treated as a path but does not look like a\n"
            "valid model folder. Use a size name (tiny.en, base.en, small.en,\n"
            "medium.en, large-v3, large-v3-turbo) or the path to a CTranslate2\n"
            "model directory."
        )
    else:
        print(
            "If your network blocks huggingface.co, download a CTranslate2\n"
            "Whisper model manually and pass the folder path to --model.\n"
            "Otherwise try a smaller model, e.g. --model base.en"
        )


def report_remote_health_failure(exc, base_url):
    # Classifies a remote-health-check exception, mirroring
    # report_model_load_failure's style for the remote-server case.
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if isinstance(exc, urllib.error.HTTPError):
        return f"{base_url} responded with HTTP {exc.code} — check the endpoint path and server logs"
    if any(keyword in lowered for keyword in ("timed out", "timeout")):
        return (
            f"timed out reaching {base_url} after {REMOTE_HEALTH_TIMEOUT_SECONDS}s "
            "— the network may be congested, the server overloaded, or the address wrong"
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            f"could not reach {base_url} — confirm the remote server is running "
            f"and the network link is up ({text})"
        )
    return f"could not verify {base_url} is healthy: {text}"


def check_remote_health(base_url):
    # Startup liveness check for remote mode, replacing load_model(). Fails
    # loudly and fast rather than discovering a dead server mid-session.
    url = base_url + REMOTE_HEALTH_PATH
    try:
        with urllib.request.urlopen(url, timeout=REMOTE_HEALTH_TIMEOUT_SECONDS) as response:
            status = response.status
    except Exception as exc:
        raise RuntimeError(report_remote_health_failure(exc, base_url)) from exc
    if status != 200:
        raise RuntimeError(f"{url} responded with HTTP {status} — check the endpoint path and server logs")


class RemoteSegment(NamedTuple):
    # Duck-types against faster-whisper's Segment for every field
    # is_hallucination/carry-prompt/transcript-writing code touches, so the
    # rest of the pipeline doesn't need to know whether inference was local
    # or remote. compression_ratio defaults to 0.0 (never trips the
    # hallucination filter) since the remote response contract doesn't
    # include it -- same "bias toward keeping real speech" default as the
    # other missing-field handling in _parse_remote_segments.
    text: str
    start: float
    end: float
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float = 0.0


class RemoteTranscriptionError(RuntimeError):
    pass


def encode_wav_bytes(audio):
    # In-memory WAV encoding purely to build the HTTP request body for
    # transcribe_chunk_remote — not disk write-ahead durability, which stays
    # out of scope.
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def build_multipart_request(url, wav_bytes, fields):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        if value is None:
            continue
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
        + wav_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return request


def _parse_remote_segments(payload):
    global _remote_field_warning_shown
    raw_segments = payload.get("segments")
    if raw_segments is None:
        raise RemoteTranscriptionError(f"remote response missing 'segments' key: {payload!r}")

    result = []
    missing_fields = False
    for seg in raw_segments:
        if "no_speech_prob" not in seg or "avg_logprob" not in seg:
            missing_fields = True
        result.append(
            RemoteSegment(
                text=seg.get("text", ""),
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                # Missing fields default toward "not a hallucination" — bias
                # toward keeping real speech over silently discarding it.
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                avg_logprob=float(seg.get("avg_logprob", 0.0)),
            )
        )

    if missing_fields and not _remote_field_warning_shown:
        print(
            "\nWarning: remote server response is missing no_speech_prob/"
            "avg_logprob on one or more segments; hallucination filtering "
            "may be less accurate for those segments."
        )
        _remote_field_warning_shown = True

    return result


def transcribe_chunk_remote(audio, settings, initial_prompt):
    # Fields are projected out of the SAME dict the local paths decode with,
    # so the wire contract cannot drift from local behaviour. The server still
    # owns its model and compute type; only the decode hints travel. A server
    # that does not declare a field simply ignores it, so this stays
    # backward-compatible with the existing /v1/audio/transcriptions sketch.
    wav_bytes = encode_wav_bytes(audio)
    options = build_decode_options(settings, initial_prompt)
    fields = {name: options[name] for name in REMOTE_FIELDS if options.get(name) is not None}
    url = settings.remote_url + REMOTE_TRANSCRIBE_PATH

    last_exc = None
    attempts = REMOTE_MAX_RETRIES + 1
    for attempt in range(attempts):
        if attempt > 0:
            time.sleep(REMOTE_RETRY_BACKOFF_SECONDS[attempt - 1])
        try:
            request = build_multipart_request(url, wav_bytes, fields)
            with urllib.request.urlopen(request, timeout=REMOTE_CHUNK_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_remote_segments(payload)
        except Exception as exc:  # noqa: BLE001 - collected below, retried
            last_exc = exc

    raise RemoteTranscriptionError(
        f"remote transcription failed after {attempts} attempts against {url}: {last_exc}"
    ) from last_exc


def remote_failure_limit_reached(count):
    return count >= REMOTE_CONSECUTIVE_FAILURE_LIMIT


def build_decode_options(settings, initial_prompt):
    """The single source of truth for local decode parameters.

    Both local decode sites (live chunks and whole-file mode) call this, so a
    parameter can no longer be tuned in one and silently missed in the other.
    transcribe_chunk_remote projects its multipart fields out of the same dict.

    Parameter choices:
      beam_size -- deterministic beam search at temperature 0, the first entry
        in TEMPERATURE_FALLBACK. This is the largest decoder-side accuracy
        lever: greedy (1) loses rare proper nouns to common words on the first
        token; 5 keeps them alive.
      patience -- only applies on that temperature-0 beam pass; above 0 the
        sampler uses best_of instead. The two are never active together.
      temperature / best_of -- if the temp=0 decode is rejected as too
        repetitive or low-confidence (faster-whisper's own
        compression_ratio_threshold / log_prob_threshold), retry at the next
        sampled temperature rather than accept a bad result. This is what
        actually escapes a repetition loop; a single fixed temperature has no
        retry to fall back to. Costs latency only on chunks that need it.
      vad_filter / vad_parameters -- Silero VAD strips non-speech before
        decoding. Tuned conservatively; see VAD_PARAMETERS.
      condition_on_previous_text=False -- decode each chunk independently,
        avoiding Whisper's failure loop where one bad window poisons the rest
        of the session. Note this is a no-op for live chunks anyway (a chunk of
        <=30s is a single window, so there is no "previous"); it is really a
        file-mode choice, and file mode is the unattended long-running case
        where poisoning hurts most.
      word_timestamps / hallucination_silence_threshold -- the latter is only
        consulted inside `if options.word_timestamps`, so they ship together.
        Costs an extra alignment pass per window, hence preset-gated.

    Deliberately NOT set, each for a specific reason:
      chunk_length -- only shrinks the seek stride, so it produces MORE padded
        30s encoder passes for the same audio, and it mutates the shared
        FeatureExtractor in place for every later call.
      length_penalty -- rescales avg_logprob, which is the input to both
        log_prob_threshold and this file's HALLUCINATION_MAX_AVG_LOGPROB.
        Changing it silently retunes the hallucination filter.
      no_repeat_ngram_size / repetition_penalty -- these apply at temperature 0
        too, and meeting speech legitimately repeats ("no, no, no", a name
        listed twice). The repetition-loop failure is already covered twice, by
        compression_ratio_threshold's temperature fallback and by
        is_hallucination. A third, blunter mechanism would trade a rare failure
        for a constant low-grade one.
      log_prob_threshold / no_speech_threshold / compression_ratio_threshold --
        upstream defaults, with this file's own filter layered on top.
    """
    return {
        "language": settings.language.strip() or None,
        "beam_size": settings.beam_size,
        "patience": settings.patience,
        "best_of": DEFAULT_BEST_OF,
        "temperature": list(TEMPERATURE_FALLBACK),
        "vad_filter": True,
        "vad_parameters": dict(VAD_PARAMETERS),
        "condition_on_previous_text": False,
        "initial_prompt": initial_prompt,
        "word_timestamps": settings.word_timestamps,
        "hallucination_silence_threshold": settings.hallucination_silence_threshold,
    }


def transcribe_chunk_local(model, audio, settings, initial_prompt):
    segments, _ = model.transcribe(audio, **build_decode_options(settings, initial_prompt))
    return list(segments)


def transcribe_chunk(model, audio, settings, initial_prompt):
    # Single dispatch point: everything downstream (hallucination filter,
    # carry-prompt update, transcript writing) is unaware of which mode ran.
    if settings.remote_url:
        return transcribe_chunk_remote(audio, settings, initial_prompt)
    return transcribe_chunk_local(model, audio, settings, initial_prompt)


def is_hallucination(segment):
    if (
        segment.no_speech_prob > HALLUCINATION_NO_SPEECH_PROB
        and segment.avg_logprob < HALLUCINATION_MAX_AVG_LOGPROB
    ):
        return True
    return segment.compression_ratio > HALLUCINATION_COMPRESSION_RATIO


def combined_initial_prompt(base, carry):
    if base and carry:
        return f"{base} {carry}"
    return base or carry or None


class SourceBuffer:
    # Per-source (mic/BlackHole) accumulator. dict.setdefault(source, ...) at
    # every use site means an unrecognized source label never raises KeyError
    # (AUDIT #4), unlike the old fixed-literal {"You": ..., "Others": ...} dict.
    def __init__(self):
        self.samples = np.array([], dtype=np.float32)
        self.carry_prompt = ""
        # Wall clock (epoch seconds) of the first sample currently buffered.
        # Segment offsets are relative to the chunk, so chunk start + offset
        # gives the true capture time of every line regardless of lag.
        self.start_time = None

    def append(self, chunk, capture_time=None):
        if len(self.samples) == 0:
            self.start_time = capture_time
        self.samples = np.concatenate([self.samples, chunk])

    def seconds(self):
        return len(self.samples) / TARGET_SAMPLE_RATE

    def tail_rms(self, window_seconds):
        n = int(window_seconds * TARGET_SAMPLE_RATE)
        if len(self.samples) < n or n == 0:
            return float("inf")
        tail = self.samples[-n:]
        return float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))

    def take_all(self):
        # Returns (audio, capture_time_of_first_sample). Explicitly paired so a
        # caller cannot take the audio and forget its clock.
        audio = self.samples
        start_time = self.start_time
        self.samples = np.array([], dtype=np.float32)
        self.start_time = None
        return audio, start_time


def ready_source(buffers, min_chunk_seconds, max_chunk_seconds):
    for source, buf in buffers.items():
        secs = buf.seconds()
        if secs <= MIN_READY_SECONDS:
            continue
        if secs >= max_chunk_seconds:
            return source
        if secs >= min_chunk_seconds and buf.tail_rms(SILENCE_WINDOW_SECONDS) < CHUNK_SILENCE_RMS:
            return source
    return None


def transcribe_file(settings):
    # File mode: decode one pre-recorded audio file end-to-end. No devices, no
    # queue, no threading -- faster-whisper's transcribe() accepts a path and
    # uses its bundled PyAV/FFmpeg to decode + resample any container/codec to
    # 16 kHz mono, so there's nothing to route or chunk ourselves.
    input_path = settings.input_file.strip()
    # Validate before the (potentially slow, download-on-first-run) model load so
    # a typo'd path fails fast instead of after minutes of model setup.
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    try:
        model = load_model(settings)
    except Exception as exc:  # noqa: BLE001 - classified below
        report_model_load_failure(exc, settings.model)
        sys.exit(1)

    # Derive the transcript name from the input basename (no interactive prompt,
    # so file mode is batch-friendly). The label prefixing each line is the file
    # stem -- a single mixed file has no You/Others separation to label.
    stem = sanitize_filename_component(os.path.splitext(os.path.basename(input_path))[0]) or "transcript"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    output_file = os.path.join(TRANSCRIPTS_DIR, f"{stem}_{timestamp}.txt")
    label = stem

    initial_prompt = settings.initial_prompt.strip() or None

    print(f"Input file    : {input_path}")
    print(f"Transcript    : {output_file}")
    print_settings(settings, local=True)
    print("\nTranscribing... (this runs the whole file through Whisper)\n")

    # Identical decode parameters to live mode, from the one shared builder --
    # this is the duplication that used to let file mode quietly keep older
    # settings. Passing the path (not a numpy array) lets faster-whisper decode
    # and normalize internally, so prepare_audio's silence-gate/AGC isn't needed
    # here; that is the only remaining difference between the two local paths.
    segments, _ = model.transcribe(input_path, **build_decode_options(settings, initial_prompt))

    # transcribe() returns a lazy generator; the decode runs as we iterate.
    with open(output_file, "a", encoding="utf-8") as transcript:
        for segment in segments:
            text = segment.text.strip()
            if not text or is_hallucination(segment):
                continue
            line = f"[{format_offset(segment.start)}] {label}: {text}"
            print(line)
            transcript.write(line + "\n")
            transcript.flush()  # write-through so a crash loses nothing

    print(f"\nTranscription saved to: {output_file}")


def print_settings(settings, local):
    # One renderer for both modes so the startup banner can never disagree with
    # what was actually resolved.
    print("\nTranscription settings:")
    print(f"  Accuracy     : {settings.accuracy}")
    if local:
        print(f"  Mode         : local")
        print(f"  Model        : {settings.model}")
        print(f"  Beam size    : {settings.beam_size}")
        print(f"  Compute type : {settings.compute_type}")
        print(f"  CPU threads  : {settings.cpu_threads}")
    else:
        print(f"  Mode         : remote ({settings.remote_url})")
    print(f"  Min/max chunk: {settings.min_chunk_seconds}s / {settings.max_chunk_seconds}s")
    print(f"  Language     : {settings.language.strip() or 'auto'}")
    if settings.explicit:
        print(f"  Overridden   : {', '.join(sorted(settings.explicit))}")


def prepare_audio(audio):
    # RMS (root-mean-square) is a cheap loudness proxy. Below SILENCE_RMS we
    # treat the chunk as silence and drop it — this both saves compute and
    # avoids Whisper hallucinating text on near-silent input.
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms < SILENCE_RMS:
        return None

    # One-shot automatic gain control: scale toward TARGET_RMS, capped at
    # MAX_GAIN. max(rms, 1e-8) avoids divide-by-zero. Only ever amplifies
    # (gain > 1.0); loud audio passes through untouched. clip prevents
    # overflow past the [-1, 1] float sample range.
    gain = min(TARGET_RMS / max(rms, 1e-8), MAX_GAIN)
    if gain > 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def line_timestamp(chunk_start_time, offset_seconds):
    # Capture-time stamping. `offset_seconds` is segment.start, relative to the
    # chunk; with vad_filter=True faster-whisper has already mapped it back to
    # the un-stripped chunk timeline (restore_speech_timestamps), so adding it
    # to the chunk's capture time is correct even when VAD removed audio.
    # Falls back to "now" only if a chunk somehow arrived without a clock.
    if chunk_start_time is None:
        moment = datetime.datetime.now()
    else:
        moment = datetime.datetime.fromtimestamp(chunk_start_time + max(0.0, offset_seconds))
    return moment.strftime("%H:%M:%S")


def transcribe_loop(model, audio_queue, output_file, settings, source_rates, worker_error):
    # Consumer thread: pulls (source, raw block, capture time) tuples off the
    # queue, downmixes + resamples them (kept off the real-time callback
    # thread, AUDIT #1), accumulates them per source via SourceBuffer.setdefault
    # (AUDIT #4), and transcribes once ready_source() says a source has a
    # full adaptive chunk. The whole body is wrapped so a crash sets
    # worker_error instead of dying silently (AUDIT #3).
    buffers = {}
    base_prompt = settings.initial_prompt.strip() or None
    last_backlog_warning = 0.0
    consecutive_remote_failures = 0

    try:
        with open(output_file, "a", encoding="utf-8") as transcript:
            while True:
                source, raw_block, capture_time = audio_queue.get()

                mono = raw_block.astype(np.float32, copy=True)
                if mono.ndim > 1:
                    mono = mono.mean(axis=1)  # average channels to mono
                mono = mono.reshape(-1)
                mono = resample_to_target(mono, source_rates[source])

                buf = buffers.setdefault(source, SourceBuffer())
                buf.append(mono, capture_time)

                # Lag indicator. Triggered at 2x the max chunk size rather than
                # 4x so a preset that cannot keep up announces itself in the
                # first minute instead of after the meeting has drifted. Note
                # this is a latency warning only -- timestamps stay correct
                # because they come from capture time, not write time.
                backlog_seconds = sum(b.seconds() for b in buffers.values())
                now = time.monotonic()
                if (
                    backlog_seconds > settings.max_chunk_seconds * BACKLOG_WARNING_MULTIPLIER
                    and now - last_backlog_warning > 30
                ):
                    print(
                        f"\nWarning: transcription is behind by about {int(backlog_seconds)} seconds "
                        "of audio, and the gap will keep growing while decoding is slower than "
                        "real time. Timestamps remain accurate; only the on-screen delay grows. "
                        "Use --accuracy fast, a smaller --model, or --beam-size 1 if this keeps rising."
                    )
                    last_backlog_warning = now

                while True:
                    ready = ready_source(buffers, settings.min_chunk_seconds, settings.max_chunk_seconds)
                    if ready is None:
                        break
                    ready_buf = buffers[ready]
                    audio, chunk_start_time = ready_buf.take_all()
                    carry = ready_buf.carry_prompt
                    audio = prepare_audio(audio)
                    if audio is None:
                        continue

                    prompt = combined_initial_prompt(base_prompt, carry)
                    try:
                        segments = transcribe_chunk(model, audio, settings, prompt)
                        consecutive_remote_failures = 0
                    except RemoteTranscriptionError as exc:
                        print(f"\nWarning: {exc}")
                        line = (
                            f"[{line_timestamp(chunk_start_time, 0.0)}] {ready}: "
                            "[transcription dropped -- remote server error]"
                        )
                        print(line)
                        transcript.write(line + "\n")
                        transcript.flush()
                        consecutive_remote_failures += 1
                        if remote_failure_limit_reached(consecutive_remote_failures):
                            raise RuntimeError(
                                f"remote server failed {consecutive_remote_failures} chunks in a row; giving up"
                            ) from exc
                        continue

                    kept_texts = []
                    for segment in segments:
                        text = segment.text.strip()
                        if not text or is_hallucination(segment):
                            continue
                        kept_texts.append(text)
                        line = f"[{line_timestamp(chunk_start_time, segment.start)}] {ready}: {text}"
                        print(line)
                        transcript.write(line + "\n")
                        transcript.flush()  # write-through so a crash loses nothing

                    if kept_texts:
                        ready_buf.carry_prompt = kept_texts[-1][-CARRY_PROMPT_MAX_CHARS:]
    except Exception:
        traceback.print_exc()
        worker_error.set()


def main():
    settings = parse_args()

    # File mode short-circuit: transcribe a pre-recorded file and exit, skipping
    # all device enumeration, mic/BlackHole selection, and the live pipeline.
    if settings.input_file.strip():
        try:
            transcribe_file(settings)
        except FileNotFoundError as exc:
            print(f"\nError: {exc}")
        return

    # Remote mode: fail fast on a dead/unreachable server before asking the
    # user to answer any interactive prompts (same "validate cheap things
    # before expensive ones" principle as transcribe_file's own path check).
    if settings.remote_url:
        try:
            check_remote_health(settings.remote_url)
        except RuntimeError as exc:
            print(f"\nError: {exc}")
            return
        # Driven by what the user actually typed, not by comparing values
        # against defaults -- the old comparison silently misfired whenever a
        # default moved, and could not tell "typed the default" from "preset
        # supplied it".
        ignored = sorted(REMOTE_IGNORED_FLAGS & settings.explicit)
        if ignored:
            flags = ", ".join("--" + name.replace("_", "-") for name in ignored)
            print(f"\nNote: {flags} ignored in remote mode; the server controls its own model.")

    devices = sd.query_devices()
    list_devices(devices)

    try:
        blackhole_index = find_blackhole_input(devices)
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        return

    mic_index = choose_mic_input(devices, blackhole_index)
    output_file = output_filename()

    mic_name = devices[mic_index]["name"]
    blackhole_name = devices[blackhole_index]["name"]
    mic_rate = device_sample_rate(devices, mic_index)
    blackhole_rate = device_sample_rate(devices, blackhole_index)
    source_rates = {SOURCE_YOU: mic_rate, SOURCE_OTHERS: blackhole_rate}

    print("")
    print(f"Mic input     : {mic_name}")
    print(f"Remote input  : {blackhole_name}")
    print(f"Transcript    : {output_file}")

    print_settings(settings, local=not settings.remote_url)

    model = None
    if not settings.remote_url:
        try:
            model = load_model(settings)
        except Exception as exc:  # noqa: BLE001 - classified below
            report_model_load_failure(exc, settings.model)
            return

    print("Listening... Press Ctrl+C to stop.\n")

    # Shared queue links the two audio callbacks (producers) to the single
    # transcribe_loop (consumer). daemon=True means the worker dies with the
    # main thread on Ctrl+C -- no explicit join/shutdown needed. worker_error
    # lets main() notice if the worker thread dies for any other reason.
    audio_queue = queue.Queue()
    worker_error = threading.Event()
    worker = threading.Thread(
        target=transcribe_loop,
        args=(model, audio_queue, output_file, settings, source_rates, worker_error),
        daemon=True,
    )
    worker.start()

    # Open both input streams at once via a single `with` (both are closed in
    # reverse order on exit). Each stream's callback tags its audio "You" vs
    # "Others" and opens at its own device's native rate (AUDIT #5).
    try:
        with sd.InputStream(
            samplerate=mic_rate,
            channels=stream_channels(devices, mic_index),
            dtype="float32",
            device=mic_index,
            callback=make_callback(SOURCE_YOU, audio_queue),
        ), sd.InputStream(
            samplerate=blackhole_rate,
            channels=stream_channels(devices, blackhole_index),
            dtype="float32",
            device=blackhole_index,
            callback=make_callback(SOURCE_OTHERS, audio_queue),
        ):
            while True:
                sd.sleep(1000)
                if worker_error.is_set() or not worker.is_alive():
                    print("\nError: the transcription worker stopped unexpectedly (see traceback above).")
                    print(f"Partial transcript saved to: {output_file}")
                    sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nTranscription saved to: {output_file}")
    except sd.PortAudioError as exc:
        print(f"\nError: could not open an audio stream: {exc}")


if __name__ == "__main__":
    main()
