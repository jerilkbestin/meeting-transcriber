# Post-meeting speaker diarization for transcripts recorded with
# `transcribe.py --record`.
#
# transcribe.py already knows exactly who "You" is -- your microphone is a
# physically separate stream -- so the only open question is which of the
# remote participants is speaking inside the mixed "Others" stream. This script
# answers just that: it re-transcribes both recorded WAVs, runs pyannote over
# the Others WAV alone, and writes a second transcript where Others is split
# into "Others 1", "Others 2", ...
#
# Run after the meeting, never during it: pyannote clusters speakers within
# whatever audio it is given, so one pass over the whole recording produces
# labels that stay consistent for the entire meeting, which per-chunk live
# diarization cannot do.
#
# pyannote.audio and torch are NOT required to install or import this module --
# they are imported lazily, only when diarization actually runs. See
# requirements-diarize.txt.

import argparse
import datetime
import json
import os
import struct
import sys

import numpy as np

import transcribe


MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
TOKEN_URL = "https://huggingface.co/settings/tokens"
REQUIREMENTS_FILE = "requirements-diarize.txt"

DIARIZED_TXT_SUFFIX = "_diarized.txt"
DIARIZED_JSON_SUFFIX = "_diarized.json"

# Post-hoc decoding has no real-time budget, so it defaults to the preset that
# transcribe.py documents as "intended for --input-file": beam 5, word
# timestamps, and the hallucination-silence filter those enable.
DEFAULT_DIARIZE_ACCURACY = "max"

# A segment is flagged as straddling a speaker change when the runner-up holds
# both a large enough share AND enough absolute time. Share alone would flag
# every half-second interjection; seconds alone would flag long segments with a
# trivial overlap.
STRADDLE_MIN_SHARE = 0.35
STRADDLE_MIN_SECONDS = 1.0

DEVICE_CHOICES = ("cpu", "mps", "cuda")


class SpeakerTurn(object):
    # One contiguous stretch of one speaker, in seconds from the start of the
    # Others WAV. Plain class rather than NamedTuple so `raw` can be compared
    # and sorted without tuple-ordering surprises.
    __slots__ = ("start", "end", "raw")

    def __init__(self, start, end, raw):
        self.start = float(start)
        self.end = float(end)
        self.raw = raw

    def __repr__(self):
        return f"SpeakerTurn({self.start:.2f}, {self.end:.2f}, {self.raw!r})"


class DiarizedLine(object):
    """One transcript line, already placed on the session timeline.

    `offset` is seconds from the start of the session (the earliest stream's
    first block), which is what the merge sorts on. `wall` is the epoch time of
    the same instant, so a line can be printed on the same clock the live
    transcript used.
    """

    __slots__ = ("offset", "wall", "stream", "raw", "label", "text", "shares", "straddled", "end")

    def __init__(self, offset, wall, stream, text, raw=None, end=None, shares=None, straddled=False):
        self.offset = offset
        self.wall = wall
        self.stream = stream
        self.text = text
        self.raw = raw
        self.end = end
        self.shares = shares or {}
        self.straddled = straddled
        self.label = stream


# ---------------------------------------------------------------------------
# Reading what transcribe.py --record wrote
# ---------------------------------------------------------------------------


def load_streams_sidecar(path):
    # The sidecar is the contract between the recorder and this script: it
    # names the WAVs and, crucially, carries each stream's first-block wall
    # clock so two independently-clocked devices can be put on one timeline.
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    schema = payload.get("schema")
    if schema != transcribe.SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {schema!r}, expected {transcribe.SIDECAR_SCHEMA_VERSION}. "
            "It was written by a different version of transcribe.py."
        )
    streams = payload.get("streams")
    if not streams:
        raise ValueError(f"{path} lists no streams; the recording produced nothing.")

    directory = os.path.dirname(os.path.abspath(path))
    for name, stream in streams.items():
        wav = stream.get("wav")
        if not wav:
            raise ValueError(f"{path}: stream {name!r} has no 'wav' entry")
        # Stored as a basename so the whole set can be copied elsewhere (for
        # example to a GPU box) without rewriting the sidecar.
        stream["path"] = os.path.join(directory, wav)
        if not os.path.isfile(stream["path"]):
            raise FileNotFoundError(f"{path} refers to {wav}, which is not next to it")
    return payload


def _parse_riff(path):
    """Return (channels, sample_rate, sample_width, pcm_bytes) from a WAV file.

    Deliberately walks the RIFF chunks instead of using the `wave` module, for
    one reason: a recorder killed mid-write can leave a `data` chunk whose
    declared size does not match what is actually on disk, and `wave` would
    then hand back a truncated (or zero-length) recording with no error. The
    file length is the ground truth here, so a crashed session still diarizes.
    """
    with open(path, "rb") as handle:
        raw = handle.read()

    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    fmt = None
    data = None
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        declared = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = pos + 8
        available = len(raw) - body

        if chunk_id == b"fmt " and available >= 16:
            fmt = struct.unpack("<HHIIHH", raw[body:body + 16])
        elif chunk_id == b"data":
            usable = declared if 0 < declared <= available else available
            data = raw[body:body + usable]
            if declared == 0 or declared > available:
                break  # header is stale; everything to EOF is audio
        # RIFF chunks are word-aligned, so an odd size is followed by a pad
        # byte. The `or 1` guarantees forward progress: a zero-size chunk would
        # otherwise loop forever on the same position.
        stride = declared + (declared & 1)
        pos = body + (stride or 1)

    if fmt is None:
        raise ValueError(f"{path} has no fmt chunk")
    if data is None:
        raise ValueError(f"{path} has no data chunk")
    _, channels, sample_rate, _, _, bits = fmt
    return channels, sample_rate, bits // 8, data


def read_wav_mono16(path):
    # Returns float32 mono at transcribe.TARGET_SAMPLE_RATE, i.e. exactly what
    # both faster-whisper and pyannote want. Handles stereo and off-rate files
    # so externally recorded audio works too, reusing the transcriber's own
    # resampler rather than a second implementation.
    channels, sample_rate, width, data = _parse_riff(path)
    if width != 2:
        raise ValueError(f"{path} is {width * 8}-bit; only 16-bit PCM is supported")
    if channels < 1:
        raise ValueError(f"{path} declares {channels} channels")

    samples = np.frombuffer(data, dtype="<i2")
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    audio = samples.astype(np.float32) / 32768.0
    if sample_rate != transcribe.TARGET_SAMPLE_RATE:
        audio = transcribe.resample_to_target(audio, sample_rate)
    return audio


# ---------------------------------------------------------------------------
# Loading the diarizer (the only part that needs torch)
# ---------------------------------------------------------------------------


class DiarizerAccessError(RuntimeError):
    # Raised when the model is reachable but not usable by this account.
    # pyannote signals that two different ways -- it raises for a gated repo,
    # but returns None when from_pretrained cannot build the pipeline -- and
    # both deserve the same "accept the terms" instructions, so the None case
    # is converted into this type rather than left to string matching.
    pass


def resolve_hf_token(env=None):
    # Deliberately no --hf-token flag: a token on the command line lands in
    # shell history and in `ps` output. Environment first, then whatever
    # `hf auth login` stored. None is a valid answer -- once the model is in
    # the local cache, no token is needed at all.
    env = os.environ if env is None else env
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = (env.get(name) or "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    try:
        return get_token() or None
    except Exception:  # noqa: BLE001 - a token store problem must not be fatal
        return None


def report_diarizer_load_failure(exc):
    # Returns the message instead of printing it, mirroring
    # transcribe.report_remote_health_failure -- that shape is what makes the
    # classification testable without capturing stdout.
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            "pyannote.audio is not installed. Diarization is optional and kept out of\n"
            f"requirements.txt on purpose; install it with:\n"
            f"  pip install -r {REQUIREMENTS_FILE}\n"
            "transcribe.py does not need it."
        )

    gated = (
        isinstance(exc, DiarizerAccessError)
        or "gatedrepoerror" in lowered
        or "gated" in lowered
        or "401" in lowered
        or "403" in lowered
    )
    if gated:
        return (
            f"access to {MODEL_ID} has not been granted for this account.\n"
            f"  1. Open {MODEL_URL} and accept the model terms.\n"
            f"  2. Create a read token at {TOKEN_URL}.\n"
            "  3. export HF_TOKEN=hf_...   (or run: hf auth login)\n"
            f"Current token: {'found' if resolve_hf_token() else 'NOT FOUND'}."
        )

    offline = "localentrynotfounderror" in lowered or "offline" in lowered
    if offline:
        return (
            f"{MODEL_ID} is not in the local Hugging Face cache and offline mode is on.\n"
            "Run once with network access (and HF_HUB_OFFLINE unset) to download it;\n"
            "afterwards it works fully offline."
        )

    if any(
        keyword in lowered
        for keyword in ("proxy", "connection", "timed out", "timeout", "resolve", "network", "ssl", "certificate", "getaddrinfo")
    ):
        return (
            f"could not reach huggingface.co to fetch {MODEL_ID}.\n"
            f"Check your connection/proxy, or pre-download the model once. ({text})"
        )

    return f"could not load {MODEL_ID}: {text}"


def load_diarizer(token, device):
    # Imports live inside the function so `import diarize` stays free of torch.
    # That is what lets the whole non-diarization path -- reading, decoding,
    # merging, writing -- be tested and used with nothing extra installed.
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    if pipeline is None:
        # from_pretrained returns None (rather than raising) when it cannot
        # build the pipeline -- most often because the terms have not been
        # accepted. Converted to a typed error so the caller reports the same
        # actionable steps as a raised gated-repo error.
        raise DiarizerAccessError(
            f"Pipeline.from_pretrained({MODEL_ID!r}) returned None -- this normally "
            "means the model terms have not been accepted, or the token is missing."
        )
    return pipeline.to(torch.device(device))


def diarize_audio(pipeline, samples, num_speakers=None, min_speakers=None, max_speakers=None):
    """Run the pipeline over in-memory audio and return sorted SpeakerTurns.

    Feeding a waveform dict rather than a path keeps torchcodec/ffmpeg out of
    the picture entirely -- the audio is already decoded, mono and 16 kHz.

    Prefers `exclusive_speaker_diarization`, which resolves overlapping speech
    to a single speaker per instant. That is the right shape here, because a
    Whisper segment is attributed to exactly one speaker; the non-exclusive
    annotation would make overlaps compete on both sides of the argmax.
    """
    import torch

    waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(
        {"waveform": waveform, "sample_rate": transcribe.TARGET_SAMPLE_RATE},
        **kwargs,
    )
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)
    return turns_from_annotation(annotation)


def turns_from_annotation(annotation):
    # itertracks(yield_label=True) rather than iterating the annotation
    # directly: bare __iter__ changed shape between pyannote.core 5 and 6,
    # itertracks did not.
    turns = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(segment.start, segment.end, label))
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return turns


# ---------------------------------------------------------------------------
# Matching speaker turns to transcript segments
# ---------------------------------------------------------------------------


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_overlap_shares(start, end, turns):
    # Seconds of this segment covered by each speaker. Summing overlap rather
    # than sampling the midpoint is what stops a short backchannel ("mm-hm")
    # that happens to land mid-segment from stealing the whole line.
    shares = {}
    for turn in turns:
        if turn.end <= start:
            continue
        if turn.start >= end:
            break  # turns are sorted, so nothing later can overlap
        covered = overlap_seconds(start, end, turn.start, turn.end)
        if covered > 0:
            shares[turn.raw] = shares.get(turn.raw, 0.0) + covered
    return shares


def assign_speaker(start, end, turns):
    """Return (raw_label, shares, straddled) for one transcript segment.

    Falls back to the nearest turn when there is no overlap at all (Whisper and
    the VAD disagree about exactly where speech starts), and to None when the
    diarizer found no speakers -- callers then keep the plain "Others" label
    rather than inventing a speaker.
    """
    shares = speaker_overlap_shares(start, end, turns)
    if shares:
        ranked = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
        winner, winner_seconds = ranked[0]
        straddled = False
        if len(ranked) > 1:
            runner_up_seconds = ranked[1][1]
            duration = max(end - start, 1e-9)
            straddled = (
                runner_up_seconds >= STRADDLE_MIN_SECONDS
                and (runner_up_seconds / duration) >= STRADDLE_MIN_SHARE
            )
        return winner, shares, straddled

    if not turns:
        return None, {}, False

    midpoint = (start + end) / 2.0
    nearest = min(turns, key=lambda turn: _distance_to(midpoint, turn))
    return nearest.raw, {}, False


def _distance_to(point, turn):
    if turn.start <= point <= turn.end:
        return 0.0
    return min(abs(point - turn.start), abs(point - turn.end))


def number_others(assignments):
    """Map pyannote's raw labels to stable "Others N" names.

    pyannote's SPEAKER_00/01/... numbering comes out of clustering and carries
    no guarantee of being stable between runs, so it is never shown. Numbering
    by first appearance is both deterministic and what a reader expects: the
    first remote voice in the meeting is Others 1.

    Only labels that actually won a segment are numbered, so the transcript can
    never mention an "Others 3" that has no lines.
    """
    first_seen = {}
    for index, (raw, start) in enumerate(assignments):
        if raw is None:
            continue
        if raw not in first_seen or start < first_seen[raw][0]:
            first_seen[raw] = (start, index)

    ordered = sorted(first_seen.items(), key=lambda item: (item[1][0], str(item[0])))
    return {raw: f"{transcribe.SOURCE_OTHERS} {n}" for n, (raw, _) in enumerate(ordered, start=1)}


# ---------------------------------------------------------------------------
# Transcription + merge + output
# ---------------------------------------------------------------------------


def transcribe_stream(model, wav_path, settings):
    # One decode of a whole WAV, using transcribe.py's single decode-options
    # builder so this pass can never drift away from the live path's parameters.
    initial_prompt = settings.initial_prompt.strip() or None
    segments, _info = model.transcribe(
        wav_path, **transcribe.build_decode_options(settings, initial_prompt)
    )
    kept = []
    for segment in segments:
        text = segment.text.strip()
        if not text or transcribe.is_hallucination(segment):
            continue
        kept.append((float(segment.start), float(segment.end), text))
    return kept


def merge_timeline(lines):
    # Stable sort on the session-relative offset, with You first on an exact
    # tie: if both sides start at the same instant, the local speaker is the
    # one we are certain about.
    return sorted(lines, key=lambda line: (line.offset, 0 if line.stream == transcribe.SOURCE_YOU else 1))


def format_line(line, use_offsets=False, mark_uncertain=False):
    if use_offsets or line.wall is None:
        stamp = transcribe.format_offset(line.offset)
    else:
        stamp = datetime.datetime.fromtimestamp(line.wall).strftime("%H:%M:%S")
    suffix = " [?]" if (mark_uncertain and line.straddled) else ""
    return f"[{stamp}] {line.label}: {line.text}{suffix}"


def write_diarized_outputs(lines, txt_path, json_path, meta, use_offsets=False, mark_uncertain=False):
    header_bits = [
        "diarized " + datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        f"asr {meta.get('asr_model')} ({meta.get('accuracy')})",
        meta.get("diarizer") or "no diarization",
        f"{meta.get('speaker_count', 0)} remote speakers",
    ]
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("# " + " | ".join(str(bit) for bit in header_bits) + "\n")
        for line in lines:
            rendered = format_line(line, use_offsets, mark_uncertain)
            print(rendered)
            handle.write(rendered + "\n")
            handle.flush()

    payload = dict(meta)
    payload["lines"] = [
        {
            "offset": round(line.offset, 3),
            "end": round(line.end, 3) if line.end is not None else None,
            "wall": line.wall,
            "stream": line.stream,
            "label": line.label,
            "raw_speaker": line.raw,
            "shares": {key: round(value, 3) for key, value in line.shares.items()},
            "straddled": line.straddled,
            "text": line.text,
        }
        for line in lines
    ]
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_settings(args, fail):
    # Reuses transcribe.resolve_settings so --accuracy/--model/--beam-size mean
    # exactly what they mean in transcribe.py, including the preset/env/flag
    # precedence and every cross-check.
    namespace = argparse.Namespace(
        accuracy=args.accuracy or DEFAULT_DIARIZE_ACCURACY,
        model=args.model,
        beam_size=args.beam_size,
        compute_type=args.compute_type,
        cpu_threads=None,
        min_chunk_seconds=None,
        max_chunk_seconds=None,
        patience=None,
        language=args.language,
        initial_prompt=args.initial_prompt,
        input_file="",
        remote_url="",
        record=False,
    )
    return transcribe.resolve_settings(namespace, fail)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Split the 'Others' stream of a recorded meeting into individual speakers.",
        epilog="Record a meeting first with: python transcribe.py --record",
    )
    parser.add_argument(
        "sidecar",
        nargs="?",
        default="",
        help="path to the *_streams.json written by transcribe.py --record",
    )
    parser.add_argument("--you-wav", default="", help="mic WAV, when there is no sidecar")
    parser.add_argument("--others-wav", default="", help="meeting-audio WAV, when there is no sidecar")
    parser.add_argument(
        "--accuracy",
        choices=transcribe.ACCURACY_CHOICES,
        default=None,
        help=f"decode preset. Default: {DEFAULT_DIARIZE_ACCURACY} (no real-time constraint here)",
    )
    parser.add_argument("--model", default=None, help="override the preset's faster-whisper model")
    parser.add_argument("--beam-size", type=int, default=None, help="override the preset's beam size")
    parser.add_argument("--compute-type", default=None, help="override the preset's compute type")
    parser.add_argument("--language", default=transcribe.DEFAULT_LANGUAGE, help="language hint")
    parser.add_argument("--initial-prompt", default="", help="vocabulary/context prompt")
    parser.add_argument("--num-speakers", type=int, default=None, help="exact number of remote speakers")
    parser.add_argument("--min-speakers", type=int, default=None, help="lower bound on remote speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="upper bound on remote speakers")
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="cpu",
        help="torch device for diarization. Default: cpu (mps support is unreliable)",
    )
    parser.add_argument("--offsets", action="store_true", help="timestamp from session start instead of wall clock")
    parser.add_argument("--mark-uncertain", action="store_true", help="append [?] to lines that straddle a speaker change")
    parser.add_argument("--skip-you", action="store_true", help="only process the Others stream")
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="re-transcribe and merge only, without pyannote (no torch needed)",
    )
    parser.add_argument("--output", default="", help="write the transcript here instead of next to the recording")

    args = parser.parse_args(argv)

    if not args.sidecar and not args.others_wav:
        parser.error("give a *_streams.json sidecar, or --others-wav (and usually --you-wav)")
    if args.sidecar and (args.you_wav or args.others_wav):
        parser.error("use either the sidecar or --you-wav/--others-wav, not both")
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers is exact; do not combine it with --min-speakers/--max-speakers")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error(f"--min-speakers ({args.min_speakers}) cannot exceed --max-speakers ({args.max_speakers})")

    args.settings = build_settings(args, parser.error)
    return args


def resolve_inputs(args):
    """Return (streams, session_start, base_path, use_offsets).

    `streams` maps a source label to {"path", "start"}, where start is the
    stream's epoch wall clock, or None when there is nothing to anchor to.
    """
    if args.sidecar:
        payload = load_streams_sidecar(args.sidecar)
        streams = {}
        for name, stream in payload["streams"].items():
            streams[name] = {"path": stream["path"], "start": stream.get("first_block_wall")}
        starts = [s["start"] for s in streams.values() if s["start"] is not None]
        session_start = min(starts) if starts else None
        base = args.sidecar[: -len(transcribe.STREAMS_SIDECAR_SUFFIX)] if args.sidecar.endswith(
            transcribe.STREAMS_SIDECAR_SUFFIX
        ) else os.path.splitext(args.sidecar)[0]
        return streams, session_start, base, args.offsets

    streams = {transcribe.SOURCE_OTHERS: {"path": args.others_wav, "start": 0.0}}
    if args.you_wav:
        streams[transcribe.SOURCE_YOU] = {"path": args.you_wav, "start": 0.0}
    base = os.path.splitext(args.others_wav)[0]
    # Without a sidecar there is no shared clock: the two files are simply
    # assumed to start together, so wall-clock stamps would be a fiction.
    return streams, 0.0, base, True


def main(argv=None):
    args = parse_args(argv)
    settings = args.settings

    try:
        streams, session_start, base, use_offsets = resolve_inputs(args)
    except (OSError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1

    if args.you_wav and not args.sidecar:
        print("Note: without a sidecar the two WAVs are assumed to start together; using --offsets.")

    others = streams.get(transcribe.SOURCE_OTHERS)
    if others is None:
        print(f"\nError: no {transcribe.SOURCE_OTHERS} stream to diarize.")
        return 1

    turns = []
    diarizer_name = None
    if not args.no_diarize:
        token = resolve_hf_token()
        try:
            pipeline = load_diarizer(token, args.device)
        except Exception as exc:  # noqa: BLE001 - classified into an actionable message
            print(f"\nError: {report_diarizer_load_failure(exc)}")
            return 1
        if args.device == "mps":
            print("Note: --device mps is opt-in; some pyannote ops fall back to CPU and can be slower.")
        print(f"Diarizing {os.path.basename(others['path'])} with {MODEL_ID} on {args.device}...")
        audio = read_wav_mono16(others["path"])
        turns = diarize_audio(
            pipeline,
            audio,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        diarizer_name = MODEL_ID
        print(f"Found {len({turn.raw for turn in turns})} speaker(s) across {len(turns)} turns.")

    try:
        model = transcribe.load_model(settings)
    except Exception as exc:  # noqa: BLE001 - classified by transcribe's reporter
        transcribe.report_model_load_failure(exc, settings.model)
        return 1

    lines = []
    assignments = []
    for source in sorted(streams):
        if source == transcribe.SOURCE_YOU and args.skip_you:
            continue
        stream = streams[source]
        print(f"\nTranscribing {os.path.basename(stream['path'])} ({source})...")
        stream_start = stream["start"]
        offset_base = 0.0
        if stream_start is not None and session_start is not None:
            offset_base = stream_start - session_start

        for start, end, text in transcribe_stream(model, stream["path"], settings):
            wall = None if stream_start is None else stream_start + start
            line = DiarizedLine(
                offset=offset_base + start,
                wall=wall,
                stream=source,
                text=text,
                end=offset_base + end,
            )
            if source == transcribe.SOURCE_OTHERS and turns:
                raw, shares, straddled = assign_speaker(start, end, turns)
                line.raw = raw
                line.shares = shares
                line.straddled = straddled
                assignments.append((raw, start))
            lines.append(line)

    # Numbering happens only after every segment is assigned, so the numbers
    # reflect which speakers actually said something.
    naming = number_others(assignments)
    for line in lines:
        if line.stream == transcribe.SOURCE_OTHERS and line.raw in naming:
            line.label = naming[line.raw]

    lines = merge_timeline(lines)

    txt_path = args.output or (base + DIARIZED_TXT_SUFFIX)
    json_path = os.path.splitext(txt_path)[0] + ".json" if args.output else base + DIARIZED_JSON_SUFFIX
    meta = {
        "asr_model": settings.model,
        "accuracy": settings.accuracy,
        "diarizer": diarizer_name,
        "speaker_count": len(naming),
        "speakers": {raw: name for raw, name in naming.items()},
        "sources": {name: os.path.basename(stream["path"]) for name, stream in streams.items()},
        "session_start_wall": session_start,
        "timestamps": "offset" if use_offsets else "wall",
    }

    print("")
    write_diarized_outputs(lines, txt_path, json_path, meta, use_offsets, args.mark_uncertain)
    print(f"\nDiarized transcript: {txt_path}")
    print(f"Details            : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
