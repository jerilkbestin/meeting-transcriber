"""Rolling, near-real-time transcription on top of faster-whisper.

Whisper is a *batch* model: it wants a chunk of audio, not a stream. To make it
feel live we accumulate audio per track and flush a chunk to the model whenever
one of two things happens:

1. we have at least ``min_chunk_seconds`` and the speaker has just gone quiet
   (a natural sentence boundary), or
2. we hit ``max_chunk_seconds`` and cut anyway.

Cutting on silence matters: slicing mid-word is the main cause of garbled or
duplicated output in naive streaming-Whisper implementations.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .config import Settings, TARGET_SR


@dataclass
class Utterance:
    track: str
    speaker: str
    start: float  # seconds from the start of the meeting
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float

    def timestamp(self) -> str:
        total = int(self.start)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class _TrackBuffer:
    """Accumulates 16 kHz mono float32 audio for a single track."""

    def __init__(self, name: str):
        self.name = name
        self.blocks: list[np.ndarray] = []
        self.n_samples = 0
        # Absolute position (in samples) of the start of the current buffer,
        # measured from the start of capture. This is what turns per-chunk
        # timestamps into meeting-wide timestamps.
        self.offset_samples = 0
        self.carry_prompt: str = ""

    def append(self, samples: np.ndarray) -> None:
        if samples.size:
            self.blocks.append(samples)
            self.n_samples += samples.size

    def seconds(self) -> float:
        return self.n_samples / TARGET_SR

    def _materialize(self) -> np.ndarray:
        if not self.blocks:
            return np.zeros(0, dtype=np.float32)
        if len(self.blocks) > 1:
            self.blocks = [np.concatenate(self.blocks)]
        return self.blocks[0]

    def tail_rms(self, window_seconds: float) -> float:
        audio = self._materialize()
        n = int(window_seconds * TARGET_SR)
        if audio.size < n or n == 0:
            return float("inf")
        tail = audio[-n:]
        return float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))

    def take_all(self) -> np.ndarray:
        audio = self._materialize()
        self.blocks = []
        self.n_samples = 0
        self.offset_samples += audio.size
        return audio

    def peek_offset_seconds(self) -> float:
        return self.offset_samples / TARGET_SR


class RollingTranscriber:
    """Feeds audio to Whisper in silence-aligned chunks and emits Utterances."""

    def __init__(self, settings: Settings, transcript_path: Path):
        self.settings = settings
        self.transcript_path = transcript_path
        self._buffers: dict[str, _TrackBuffer] = {}
        self._lock = threading.Lock()
        self._model = None
        self._utterances: list[Utterance] = []
        self._jsonl = transcript_path.open("a", encoding="utf-8")

    # -- model ------------------------------------------------------------
    def load_model(self):
        """Import and construct the model lazily.

        faster-whisper pulls in CTranslate2 and downloads weights on first use;
        doing that at import time would make ``--help`` take 30 seconds.
        """
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.model_size,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        return self._model

    # -- ingest -----------------------------------------------------------
    def feed(self, track: str, samples: np.ndarray) -> None:
        with self._lock:
            self._buffers.setdefault(track, _TrackBuffer(track)).append(samples)

    def _ready_track(self, force: bool) -> str | None:
        s = self.settings
        for name, buf in self._buffers.items():
            secs = buf.seconds()
            if secs <= 0.25:
                continue
            if force:
                return name
            if secs >= s.max_chunk_seconds:
                return name
            if secs >= s.min_chunk_seconds and buf.tail_rms(s.silence_window_seconds) < s.silence_rms:
                return name
        return None

    def drain(self, force: bool = False) -> list[Utterance]:
        """Transcribe whatever is ready. Returns newly produced utterances."""
        produced: list[Utterance] = []
        while True:
            with self._lock:
                name = self._ready_track(force)
                if name is None:
                    break
                buf = self._buffers[name]
                offset = buf.peek_offset_seconds()
                prompt = buf.carry_prompt
                audio = buf.take_all()
            if audio.size == 0:
                continue
            new = self._transcribe_chunk(name, audio, offset, prompt)
            if new:
                with self._lock:
                    # Carry a little context forward so the next chunk knows how
                    # the previous sentence ended (helps with names and acronyms).
                    self._buffers[name].carry_prompt = new[-1].text[-220:]
                produced.extend(new)
            if not force:
                # In live mode do one chunk per call so we stay responsive.
                break
        return produced

    def _transcribe_chunk(
        self, track: str, audio: np.ndarray, offset: float, prompt: str
    ) -> list[Utterance]:
        model = self.load_model()
        s = self.settings
        initial_prompt = s.initial_prompt or None
        if prompt:
            initial_prompt = f"{initial_prompt + ' ' if initial_prompt else ''}{prompt}"

        segments, _info = model.transcribe(
            audio,
            language=s.language,
            beam_size=s.beam_size,
            vad_filter=s.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=False,
        )

        speaker = s.speaker_names.get(track, track)
        out: list[Utterance] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Whisper hallucinates filler on near-silence; these two thresholds
            # are the standard guard and cost us nothing on real speech.
            if seg.no_speech_prob > 0.85 and seg.avg_logprob < -0.9:
                continue
            utt = Utterance(
                track=track,
                speaker=speaker,
                start=offset + seg.start,
                end=offset + seg.end,
                text=text,
                avg_logprob=float(seg.avg_logprob),
                no_speech_prob=float(seg.no_speech_prob),
            )
            out.append(utt)
            self._utterances.append(utt)
            self._jsonl.write(json.dumps(asdict(utt), ensure_ascii=False) + "\n")
        self._jsonl.flush()
        return out

    # -- results ----------------------------------------------------------
    def utterances(self) -> list[Utterance]:
        return sorted(self._utterances, key=lambda u: u.start)

    def close(self) -> None:
        try:
            self._jsonl.close()
        except Exception:  # noqa: BLE001
            pass


def transcribe_file(settings: Settings, path: Path, speaker: str = "Speaker") -> list[Utterance]:
    """Offline path: transcribe an existing recording in one pass.

    Used by the ``file`` subcommand and as the fallback when live capture is not
    possible. faster-whisper decodes the container itself (via PyAV), so this
    accepts mp4/m4a/wav/mp3 without a separate ffmpeg step.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.model_size, device=settings.device, compute_type=settings.compute_type
    )
    segments, _info = model.transcribe(
        str(path),
        language=settings.language,
        beam_size=max(settings.beam_size, 5),
        vad_filter=settings.vad_filter,
        initial_prompt=settings.initial_prompt,
    )
    out: list[Utterance] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        out.append(
            Utterance(
                track="file",
                speaker=speaker,
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                avg_logprob=float(seg.avg_logprob),
                no_speech_prob=float(seg.no_speech_prob),
            )
        )
    return out


def merge_adjacent(utterances: list[Utterance], gap: float = 2.0) -> list[Utterance]:
    """Glue consecutive utterances from the same speaker into paragraphs.

    Whisper emits one segment per ~sentence. Left alone that produces a
    transcript that is 400 one-line rows; merging by speaker makes it readable
    and cuts the token count we send to the summarizer.
    """
    merged: list[Utterance] = []
    for u in sorted(utterances, key=lambda x: x.start):
        if merged and merged[-1].track == u.track and u.start - merged[-1].end <= gap:
            prev = merged[-1]
            merged[-1] = Utterance(
                track=prev.track,
                speaker=prev.speaker,
                start=prev.start,
                end=u.end,
                text=(prev.text + " " + u.text).strip(),
                avg_logprob=min(prev.avg_logprob, u.avg_logprob),
                no_speech_prob=max(prev.no_speech_prob, u.no_speech_prob),
            )
        else:
            merged.append(u)
    return merged


def render_markdown_transcript(utterances: list[Utterance], title: str) -> str:
    lines = [f"# Transcript — {title}", ""]
    for u in merge_adjacent(utterances):
        lines.append(f"**[{u.timestamp()}] {u.speaker}:** {u.text}")
        lines.append("")
    return "\n".join(lines)


def render_plain_transcript(utterances: list[Utterance]) -> str:
    return "\n".join(
        f"[{u.timestamp()}] {u.speaker}: {u.text}" for u in merge_adjacent(utterances)
    )


def utterances_from_jsonl(path: Path) -> list[Utterance]:
    out: list[Utterance] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Utterance(**d))
    return out
