"""Windows audio capture: microphone + WASAPI speaker loopback.

Why two tracks
--------------
Real speaker diarization (telling Alice from Bob inside one audio stream) is a
hard, error-prone problem. But on a Teams call there is a free, perfectly
reliable split available: your microphone carries *you*, and the WASAPI
loopback device carries *everyone else* (it is a copy of what the OS sends to
your speakers). Capturing them as two independent tracks gives us honest
two-way attribution with zero ML guesswork.

The catch: this only holds if you wear headphones. On open speakers the mic
also picks up the remote audio and both tracks contain everybody.
"""

from __future__ import annotations

import queue
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TARGET_SR

try:  # pragma: no cover - import guard exercised only on non-Windows
    import pyaudiowpatch as pyaudio

    PYAUDIO_AVAILABLE = True
    PYAUDIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    pyaudio = None  # type: ignore[assignment]
    PYAUDIO_AVAILABLE = False
    PYAUDIO_IMPORT_ERROR = exc

try:
    import soxr

    _HAVE_SOXR = True
except Exception:  # noqa: BLE001
    soxr = None  # type: ignore[assignment]
    _HAVE_SOXR = False


class AudioUnavailable(RuntimeError):
    """Raised when the platform cannot provide the requested capture device."""


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool


def _require_pyaudio() -> None:
    if not PYAUDIO_AVAILABLE:
        raise AudioUnavailable(
            "PyAudioWPatch is not importable, so live capture is unavailable on "
            f"this machine (Windows-only package). Underlying error: "
            f"{PYAUDIO_IMPORT_ERROR!r}\n"
            "Install it with:  pip install PyAudioWPatch"
        )
    if not sys.platform.startswith("win"):
        raise AudioUnavailable(
            "WASAPI loopback capture only exists on Windows. On macOS install a "
            "virtual loopback device (BlackHole); on Linux use a PulseAudio "
            "'.monitor' source. Either way, then run this tool against the "
            "resulting input device with --system-device."
        )


def resample_to_target(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz.

    soxr is a high-quality band-limited resampler and is what we want for ASR
    input. If it is missing we fall back to linear interpolation, which aliases
    a little but still transcribes acceptably -- better than refusing to run.
    """
    if src_rate == TARGET_SR:
        return samples.astype(np.float32, copy=False)
    if _HAVE_SOXR:
        return soxr.resample(samples, src_rate, TARGET_SR).astype(np.float32, copy=False)
    ratio = TARGET_SR / float(src_rate)
    n_out = int(round(len(samples) * ratio))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def to_mono_float32(raw: bytes, channels: int) -> np.ndarray:
    """Convert an interleaved int16 buffer into mono float32 in [-1, 1]."""
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return (data.astype(np.float32) / 32768.0).astype(np.float32)


def list_devices() -> list[DeviceInfo]:
    """Enumerate every input-capable device, including WASAPI loopbacks."""
    _require_pyaudio()
    devices: list[DeviceInfo] = []
    pa = pyaudio.PyAudio()
    try:
        loopback_indices = set()
        try:
            for lb in pa.get_loopback_device_info_generator():
                loopback_indices.add(int(lb["index"]))
        except Exception:  # noqa: BLE001 - generator absent on odd configs
            pass
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            devices.append(
                DeviceInfo(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    channels=int(info["maxInputChannels"]),
                    sample_rate=int(info["defaultSampleRate"]),
                    is_loopback=int(info["index"]) in loopback_indices
                    or bool(info.get("isLoopbackDevice", False)),
                )
            )
    finally:
        pa.terminate()
    return devices


def find_default_loopback(pa) -> DeviceInfo:
    """Locate the loopback device that mirrors the default speakers.

    PyAudioWPatch exposes a convenience helper, but it has not existed in every
    release, so we fall back to the documented manual walk from the project's
    own example.
    """
    getter = getattr(pa, "get_default_wasapi_loopback", None)
    if callable(getter):
        try:
            info = getter()
            if info:
                return DeviceInfo(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    channels=int(info["maxInputChannels"]),
                    sample_rate=int(info["defaultSampleRate"]),
                    is_loopback=True,
                )
        except Exception:  # noqa: BLE001 - fall through to manual lookup
            pass

    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:  # pragma: no cover
        raise AudioUnavailable("WASAPI host API is not available on this system.") from exc

    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if not speakers.get("isLoopbackDevice", False):
        for lb in pa.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                speakers = lb
                break
        else:
            raise AudioUnavailable(
                "Could not find a loopback device matching your default speakers.\n"
                "Run:  python -m pyaudiowpatch     to inspect available devices, "
                "then pass --system-device <index>."
            )
    return DeviceInfo(
        index=int(speakers["index"]),
        name=str(speakers["name"]),
        channels=int(speakers["maxInputChannels"]),
        sample_rate=int(speakers["defaultSampleRate"]),
        is_loopback=True,
    )


def find_default_mic(pa) -> DeviceInfo:
    info = pa.get_default_input_device_info()
    return DeviceInfo(
        index=int(info["index"]),
        name=str(info["name"]),
        channels=min(int(info["maxInputChannels"]), 2),
        sample_rate=int(info["defaultSampleRate"]),
        is_loopback=False,
    )


class WavWriter:
    """Tiny append-only 16-bit WAV sink, written incrementally.

    Writing as we go (rather than buffering the whole meeting) means a crash or
    a hard kill still leaves a playable recording on disk.
    """

    def __init__(self, path: Path, sample_rate: int = TARGET_SR):
        self.path = path
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        self._lock = threading.Lock()
        self._closed = False

    def write(self, samples: np.ndarray) -> None:
        if self._closed:
            return
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with self._lock:
            self._wav.writeframes(pcm.tobytes())

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._wav.close()


class TrackCapture:
    """One capture stream -> resampled mono 16 kHz -> a queue of numpy blocks.

    PortAudio delivers audio on its own high-priority thread. We do the absolute
    minimum there (convert, resample, enqueue) and let a consumer thread do the
    expensive transcription, so we never stall the audio callback and drop
    frames.
    """

    def __init__(
        self,
        pa,
        device: DeviceInfo,
        name: str,
        out_queue: "queue.Queue[tuple[str, np.ndarray]]",
        wav_writer: WavWriter | None = None,
        frames_per_buffer: int = 1024,
    ):
        self.pa = pa
        self.device = device
        self.name = name
        self.queue = out_queue
        self.wav_writer = wav_writer
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self.dropped_blocks = 0

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001
        try:
            mono = to_mono_float32(in_data, self.device.channels)
            mono = resample_to_target(mono, self.device.sample_rate)
            if self.wav_writer is not None:
                self.wav_writer.write(mono)
            try:
                self.queue.put_nowait((self.name, mono))
            except queue.Full:
                self.dropped_blocks += 1
        except Exception:  # noqa: BLE001 - never raise inside the audio callback
            self.dropped_blocks += 1
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.device.channels,
            rate=self.device.sample_rate,
            frames_per_buffer=self.frames_per_buffer,
            input=True,
            input_device_index=self.device.index,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
