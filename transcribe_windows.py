import argparse
import datetime
import os
import queue
import re
import threading
import time
import warnings

import numpy as np
import soundcard as sc
from soundcard.mediafoundation import SoundcardRuntimeWarning
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = 48000
DEFAULT_MODEL_SIZE = "small.en"
DEFAULT_CHUNK_SECONDS = 10
DEFAULT_BEAM_SIZE = 1
DEFAULT_LANGUAGE = "en"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
SILENCE_RMS = 0.0015
TARGET_RMS = 0.05
MAX_GAIN = 8.0


warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    category=SoundcardRuntimeWarning,
)


def sanitize_filename_component(value):
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value


def output_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = input("\nTranscript name (blank for 'transcript'): ")
    prefix = sanitize_filename_component(name) or "transcript"
    return f"{prefix}_{timestamp}.txt"


def device_label(device):
    name = getattr(device, "name", "")
    identifier = getattr(device, "id", "")
    if identifier and identifier != name:
        return f"{name} [{identifier}]"
    return name


def is_loopback(device):
    if bool(getattr(device, "isloopback", False)):
        return True
    text = f"{getattr(device, 'name', '')} {getattr(device, 'id', '')}".lower()
    return "loopback" in text


def list_windows_audio_devices(mics, loopbacks, speakers):
    print("\nWindows Audio Devices:")
    print("-" * 80)
    print("Microphones:")
    for i, device in enumerate(mics):
        print(f"  [{i}] {device_label(device)}")

    print("\nSpeaker loopback capture devices:")
    for i, device in enumerate(loopbacks):
        print(f"  [{i}] {device_label(device)}")

    print("\nPlayback devices:")
    for i, device in enumerate(speakers):
        print(f"  [{i}] {device_label(device)}")
    print("-" * 80)


def choose_device(devices, title):
    if not devices:
        raise RuntimeError(f"No {title.lower()} devices were found.")

    print(f"\n{title}:")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device_label(device)}")

    while True:
        value = input(f"\nEnter {title.lower()} number: ").strip()
        if value.isdigit() and int(value) in range(len(devices)):
            return devices[int(value)]
        print("Invalid device number.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a Windows microphone plus WASAPI speaker loopback with faster-whisper."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("WHISPER_MODEL", DEFAULT_MODEL_SIZE),
        help=f"faster-whisper model to use. Default: {DEFAULT_MODEL_SIZE}",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=int(os.getenv("WHISPER_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)),
        help=f"seconds per transcription chunk. Default: {DEFAULT_CHUNK_SECONDS}",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.getenv("WHISPER_BEAM_SIZE", DEFAULT_BEAM_SIZE)),
        help=f"beam size. 1 is fastest; 3-5 can improve accuracy. Default: {DEFAULT_BEAM_SIZE}",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("WHISPER_LANGUAGE", DEFAULT_LANGUAGE),
        help=f"language hint, or empty string for auto-detect. Default: {DEFAULT_LANGUAGE}",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.getenv("WHISPER_INITIAL_PROMPT", ""),
        help="optional vocabulary/context prompt for names, acronyms, or project terms.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("WHISPER_DEVICE", DEFAULT_DEVICE),
        choices=["cuda", "cpu", "auto"],
        help=f"Whisper runtime device. Default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--compute-type",
        default=os.getenv("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE),
        help=f"ctranslate2 compute type. Use float16 on NVIDIA, int8 on CPU. Default: {DEFAULT_COMPUTE_TYPE}",
    )
    parser.add_argument(
        "--playback-label",
        default=os.getenv("PLAYBACK_TRANSCRIPT_LABEL", "Others"),
        help="label for audio captured from the selected speaker loopback. Default: Others",
    )
    return parser.parse_args()


def prepare_audio(audio):
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms < SILENCE_RMS:
        return None

    gain = min(TARGET_RMS / max(rms, 1e-8), MAX_GAIN)
    if gain > 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def resample_audio(audio, source_rate, target_rate):
    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def capture_loop(source, device, audio_queue, stop_event):
    block_frames = CAPTURE_SAMPLE_RATE
    try:
        with device.recorder(samplerate=CAPTURE_SAMPLE_RATE, channels=1, blocksize=block_frames) as recorder:
            while not stop_event.is_set():
                data = recorder.record(numframes=block_frames)
                audio = np.asarray(data, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = resample_audio(audio.reshape(-1), CAPTURE_SAMPLE_RATE, SAMPLE_RATE)
                audio_queue.put((source, audio.reshape(-1)))
    except Exception as exc:
        audio_queue.put(("__error__", f"{source} capture failed: {exc}"))


def cuda_runtime_hint(exc):
    message = str(exc)
    lowered = message.lower()
    cuda_dll_names = (
        "cublas64",
        "cudnn",
        "cudart64",
        "cufft64",
        "curand64",
    )
    if any(name in lowered for name in cuda_dll_names):
        return (
            f"{message}\n\n"
            "The NVIDIA/CUDA runtime is missing from PATH. "
            "To keep testing now, rerun with: "
            "python transcribe_windows.py --device cpu --compute-type int8"
        )
    return message


def transcribe_loop(model, audio_queue, output_file, args, stop_event):
    buffers = {
        "You": np.array([], dtype=np.float32),
        args.playback_label: np.array([], dtype=np.float32),
    }
    chunk_size = SAMPLE_RATE * args.chunk_seconds
    language = args.language.strip() or None
    initial_prompt = args.initial_prompt.strip() or None
    last_backlog_warning = 0.0

    with open(output_file, "a", encoding="utf-8") as transcript:
        while not stop_event.is_set():
            source, chunk = audio_queue.get()
            if source == "__error__":
                print(f"\nError: {chunk}")
                stop_event.set()
                break

            buffers[source] = np.concatenate([buffers[source], chunk])

            backlog_seconds = sum(len(buffer) for buffer in buffers.values()) / SAMPLE_RATE
            now = time.monotonic()
            if backlog_seconds > args.chunk_seconds * 4 and now - last_backlog_warning > 30:
                print(
                    f"\nWarning: transcription is behind by about {int(backlog_seconds)} seconds. "
                    "Use a smaller model or lower beam size if this keeps growing."
                )
                last_backlog_warning = now

            while len(buffers[source]) >= chunk_size:
                audio = buffers[source][:chunk_size]
                buffers[source] = buffers[source][chunk_size:]
                audio = prepare_audio(audio)
                if audio is None:
                    continue

                try:
                    segments, _ = model.transcribe(
                        audio,
                        language=language,
                        beam_size=args.beam_size,
                        best_of=1,
                        temperature=0,
                        vad_filter=True,
                        condition_on_previous_text=False,
                        initial_prompt=initial_prompt,
                    )
                    for segment in segments:
                        text = segment.text.strip()
                        if not text:
                            continue
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        line = f"[{timestamp}] {source}: {text}"
                        print(line)
                        transcript.write(line + "\n")
                        transcript.flush()
                except Exception as exc:
                    print(f"\nTranscription failed: {cuda_runtime_hint(exc)}")
                    stop_event.set()
                    break


def main():
    args = parse_args()
    microphones = list(sc.all_microphones(include_loopback=False))
    all_capture_devices = list(sc.all_microphones(include_loopback=True))
    loopbacks = [device for device in all_capture_devices if is_loopback(device)]
    speakers = list(sc.all_speakers())

    list_windows_audio_devices(microphones, loopbacks, speakers)

    mic = choose_device(microphones, "MIC input device for You")
    loopback = choose_device(loopbacks, f"SPEAKER LOOPBACK input device for {args.playback_label}")
    output_file = output_filename()

    print("")
    print(f"Mic input       : {device_label(mic)}")
    print(f"Playback loopback: {device_label(loopback)}")
    print(f"Transcript      : {output_file}")
    print("")
    print("Any audio played through the matching Windows playback device will be captured.")
    print("For meetings, set the meeting app's Speaker to the playback device that matches this loopback.")

    print("\nTranscription settings:")
    print(f"  Model        : {args.model}")
    print(f"  Device       : {args.device}")
    print(f"  Compute type : {args.compute_type}")
    print(f"  Chunk seconds: {args.chunk_seconds}")
    print(f"  Beam size    : {args.beam_size}")
    print(f"  Language     : {args.language.strip() or 'auto'}")
    print(f"  Playback label: {args.playback_label}")

    print("\nLoading Whisper model... (first run downloads model files)")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=max(4, min((os.cpu_count() or 4), 8)),
        num_workers=1,
    )
    print("Model loaded.")
    print("Listening... Press Ctrl+C to stop.\n")

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=transcribe_loop,
        args=(model, audio_queue, output_file, args, stop_event),
        daemon=True,
    )
    mic_worker = threading.Thread(
        target=capture_loop,
        args=("You", mic, audio_queue, stop_event),
        daemon=True,
    )
    loopback_worker = threading.Thread(
        target=capture_loop,
        args=(args.playback_label, loopback, audio_queue, stop_event),
        daemon=True,
    )

    worker.start()
    mic_worker.start()
    loopback_worker.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\nTranscription saved to: {output_file}")


if __name__ == "__main__":
    main()
