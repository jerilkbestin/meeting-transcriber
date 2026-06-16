import argparse
import datetime
import os
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
BLACKHOLE_NAME = "BlackHole 2ch"
DEFAULT_MODEL_SIZE = "small.en"
DEFAULT_CHUNK_SECONDS = 10
DEFAULT_BEAM_SIZE = 1
DEFAULT_LANGUAGE = "en"
SILENCE_RMS = 0.0015
TARGET_RMS = 0.05
MAX_GAIN = 8.0


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


def make_callback(source, audio_queue):
    def callback(indata, frames, time, status):
        if status:
            print(f"\n{source} stream status: {status}")
        mono = indata.astype(np.float32, copy=True)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        audio_queue.put((source, mono.reshape(-1)))

    return callback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe mic audio plus BlackHole meeting audio."
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
    return parser.parse_args()


def prepare_audio(audio):
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if rms < SILENCE_RMS:
        return None

    gain = min(TARGET_RMS / max(rms, 1e-8), MAX_GAIN)
    if gain > 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


def transcribe_loop(model, audio_queue, output_file, args):
    buffers = {
        "You": np.array([], dtype=np.float32),
        "Others": np.array([], dtype=np.float32),
    }
    chunk_size = SAMPLE_RATE * args.chunk_seconds
    language = args.language.strip() or None
    initial_prompt = args.initial_prompt.strip() or None
    last_backlog_warning = 0.0

    with open(output_file, "a", encoding="utf-8") as transcript:
        while True:
            source, chunk = audio_queue.get()
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


def main():
    args = parse_args()
    devices = sd.query_devices()
    list_devices(devices)

    blackhole_index = find_blackhole_input(devices)
    mic_index = choose_mic_input(devices, blackhole_index)
    output_file = output_filename()

    mic_name = devices[mic_index]["name"]
    blackhole_name = devices[blackhole_index]["name"]

    print("")
    print(f"Mic input     : {mic_name}")
    print(f"Remote input  : {blackhole_name}")
    print(f"Transcript    : {output_file}")

    print("\nTranscription settings:")
    print(f"  Model        : {args.model}")
    print(f"  Chunk seconds: {args.chunk_seconds}")
    print(f"  Beam size    : {args.beam_size}")
    print(f"  Language     : {args.language.strip() or 'auto'}")

    print("\nLoading Whisper model... (first run downloads model files)")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(4, min((os.cpu_count() or 4), 8)),
        num_workers=1,
    )
    print("Model loaded.")
    print("Listening... Press Ctrl+C to stop.\n")

    audio_queue = queue.Queue()
    worker = threading.Thread(
        target=transcribe_loop,
        args=(model, audio_queue, output_file, args),
        daemon=True,
    )
    worker.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=stream_channels(devices, mic_index),
            dtype="float32",
            device=mic_index,
            callback=make_callback("You", audio_queue),
        ), sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=stream_channels(devices, blackhole_index),
            dtype="float32",
            device=blackhole_index,
            callback=make_callback("Others", audio_queue),
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print(f"\nTranscription saved to: {output_file}")
    except Exception as exc:
        print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
