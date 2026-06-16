import time
import warnings

import numpy as np
import soundcard as sc
from soundcard.mediafoundation import SoundcardRuntimeWarning


SAMPLE_RATE = 48000


warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    category=SoundcardRuntimeWarning,
)


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


def choose_device(devices, title):
    if not devices:
        raise RuntimeError(f"No {title.lower()} were found.")

    print(f"\n{title}:")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device_label(device)}")

    while True:
        value = input(f"\nEnter {title.lower()} number: ").strip()
        if value.isdigit() and int(value) in range(len(devices)):
            return devices[int(value)]
        print("Invalid device number.")


def print_level(audio):
    volume = float(np.linalg.norm(audio)) * 10
    bars = min(int(volume), 40)
    level = "#" * bars
    space = " " * (40 - bars)
    print(f"\r[{level}{space}] {volume:.2f}", end="", flush=True)


def monitor_capture(device):
    print(f"\nListening to [{device_label(device)}]... make some noise or play audio through that device. Ctrl+C to stop.\n")
    try:
        with device.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=SAMPLE_RATE // 10) as recorder:
            while True:
                audio = recorder.record(numframes=SAMPLE_RATE // 10)
                print_level(audio)
    except KeyboardInterrupt:
        print("\nDone.")
    except Exception as exc:
        print(f"\nError: {exc}")


def play_tone(speaker):
    print(f"\nPlaying test tone to [{device_label(speaker)}]...")
    duration = 3
    frequency = 440
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = (np.sin(2 * np.pi * frequency * t) * 0.25).astype(np.float32)
    stereo = np.column_stack([tone, tone])

    try:
        with speaker.player(samplerate=SAMPLE_RATE, channels=2) as player:
            player.play(stereo)
            time.sleep(duration)
        print("Done.")
    except Exception as exc:
        print(f"\nError: {exc}")


def main():
    microphones = list(sc.all_microphones(include_loopback=False))
    capture_devices = list(sc.all_microphones(include_loopback=True))
    loopbacks = [device for device in capture_devices if is_loopback(device)]
    speakers = list(sc.all_speakers())

    print("\nWindows Audio Devices")
    print("-" * 80)
    print("Microphones:")
    for i, device in enumerate(microphones):
        print(f"  [{i}] {device_label(device)}")

    print("\nSpeaker loopback capture devices:")
    for i, device in enumerate(loopbacks):
        print(f"  [{i}] {device_label(device)}")

    print("\nPlayback devices:")
    for i, device in enumerate(speakers):
        print(f"  [{i}] {device_label(device)}")
    print("-" * 80)

    print("\nOptions:")
    print("  [1] Test microphone/input levels")
    print("  [2] Test speaker loopback levels")
    print("  [3] Play output test tone")

    choice = input("\nEnter choice (1/2/3): ").strip()
    if choice == "1":
        monitor_capture(choose_device(microphones, "MIC input devices"))
    elif choice == "2":
        monitor_capture(choose_device(loopbacks, "SPEAKER LOOPBACK input devices"))
    elif choice == "3":
        play_tone(choose_device(speakers, "OUTPUT playback devices"))
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
