import sounddevice as sd
import numpy as np

# Print all available devices
print("\n📋 All Available Audio Devices:")
print("-" * 60)
devices = sd.query_devices()
for i, d in enumerate(devices):
    ins = f"in: {d['max_input_channels']}ch" if d['max_input_channels'] > 0 else "no input"
    outs = f"out: {d['max_output_channels']}ch" if d['max_output_channels'] > 0 else "no output"
    print(f"  [{i}] {d['name']} ({ins}, {outs})")
print("-" * 60)

# Pick input device
print("\n🎙️ INPUT DEVICES (capture/listen from):")
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"  [{i}] {d['name']}")

input_index = int(input("\nEnter INPUT device number to test: "))
input_name = devices[input_index]['name']

# Pick output device
print("\n🔊 OUTPUT DEVICES (play audio to):")
for i, d in enumerate(devices):
    if d['max_output_channels'] > 0:
        print(f"  [{i}] {d['name']}")

output_index = int(input("\nEnter OUTPUT device number to test: "))
output_name = devices[output_index]['name']

SAMPLE_RATE = 16000

print(f"\n✅ Input  : {input_name}")
print(f"✅ Output : {output_name}")
print("\nOptions:")
print("  [1] Test INPUT only - show audio level bars")
print("  [2] Test OUTPUT only - play a test tone")
print("  [3] Test BOTH - capture input and play through output (loopback)")

choice = input("\nEnter choice (1/2/3): ").strip()

# ── Option 1: Test Input ──────────────────────────────────────────
if choice == "1":
    print(f"\n🎙️ Listening to [{input_name}]... make some noise! (Ctrl+C to stop)\n")

    def audio_callback(indata, frames, time, status):
        volume = np.linalg.norm(indata) * 10
        bars = int(volume)
        level = "█" * min(bars, 40)
        space = " " * (40 - min(bars, 40))
        print(f"\r[{level}{space}] {volume:.2f}", end="", flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='float32', device=input_index,
                            callback=audio_callback):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print(f"\n\n✅ Done testing input: {input_name}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# ── Option 2: Test Output ─────────────────────────────────────────
elif choice == "2":
    print(f"\n🔊 Playing test tone to [{output_name}]... (Ctrl+C to stop)\n")
    duration = 3  # seconds
    frequency = 440  # Hz (A note)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)

    try:
        sd.play(tone, samplerate=SAMPLE_RATE, device=output_index)
        sd.wait()
        print(f"✅ Done. Did you hear the tone on [{output_name}]?")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# ── Option 3: Loopback (Input → Output) ──────────────────────────
elif choice == "3":
    print(f"\n🔁 Loopback: [{input_name}] → [{output_name}]")
    print("Speak into the input device, you should hear it on the output. (Ctrl+C to stop)\n")

    def loopback_callback(indata, outdata, frames, time, status):
        volume = np.linalg.norm(indata) * 10
        bars = int(volume)
        level = "█" * min(bars, 40)
        space = " " * (40 - min(bars, 40))
        print(f"\r[{level}{space}] {volume:.2f}", end="", flush=True)
        outdata[:] = indata

    try:
        with sd.Stream(samplerate=SAMPLE_RATE, channels=1,
                       dtype='float32',
                       device=(input_index, output_index),
                       callback=loopback_callback):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print(f"\n\n✅ Done loopback test.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

else:
    print("Invalid choice.")