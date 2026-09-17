# Phase 3 — Running a meeting with speaker diarization

How to capture a Teams meeting so that afterwards you get a transcript where
each remote participant is labelled separately (`Others 1`, `Others 2`, …)
instead of a single `Others`.

The split is done in two stages on purpose:

- **During the meeting** you only capture and transcribe. Recording costs
  almost nothing.
- **After the meeting** you run the diarizer. It needs the whole recording at
  once, because it groups voices by comparing them against each other - which
  is also why per-chunk live diarization cannot keep a speaker's identity
  stable from one chunk to the next.

---

## One-time setup

Already done on this machine; listed so it can be repeated elsewhere.

```bash
cd ~/meeting-transcriber
source venv/bin/activate

pip install -r requirements-diarize.txt   # pyannote.audio 4 + torch
hf auth login                             # paste a read token from
                                          # https://huggingface.co/settings/tokens
```

Accept the model terms once at
<https://huggingface.co/pyannote/speaker-diarization-community-1>.

`hf auth login` stores the token in `~/.cache/huggingface/token`, so it
survives restarts and needs no `HF_TOKEN` export. After the first run the model
(32 MB) is cached and diarization works with no network and no token at all.

---

## Before the meeting (about 2 minutes)

```bash
# 1. Route the meeting audio. Pick the output you want to HEAR through.
switch_meeting_output
```

Then in **Teams → Settings → Devices**, set the **Speaker** to
`Multi-Output Device`. Leave the microphone as your normal mic.

This is what makes the other participants capturable: the Multi-Output Device
sends their audio both to your headset (so you hear it) and to `BlackHole 2ch`
(so it can be recorded as an input).

```bash
# 2. Optional, worth doing the first few times: confirm BlackHole hears audio.
python test_audio.py       # choose BlackHole 2ch as input, option [1],
                           # play anything and watch the level bars move
```

**Check free disk space** if the meeting is long: recording costs about
**230 MB per hour** for both streams together.

---

## Starting the transcriber

```bash
cd ~/meeting-transcriber
source venv/bin/activate
python transcribe.py --record
```

`--record` is the only difference from a normal session. Without it you get the
live transcript but no audio, and **diarization afterwards is impossible** -
there is nothing left to analyse.

You will be asked two things:

1. **MIC input device number** - your real microphone (the tool refuses
   BlackHole here, since that is the other side of the conversation).
2. **Transcript name** - short and meaningful, e.g. `seaspan-weekly`. Every
   output file for the session is named from it.

Then check the banner before the meeting starts:

```text
Mic input     : MacBook Pro Microphone
Remote input  : BlackHole 2ch
Transcript    : transcripts/seaspan-weekly_20260917_101500.txt
Recording     : transcripts/seaspan-weekly_20260917_101500_others.wav
Recording     : transcripts/seaspan-weekly_20260917_101500_you.wav
Stream info   : transcripts/seaspan-weekly_20260917_101500_streams.json
```

If the two `Recording` lines are missing, `--record` did not take effect - stop
and restart. That is the one mistake that cannot be repaired afterwards.

---

## During the meeting

Leave the terminal alone. Lines appear a few seconds behind the speech; this is
normal and does **not** affect the timestamps, which are taken when the audio
was captured, not when the line was printed.

Two things are worth reacting to:

| What you see | What it means | What to do |
| --- | --- | --- |
| `Warning: transcription is behind by about N seconds` and N keeps growing | Decoding is slower than real time | Nothing mid-meeting - the recording and timestamps stay correct. Next time use `--accuracy fast` |
| `Others stream status: input overflow` | The machine dropped audio blocks | Close heavy apps. The recorder fills the gap with silence so the timeline stays aligned |

**Do not run `diarize.py` while the meeting is running.** It would compete for
the same CPU the live transcription needs.

If Teams resets your speaker to something else mid-meeting (it does this when
headphones reconnect), the `Others` side goes silent. Set the Teams speaker
back to `Multi-Output Device`; capture resumes on its own.

### Ending

Press `Ctrl+C`. You should see:

```text
Transcription saved to: transcripts/seaspan-weekly_20260917_101500.txt
Recordings saved. Diarize with:
  python diarize.py transcripts/seaspan-weekly_20260917_101500_streams.json
```

That last line is the exact command for the next stage - copy it.

---

## After the meeting

```bash
python diarize.py transcripts/seaspan-weekly_20260917_101500_streams.json
```

If you know how many people were on the call (excluding yourself), say so. It
is the single biggest accuracy lever, because it stops the clustering from
splitting one voice in two or merging two quiet ones:

```bash
python diarize.py transcripts/..._streams.json --num-speakers 3
# or, if unsure:
python diarize.py transcripts/..._streams.json --min-speakers 2 --max-speakers 5
```

### How long it takes

Measured on this M4, CPU:

| Stage | Speed | For a 1-hour meeting |
| --- | --- | --- |
| Diarization (`Others` only) | RTF 0.38 | about 23 minutes |
| Re-transcription, `--accuracy max` (default) | RTF ~0.50 per stream | about 60 minutes for both streams |
| Re-transcription, `--accuracy balanced` | RTF ~0.27 per stream | about 32 minutes for both streams |

So a 1-hour meeting takes roughly **1h20m at the default**, or about **55
minutes** with `--accuracy balanced`. Start it and go do something else; it
needs no supervision.

To cut that down:

```bash
# skip re-transcribing your own mic (You lines come only from the live file)
python diarize.py transcripts/..._streams.json --skip-you

# faster, slightly less accurate decode
python diarize.py transcripts/..._streams.json --accuracy balanced
```

It re-transcribes rather than reusing the live text for two reasons: the live
`.txt` only keeps whole seconds, which is too coarse to match against speaker
turns, and the post pass can afford the `max` preset that live capture cannot.
The diarized transcript is therefore usually *better* text, not just relabelled
text.

### What you get

```text
transcripts/seaspan-weekly_20260917_101500_diarized.txt    # the readable result
transcripts/seaspan-weekly_20260917_101500_diarized.json   # per-line detail
```

```text
[10:15:03] You: right, let's get started
[10:15:07] Others 1: sounds good, I'll share my screen
[10:15:19] Others 2: can you make that a bit bigger
```

Timestamps are wall-clock, the same clock the live transcript used, so the two
files line up side by side. Use `--offsets` instead if you are sharing the
transcript and would rather show elapsed time from the start of the meeting.

`Others 1` is whoever spoke first, `Others 2` second, and so on. Numbering is
by first appearance, so it is stable if you run the command twice, but it is
**not** stable across different meetings - `Others 1` is a different person
next week.

Useful extras:

```bash
--mark-uncertain    # append [?] to lines that straddle a speaker change
--offsets           # timestamp from meeting start instead of wall clock
--device mps        # try the GPU; opt-in, some ops fall back to CPU anyway
```

The `.json` is worth opening when a label looks wrong. Each line carries
`shares` (how many seconds each speaker held of that segment) and `straddled`
(whether a second speaker had a substantial share), which tells you whether the
label was clear-cut or a close call.

---

## Then: meeting notes

Run the `/meeting-notes` skill against the **diarized** transcript rather than
the live one - knowing who said what materially improves the summary and the
action items. Notes are written to `notes/`.

---

## Putting names to the speakers

Diarization gives you `Others 1`/`Others 2`, not names - it can tell voices
apart but has never been told who they belong to. Skim the first line of each
speaker; in most meetings people identify themselves early, and a
find-and-replace over the diarized `.txt` takes a few seconds:

```bash
sed -i '' 's/Others 1:/Priya:/g; s/Others 2:/Marc:/g' \
  transcripts/seaspan-weekly_20260917_101500_diarized.txt
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| No `Recording` lines in the startup banner | `--record` missing | Restart with `--record`. Cannot be fixed after the fact |
| `Others` transcript is empty or silent | Teams speaker is not the Multi-Output Device | Set it in Teams → Devices, re-run `switch_meeting_output` if the device is gone |
| `BlackHole 2ch was not found as an input device` | BlackHole not installed or not visible | Check Audio MIDI Setup; reinstall BlackHole if absent |
| `access to pyannote/... has not been granted` | Terms not accepted, or no token | Accept at the model page, then `hf auth login` |
| `pyannote.audio is not installed` | Diarization deps missing | `pip install -r requirements-diarize.txt` |
| Everyone lands in one `Others 1` | Voices too similar, or the clustering merged them | Re-run with `--num-speakers N` |
| One person split into two labels | Clustering over-split, often from varying mic quality | Re-run with `--num-speakers N` |
| `objc[...] Class AVFFrameReceiver is implemented in both...` | PyAV and Homebrew ffmpeg both loaded | Harmless, ignore |

---

## Housekeeping

The WAVs are the largest artefact and are only needed until you have diarized.
They are gitignored and never committed, but they do accumulate:

```bash
# after diarizing and checking the result, reclaim the space
rm transcripts/seaspan-weekly_20260917_101500_{you,others}.wav
```

Keep the `_streams.json` if you want a record of the session's timing metadata;
it is tiny. Deleting the WAVs means the meeting can never be re-diarized, so do
it only once you are happy with the diarized transcript.
