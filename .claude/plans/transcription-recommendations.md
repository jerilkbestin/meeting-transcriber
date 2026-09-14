# Transcription Improvement Recommendations (plain English)

This document explains, in everyday language, the nine recommendations for improving the meeting transcriber. They came out of two things:

1. An analysis of 10 recent transcripts, which found about **402 transcription mistakes** across roughly 60,000 words (~0.66% of words). Most mistakes were **names and technical product terms** (Databricks, Lakebase, ABAC/RBAC, Entra, Cavallo, and people's names), plus some **"stuck record" repetitions** where the same phrase repeats many times.
2. A **code audit** (`AUDIT.md`) of the transcriber itself, which found reliability issues that can quietly corrupt or lose the transcript.

Each recommendation below is written as: **the problem -> the recommendation -> why it helps -> what we decided -> where it runs**. The decisions reflect your feedback during the discussion.

> Note on the true error rate: 0.66% sounds tiny, but it is understated. The two audio channels (your voice and everyone else's) overlap and get transcribed twice, and long "stuck record" repetitions pad the word count. The errors that remain are concentrated exactly where they hurt most - the names and product terms you actually care about.

---

## 1. Duplicate and garbled repeats

**The problem.** The transcripts had the same speech appearing twice and long runs of a repeated phrase (for example "it didn't work" printed 28 times in a row).

**My original recommendation was wrong.** I guessed this came from listening on speakers, so your microphone re-recorded other people's voices. You corrected me: you use wired earphones 99% of the time, which rules that out. I am not going to replace one guess with another.

**What it most likely is.** Two things, neither of which is "speaker bleed":
- **"Stuck record" hallucinations.** Whisper (the speech-to-text engine) is known to loop on near-silent or low-quality audio and repeat a phrase over and over. This is a decoding quirk, not a recording problem.
- **Over-counting.** When both speakers talk about the same topic, an automated counter can mistake similar content for literal duplication.

**Why the fix helps.** The concrete fixes for the looping are covered by two other recommendations: cutting audio at natural pauses (#6) and turning on Whisper's built-in "this looks like nonsense, discard it" safety checks (part of #3's server settings).

**What we decided.** Drop the headphones advice. Diagnose the duplication properly, then rely on #6 and the anti-looping settings rather than any hardware change.

**Where it runs.** Diagnosis is a quick look at one transcript; the fixes live in the chunking logic (your Mac) and the model server settings.

---

## 2. Names and jargon get misheard

**The problem.** The biggest single source of errors is proper nouns: product names (Databricks, Lakebase, Unity Catalog, Purview, Entra), acronyms (ABAC, RBAC), the company name (Cavallo), client names (Seaspan, Samara, Co-operators), and people's names (Sergey, Jeril, Aayushi, Lior, Amine, Rakhi). The engine has never been trained to expect these, so it substitutes ordinary words that sound similar.

**The recommendation.** Turn on **"hotword" biasing** by default. Hotwords are a short list of important terms you hand the engine up front; it then leans toward hearing those terms when the audio is ambiguous. The list is built automatically from the "Correct term" column of your existing glossary (`~/.claude/meeting-notes-glossary.md`), plus any extra names you type in when you start a meeting.

**Why it helps.** This fixes the errors at the moment of listening, using context, rather than find-and-replace after the fact. It directly targets the category that produces most of the 402 mistakes.

**What we decided.**
- Hotwords should be **on by default** (no flag to remember).
- Two important constraints:
  - **There is a size limit.** The engine only accepts roughly 200 words of hints. Your glossary has 368 entries, far too many to send all at once. So we send a curated "core" list (the names and products that come up constantly) and leave the long tail to the after-the-meeting notes step.
  - **Timing matters.** Hotwords must be given *before/at the start* of the meeting, to the transcriber. Names you mention *afterward* to the notes tool can only fix the written notes, not what the engine actually heard. So there will be a short "any special names today?" prompt when you launch the transcriber.

**Where it runs.** On your Mac, at transcriber startup, and sent along with every request to the model server.

---

## 3. Bigger, more accurate model on a GPU

**The problem.** The transcriber currently uses a small English model with the fastest (least accurate) decoding setting, running on the Mac's CPU. That trade favors speed over getting names right.

**The recommendation.** Move the heavy lifting to a **GPU model server** and use a much stronger model: **`large-v3-turbo`** with **8-bit ("int8") compression** and **beam size 5** (a setting that lets the engine consider several interpretations and pick the best, instead of blindly taking the first).

**Why it helps.** A larger model is dramatically better at proper nouns. On a GPU, the higher-quality "beam" setting is cheap, and beam size above 1 is also what makes the hotwords in #2 actually take effect. Verified fit: `large-v3-turbo` in int8 uses roughly 1.5 GB of video memory, well within the RTX 4050's ~6 GB.

**What we decided.**
- Model: `large-v3-turbo`, int8, beam 5 (float16 is a higher-quality option to benchmark later).
- **One live stream/workload only** - no separate slower "archival" re-run.
- Build the **client-server split now**: the Mac captures audio and sends it over a **direct wired Ethernet cable** to a Windows laptop (ASUS ROG Strix, RTX 4050) that runs the model. Full setup is in `project-local-model-setup.md`.

**Trade-off to be aware of.** Sending audio to a separate machine means the audio now leaves the Mac (over your own private cable, not the internet). It also is not a simple config change: the GPU machine needs a small server program wrapped around the model.

**Where it runs.** The model runs on the Windows GPU laptop (L1); the Mac (L2) becomes a thin client.

---

## 4. Don't lose audio that only lives in memory

**The problem.** You raised the right concern: if a piece of audio exists only in the computer's memory (RAM) while waiting to be transcribed, and the program crashes or the network drops, that audio is gone forever - you cannot re-record a meeting.

**The recommendation.** Write every second of captured audio straight to a **WAV file on disk as it arrives**, independently of transcription. The in-memory queue stays as the fast live path, but the disk file is the durable safety copy. If transcription lags or the GPU server drops, the audio is still safe on disk and can be re-sent or re-transcribed.

**Why it helps.** It separates "capturing the meeting" (must never fail) from "transcribing the meeting" (can be retried). RAM is volatile; disk is not.

**The memory math you asked about** (16 kHz mono audio):
- In memory (float format): about **3.84 MB per minute** per stream, so ~230 MB per hour.
- On disk (standard WAV, int16): about **1.9 MB per minute** per stream, so ~115 MB per hour.
- Both streams to disk: about **230 MB per hour** total. A 2-hour meeting is roughly 460 MB - trivial.

So RAM can easily hold hours of audio, but that is not the point: the disk copy is the insurance, and it is also the exact file a re-transcription would use.

**What we decided.** Keep both audio streams (your voice + everyone else), live-only, and add the write-to-disk safety copy using Python's built-in WAV support (no new software needed).

**Where it runs.** On your Mac, during capture.

---

## 5. Auto-correcting the raw transcript file (decided against)

**The problem/idea.** We could add a standalone step that automatically applies the glossary corrections (for example, replace "data breaks" with "Databricks") directly to the raw transcript text file.

**Why we are skipping it.** It mostly duplicates work already done elsewhere:
- Your `/meeting-notes` tool already applies the glossary when it turns a transcript into notes.
- With hotwords now fixing many terms at the listening stage (#2), fewer errors survive to be corrected.
- Blind find-and-replace is risky: several glossary entries are ordinary English words in some contexts (for example "saw" can mean the tool or the acronym "SOW"), so automatic replacement could corrupt normal sentences.

**What we decided.** Keep glossary correction in **one place** (the notes tool). Only revisit a standalone filter if you start feeding raw transcripts directly into other tools that never pass through the notes step.

**Where it runs.** N/A - intentionally not built.

---

## 6. Cut audio at pauses, not on a fixed timer

**The problem.** The transcriber currently slices audio into fixed 10-second pieces. A 10-second boundary often lands in the middle of a word or sentence, which garbles the words on either side and, because each piece is transcribed independently, there is no chance to repair the split.

**The recommendation.** Cut each piece at a **natural silence gap** instead of on a stopwatch, so each piece is a complete thought.

**Why it helps.** Whole sentences transcribe far more accurately than sentences chopped mid-word, and it reduces the "stuck record" looping from #1.

**What we decided.** Approved. We can reuse the loudness measurement the code already computes (no new software needed) to detect the pauses.

**Where it runs.** On your Mac, in the chunking logic.

---

## 7. Stop dropping audio when the computer is busy (audit finding #1)

**The problem.** There is a small, time-critical piece of code (the "audio callback") that the sound system calls many times per second to hand over freshly captured audio. It has a hard deadline: finish before the next batch arrives, or the system throws away audio. Right now that code does more work than it should inside the deadline (it allocates memory and does math there). Today there is enough slack that it rarely misses. But recommendations #2/#3/#6 make the rest of the program work much harder, and under that load this code can start missing its deadline and dropping audio - which shows up as missing or garbled words.

**The recommendation.** Make that time-critical code do the bare minimum - just grab the audio and hand it off - and move the math to a background worker that has no deadline.

**Why it helps.** It removes the unpredictable work from the deadline path, so captured audio is never dropped even when the machine is busy. This is a prerequisite for safely turning on the heavier model.

**What we decided.** Approved, and it should be done *first*, before turning up the model and beam settings.

**Where it runs.** On your Mac, in the capture code.

---

## 8. Make sure the audio is at the right sample rate (audit finding #5)

**The problem.** Whisper expects audio at 16,000 samples per second (16 kHz). The code asks the operating system for 16 kHz, but if a microphone or the BlackHole device actually runs at a different rate (many run at 48 kHz), the conversion is left to the operating system with no checking. A poor conversion quietly degrades the audio quality feeding the engine.

**The recommendation.** Check each device's real sample rate at startup, and either convert to 16 kHz cleanly ourselves or stop with a clear message if something is off.

**Why it helps.** Clean, correct-rate audio is the foundation; garbage in means garbage out no matter how good the model is.

**What we decided.** Approved.

**Where it runs.** On your Mac, when opening the audio devices.

---

## 9. Fail loudly instead of silently (audit findings #2 and #3)

**The problem.** Two ways the tool can fail without telling you:
- If you accidentally set the chunk length to 0, a background loop spins forever using 100% CPU and produces nothing.
- If the background transcription worker crashes mid-meeting, the main program keeps running as if nothing happened - the app looks alive, the microphone light is on, but no new text is being written. You could lose most of a meeting and not notice until afterward.

**The recommendation.** Add simple guards: reject invalid settings up front, and have the main program watch the worker and **stop loudly** (clear on-screen error) if it dies.

**Why it helps.** The worst outcome for a transcriber is silently stopping. A loud failure lets you restart and save the meeting.

**What we decided.** Approved.

**Where it runs.** On your Mac, in the main control loop and settings validation.

---

## Priority summary

If implementing incrementally, a sensible order:

1. **#7 (callback fix)** first - it protects everything else from dropped audio.
2. **#3 (GPU server + turbo/beam 5)** and **#2 (hotwords)** together - the biggest accuracy gains, and they depend on each other (beam 5 makes hotwords work).
3. **#4 (disk safety copy)**, **#6 (pause-based chunking)**, **#8 (sample rate)**, **#9 (fail loudly)** - reliability and quality hardening.
4. **#1** is handled by #6 plus the server's anti-looping settings; **#5** is intentionally not built.

Full technical implementation is in `gpu-transcription-plan.md` (same folder). Network and server setup is in `project-local-model-setup.md`. The underlying code audit is in `AUDIT.md`.
