"""Turn a transcript into structured meeting notes.

Three ways to get notes, in descending order of setup cost:

1. ``claude``  -- the Anthropic API. Best quality, needs ANTHROPIC_API_KEY.
2. ``local``   -- any OpenAI-compatible server on your own machine (Ollama,
                  LM Studio, llama.cpp's server, vLLM). No key, no cost, no data
                  leaving the box.
3. ``none``    -- write ``notes-prompt.md``: the prompt and the transcript
                  concatenated, ready to paste into claude.ai by hand.

Option 3 always happens regardless, so there is never a run that leaves you with
nothing to paste.

The transcript is raw ASR output: no punctuation guarantees, occasional
misheard proper nouns, and no notion of what mattered. The model's job is to
impose structure -- decisions, owners, deadlines, open questions -- and to say
so explicitly when the transcript does not support a claim, rather than
inventing a plausible action item.
"""

from __future__ import annotations

import json
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous meeting-notes writer working from an automatic speech
    recognition transcript of a Microsoft Teams call.

    Ground rules, in priority order:

    1. Never invent content. Every decision, action item, name, number and date
       must be traceable to something actually said in the transcript. If the
       transcript is ambiguous, say so rather than resolving it silently.
    2. The transcript comes from speech recognition and WILL contain errors,
       especially in proper nouns, product names and acronyms. When a term looks
       garbled but you can infer it from context, write your best reading and
       append "(as heard: <literal text>)" the first time.
    3. Speaker labels are coarse. "Me" is the person running the recorder (their
       microphone). "Participants" is everyone else combined (the speaker
       output), so individual remote speakers are NOT separated. Only attribute
       a statement to a named person when someone is addressed or introduces
       themselves by name in the transcript.
    4. An action item needs an owner and, ideally, a due date. If either is
       missing from the transcript, write "owner: unassigned" or
       "due: not stated" -- do not guess.
    5. Be concise. Notes are read by people who were in the meeting.

    Output GitHub-flavoured Markdown with exactly these sections, in this order,
    and omit none of them (write "None captured." if a section is empty):

    ## TL;DR
    Three to five bullets. What a colleague who missed the meeting needs.

    ## Decisions
    What was actually decided. One bullet each, stated as a settled outcome.

    ## Action items
    A Markdown table with columns: Action | Owner | Due | Source timestamp.

    ## Discussion notes
    Grouped by topic with `###` sub-headings. This is the substantive body.

    ## Open questions & risks
    Things left unresolved, disagreements, flagged risks or blockers.

    ## Follow-up needed from the transcript
    Points where the transcript was unclear, cut off, or too garbled to trust,
    with timestamps, so the reader knows where to check the recording.
    """
).strip()

REDUCE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are consolidating several partial note sets, each covering a
    consecutive slice of one long meeting, into a single coherent set of notes.

    Merge duplicates, keep chronological sense, preserve every distinct decision
    and action item, and do not introduce anything that is not present in the
    partial notes. Keep the exact same section structure as the inputs:

    ## TL;DR
    ## Decisions
    ## Action items
    ## Discussion notes
    ## Open questions & risks
    ## Follow-up needed from the transcript
    """
).strip()


class SummarizationError(RuntimeError):
    pass


# A backend is just "(system, user) -> text". Keeping it to one callable means
# the map-reduce logic below does not care which model is behind it.
CallFn = Callable[[str, str], str]


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def claude_backend(api_key: str | None, model: str, max_tokens: int = 8000) -> CallFn:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise SummarizationError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from exc
    if not api_key:
        raise SummarizationError(
            "No API key found. Set ANTHROPIC_API_KEY, or use --notes local with a "
            "local model, or --skip-notes to stop after the transcript."
        )
    client = anthropic.Anthropic(api_key=api_key)

    def call(system: str, user: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        out = "\n".join(parts).strip()
        if not out:
            raise SummarizationError("The model returned an empty response.")
        return out

    return call


def local_backend(
    base_url: str,
    model: str,
    max_tokens: int = 4000,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> CallFn:
    """Talk to any OpenAI-compatible /v1/chat/completions endpoint.

    Ollama, LM Studio, llama.cpp's server and vLLM all speak this, so one
    implementation covers every local runtime worth using. Built on urllib so
    the tool gains no extra dependency for this path.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"

    def call(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # server reachable, request rejected
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise SummarizationError(
                f"Local model server returned HTTP {exc.code} from {url}.\n{detail}"
            ) from exc
        except urllib.error.URLError as exc:  # nothing listening
            raise SummarizationError(
                f"Could not reach a local model server at {url} ({exc.reason}).\n"
                "Start one first, e.g.:  ollama serve   and   ollama pull llama3.1:8b"
            ) from exc
        except TimeoutError as exc:
            raise SummarizationError(
                f"Local model timed out after {timeout:.0f}s. Try a smaller model "
                "or a smaller --notes-chunk-chars."
            ) from exc

        try:
            out = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise SummarizationError(
                f"Unexpected response shape from {url}: {json.dumps(body)[:400]}"
            ) from exc
        if not out:
            raise SummarizationError("The local model returned an empty response.")
        return out

    return call


# --------------------------------------------------------------------------
# prompt assembly + map-reduce
# --------------------------------------------------------------------------

def build_user_prompt(
    transcript: str, title: str, context: str | None = None, part: tuple[int, int] | None = None
) -> str:
    header = f"Meeting title: {title}\n"
    if context:
        header += f"Additional context supplied by the organiser: {context}\n"
    if part:
        i, n = part
        header += (
            f"\nThis is part {i} of {n} of the transcript. Write notes for THIS PART ONLY.\n"
        )
    return f"{header}\nTranscript:\n\n{transcript}"


def build_paste_prompt(transcript: str, title: str, context: str | None = None) -> str:
    """The whole thing as one block, ready to paste into a chat UI."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"{build_user_prompt(transcript, title, context)}\n"
    )


def _split_transcript(text: str, budget: int) -> list[str]:
    """Split on line boundaries into pieces under ``budget`` characters.

    Splitting on lines (never mid-utterance) keeps each piece independently
    readable, which is what makes the map step produce sane partial notes.
    """
    if len(text) <= budget:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > budget and current:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        pieces.append("".join(current))
    return pieces


def summarize_transcript(
    transcript: str,
    *,
    title: str,
    call: CallFn,
    char_budget: int = 400_000,
    context: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Produce notes. Uses map-reduce only when the transcript exceeds the budget.

    ``char_budget`` is what makes local models viable: an 8B model with a 8k
    context cannot swallow an hour-long meeting, so we summarise slices and then
    merge the slice notes.
    """
    say = progress or (lambda _m: None)
    pieces = _split_transcript(transcript, char_budget)

    if len(pieces) == 1:
        return call(SYSTEM_PROMPT, build_user_prompt(transcript, title, context))

    partials: list[str] = []
    for i, piece in enumerate(pieces, 1):
        say(f"  summarising part {i}/{len(pieces)}...")
        partials.append(
            call(SYSTEM_PROMPT, build_user_prompt(piece, title, context, part=(i, len(pieces))))
        )

    say(f"  merging {len(partials)} partial note sets...")
    joined = "\n\n---\n\n".join(
        f"### Partial notes {i}\n\n{p}" for i, p in enumerate(partials, 1)
    )
    header = f"Meeting title: {title}\n"
    if context:
        header += f"Additional context supplied by the organiser: {context}\n"
    return call(REDUCE_SYSTEM_PROMPT, f"{header}\nPartial note sets to merge:\n\n{joined}")


# --------------------------------------------------------------------------
# input adapters
# --------------------------------------------------------------------------

def read_transcript_source(path: Path) -> str:
    """Read a transcript from .vtt, .srt, .jsonl, .md or .txt.

    Lets you run the notes step against a transcript Teams exported itself,
    which is the escape hatch when live capture is not an option.
    """
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".vtt", ".srt"}:
        return _parse_cue_format(raw)
    if suffix == ".jsonl":
        from .engine import utterances_from_jsonl, render_plain_transcript

        return render_plain_transcript(utterances_from_jsonl(path))
    return raw


def _parse_cue_format(raw: str) -> str:
    """Flatten WebVTT/SRT cues into '[hh:mm:ss] Speaker: text' lines.

    Teams' exported .vtt embeds the speaker as '<v Name>text</v>', which is the
    one place we get real per-person attribution for free.
    """
    import re

    lines_out: list[str] = []
    timestamp = "00:00:00"
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2})[.,]\d{3}\s*-->")
    voice_re = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>|$)", re.S)
    tag_re = re.compile(r"<[^>]+>")

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"WEBVTT"} or stripped.isdigit():
            continue
        m = time_re.search(stripped)
        if m:
            timestamp = m.group(1)
            continue
        if stripped.startswith("NOTE") or stripped.startswith("STYLE"):
            continue
        vm = voice_re.search(stripped)
        if vm:
            speaker = vm.group(1).strip()
            text = tag_re.sub("", vm.group(2)).strip()
        else:
            speaker = "Speaker"
            text = tag_re.sub("", stripped).strip()
        if text:
            lines_out.append(f"[{timestamp}] {speaker}: {text}")
    return "\n".join(lines_out)
