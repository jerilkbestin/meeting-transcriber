"""Configuration objects for teams-notetaker.

Everything the pipeline needs is funnelled through :class:`Settings` so that the
CLI, the capture threads and the summarizer all read from one immutable place
instead of passing a dozen loose arguments around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Whisper always wants 16 kHz mono float32. Every capture device gets resampled
# to this before it reaches the model.
TARGET_SR = 16_000

# Labels used for the two capture tracks. "mic" is you; "system" is whatever the
# OS is playing out of the speakers, i.e. the remote participants.
TRACK_MIC = "mic"
TRACK_SYSTEM = "system"


@dataclass
class Settings:
    # --- output ---------------------------------------------------------
    out_dir: Path
    title: str

    # --- capture --------------------------------------------------------
    capture_mic: bool = True
    capture_system: bool = True
    mic_device_index: int | None = None
    system_device_index: int | None = None
    save_audio: bool = True

    # --- chunking -------------------------------------------------------
    # Minimum audio accumulated before we even consider transcribing.
    min_chunk_seconds: float = 12.0
    # Hard ceiling: cut here even if nobody has stopped talking.
    max_chunk_seconds: float = 30.0
    # A trailing window this long and this quiet is treated as a sentence break.
    silence_window_seconds: float = 0.6
    silence_rms: float = 0.006

    # --- transcription --------------------------------------------------
    model_size: str = "small.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 1
    vad_filter: bool = True
    initial_prompt: str | None = None

    # --- summarization --------------------------------------------------
    # "auto"   -> claude if a key is present, else local if reachable, else none
    # "claude" -> Anthropic API
    # "local"  -> any OpenAI-compatible server (Ollama, LM Studio, llama.cpp)
    # "none"   -> transcript + paste-ready prompt file only
    notes_backend: str = "auto"
    anthropic_model: str = "claude-sonnet-5"
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "llama3.1:8b"
    max_output_tokens: int = 8000
    # Rough character budget per summarization call (~4 chars/token). Cloud
    # models take the whole meeting in one go; local models usually have a much
    # smaller context, so the CLI drops this for the local backend.
    summarize_char_budget: int = 400_000
    local_char_budget: int = 24_000
    skip_notes: bool = False

    # --- misc -----------------------------------------------------------
    speaker_names: dict[str, str] = field(
        default_factory=lambda: {TRACK_MIC: "Me", TRACK_SYSTEM: "Participants"}
    )

    @property
    def session_dir(self) -> Path:
        return self.out_dir

    def api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")
