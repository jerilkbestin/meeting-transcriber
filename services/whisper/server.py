"""Native Windows/CUDA Whisper service; see README.md beside this file."""

from contextlib import asynccontextmanager
import logging
import os
from threading import Lock
from time import perf_counter

import av
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from faster_whisper import WhisperModel

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load one model before readiness; fail startup if CUDA/model setup fails."""
    app.state.config = {
        "model": os.getenv("WHISPER_MODEL", "large-v3-turbo"),
        "device": os.getenv("WHISPER_DEVICE", "cuda"),
        "compute_type": os.getenv("WHISPER_COMPUTE", "int8"),
    }
    config = app.state.config
    app.state.model = WhisperModel(
        config["model"], device=config["device"], compute_type=config["compute_type"],
    )
    app.state.inference_lock = Lock()
    try:
        yield
    finally:
        del app.state.model


app = FastAPI(title="Jarvis local Whisper server", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    """Readiness means model loaded, not a guarantee about inference latency."""
    return {"status": "ok", **request.app.state.config}


@app.post("/v1/audio/transcriptions")
def transcribe(
    request: Request,
    file: UploadFile = File(...),
    initial_prompt: str = Form(default="", max_length=10000),
    language: str = Form(default=""),
    beam_size: int = Form(default=5, ge=1, le=10),
):
    """Decode an uploaded audio stream in a worker thread, then return segments."""
    language = language.strip() or None
    if language is not None and language not in request.app.state.model.supported_languages:
        raise HTTPException(status_code=422, detail="Unsupported Whisper language code")
    if file.size == 0:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    # Avoid an unbounded GPU queue when retries or two capture streams overlap.
    lock = request.app.state.inference_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Whisper is busy; retry shortly",
                            headers={"Retry-After": "1"})
    started = perf_counter()
    try:
        # UploadFile already provides a seekable spooled file; no second copy or
        # filename-derived path is needed. FastAPI owns its cleanup.
        segments, info = request.app.state.model.transcribe(
            file.file, initial_prompt=initial_prompt or None, language=language,
            beam_size=beam_size, vad_filter=True,
        )
        # Inference is lazy: keep the lock until the generator is exhausted.
        segments = list(segments)
        return {
            "text": "".join(seg.text for seg in segments).strip(),
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {"text": seg.text, "start": seg.start, "end": seg.end,
                 "no_speech_prob": seg.no_speech_prob,
                 "avg_logprob": seg.avg_logprob}
                for seg in segments
            ],
        }
    except av.error.FFmpegError as exc:
        raise HTTPException(status_code=400, detail="Cannot decode audio upload") from exc
    except Exception as exc:
        logger.exception("Whisper inference failed")
        raise HTTPException(status_code=500, detail="Whisper inference failed; see server log") from exc
    finally:
        logger.info("Transcription request completed in %.3fs", perf_counter() - started)
        lock.release()
