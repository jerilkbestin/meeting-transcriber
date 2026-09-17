"""HTTP contract and lifecycle tests; no model downloads or GPU required."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
from fastapi.testclient import TestClient

import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(supported_languages=["en", "fr"])
        self.segment = SimpleNamespace(text=" Hello.", start=0.0, end=1.0,
                                       no_speech_prob=0.02, avg_logprob=-0.3)
        self.model.transcribe.return_value = (
            iter([self.segment]), SimpleNamespace(language="en", duration=1.0),
        )
        self.factory = self.enterContext(patch.object(server, "WhisperModel", return_value=self.model))
        self.client = self.enterContext(TestClient(server.app))

    def post(self, **data):
        return self.client.post("/v1/audio/transcriptions",
                                files={"file": ("chunk.wav", b"test audio", "audio/wav")}, data=data)

    def test_readiness_and_single_model_load(self):
        for _ in range(2):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
        self.factory.assert_called_once()

    def test_multipart_contract_and_upload_cleanup(self):
        response = self.post(language="en", initial_prompt="Jarvis", beam_size="3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "text": "Hello.", "language": "en", "duration": 1.0,
            "segments": [vars(self.segment)],
        })
        args, kwargs = self.model.transcribe.call_args
        self.assertEqual(kwargs, dict(language="en", initial_prompt="Jarvis", beam_size=3, vad_filter=True))
        self.assertTrue(args[0].closed)

    def test_optional_fields_and_silence(self):
        self.model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=1.0))
        self.assertEqual(self.post().json()["segments"], [])
        self.assertIsNone(self.model.transcribe.call_args.kwargs["language"])
        self.assertIsNone(self.model.transcribe.call_args.kwargs["initial_prompt"])

    def test_validation_before_inference(self):
        self.assertEqual(self.post(language="not-a-language").status_code, 422)
        self.assertEqual(self.post(beam_size="0").status_code, 422)
        self.assertEqual(self.client.post("/v1/audio/transcriptions").status_code, 422)
        self.assertEqual(self.client.post("/v1/audio/transcriptions",
                         files={"file": ("empty.wav", b"")}).status_code, 400)
        self.model.transcribe.assert_not_called()

    def test_bad_audio_releases_lock(self):
        self.model.transcribe.side_effect = av.error.InvalidDataError(1, "bad audio")
        self.assertEqual(self.post().status_code, 400)
        self.model.transcribe.side_effect = None
        self.assertEqual(self.post().status_code, 200)

    def test_lazy_inference_failure_releases_lock(self):
        def failed_segments():
            yield self.segment
            raise RuntimeError("private CUDA diagnostic")

        self.model.transcribe.return_value = (failed_segments(), SimpleNamespace(language="en", duration=1))
        with self.assertLogs("uvicorn.error", level="ERROR"):
            response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private CUDA diagnostic", response.text)
        self.model.transcribe.return_value = (iter([]), SimpleNamespace(language="en", duration=1))
        self.assertEqual(self.post().status_code, 200)

    def test_health_and_busy_response_during_inference(self):
        entered, release = Event(), Event()

        def slow_segments():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            yield self.segment

        self.model.transcribe.return_value = (slow_segments(), SimpleNamespace(language="en", duration=1))
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.post)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.client.get("/health").status_code, 200)
                second = self.post()
                self.assertEqual(second.status_code, 503)
                self.assertEqual(second.headers["Retry-After"], "1")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)

    def test_startup_failure_is_not_healthy(self):
        with patch.object(server, "WhisperModel", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                with TestClient(server.app):
                    pass


if __name__ == "__main__":
    unittest.main()
