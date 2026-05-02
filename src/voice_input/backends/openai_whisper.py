from __future__ import annotations

import io
import logging
import struct
import wave

import httpx
import numpy as np

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class OpenAIWhisperBackend(TranscriptionBackend):
    """OpenAI Whisper API backend (compatible with any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "whisper-1",
        timeout: float = 30.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            trust_env=False,
        )

    def is_streaming(self) -> bool:
        return False

    async def initialize(self) -> None:
        log.info("OpenAI Whisper backend initialized (base=%s, model=%s)", self.api_base, self.model)

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        """Encode int16 numpy array as WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        try:
            wav_bytes = self._encode_wav(audio_data)
            response = await self._client.post(
                "/audio/transcriptions",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": self.model, "language": language},
            )
            response.raise_for_status()
            return response.json().get("text", "").strip()
        except Exception as e:
            log.error("OpenAI Whisper API error: %s", e)
            return ""

    async def cleanup(self) -> None:
        await self._client.aclose()

    def is_ready(self) -> bool:
        return bool(self._client)

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="openai-whisper",
            model_id=self.model,
            capabilities=BackendCapabilities(
                supports_streaming=False,
                requires_network=True,
            ),
        )

    def create_session(self, language: str) -> Session:
        return _OpenAISession(self, language)

    async def shutdown(self) -> None:
        await self._client.aclose()


class _OpenAISession(Session):
    def __init__(self, backend: OpenAIWhisperBackend, language: str) -> None:
        self._backend = backend
        self._language = language
        self._buffer: list[np.ndarray] = []
        self._cancelled = False

    def push_audio(self, pcm_int16: np.ndarray) -> None:
        if not self._cancelled:
            self._buffer.append(pcm_int16)

    def cancel(self) -> None:
        self._cancelled = True
        self._buffer.clear()

    async def finish(self) -> str:
        if self._cancelled or not self._buffer:
            return ""
        audio = np.concatenate(self._buffer)
        try:
            return await self._backend.transcribe(audio, self._language)
        except Exception as e:
            raise RecognitionError(
                str(e), user_message=f"OpenAI Whisper error: {str(e)[:80]}"
            ) from e
