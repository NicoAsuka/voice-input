from __future__ import annotations

import io
import logging
import struct
import wave

import httpx
import numpy as np

from voice_input.backends.base import TranscriptionBackend

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
