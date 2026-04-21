from __future__ import annotations

import base64
import logging
import uuid

import httpx
import numpy as np

from voice_input.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VOLCENGINE_ASR_URL = "https://openspeech.bytedance.com/api/v1/auc/submit"


class VolcengineSpeechBackend(TranscriptionBackend):
    """Volcengine speech-to-text API backend."""

    def __init__(
        self,
        app_id: str = "",
        access_key: str = "",
        secret_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.app_id = app_id
        self.access_key = access_key
        self.secret_key = secret_key
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    def is_streaming(self) -> bool:
        return False

    async def initialize(self) -> None:
        if not self.app_id or not self.access_key:
            log.warning("Volcengine credentials not fully configured")
            return
        log.info("Volcengine Speech backend initialized (app_id=%s)", self.app_id)

    def _build_headers(self, audio_bytes: bytes, language: str) -> dict[str, str]:
        """Build request headers with authentication."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer; {self.access_key}",
            "X-Api-App-Key": self.app_id,
        }

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        try:
            audio_bytes = audio_data.tobytes()
            response = await self._client.post(
                VOLCENGINE_ASR_URL,
                json={
                    "app": {
                        "appid": self.app_id,
                        "cluster": "volcengine_input_common",
                    },
                    "user": {"uid": "voice-input-user"},
                    "audio": {
                        "format": "raw",
                        "rate": SAMPLE_RATE,
                        "bits": 16,
                        "channel": 1,
                        "language": language,
                        "data": base64.b64encode(audio_bytes).decode(),
                    },
                    "request": {
                        "reqid": str(uuid.uuid4()),
                        "sequence": -1,
                        "nbest": 1,
                        "text": "",
                    },
                    "additions": {"with_frontend_asr": "true"},
                },
                headers=self._build_headers(audio_bytes, language),
            )
            response.raise_for_status()
            return response.json().get("result", {}).get("text", "").strip()
        except Exception as e:
            log.error("Volcengine Speech API error: %s", e)
            return ""

    async def cleanup(self) -> None:
        await self._client.aclose()
