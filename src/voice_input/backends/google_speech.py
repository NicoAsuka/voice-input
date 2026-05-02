from __future__ import annotations

import base64
import logging
import os

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
GOOGLE_SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"

_LANGUAGE_MAP = {
    "zh": "zh-CN",
    "zh-TW": "zh-TW",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
}


class GoogleSpeechBackend(TranscriptionBackend):
    """Google Cloud Speech-to-Text REST API backend."""

    def __init__(
        self,
        credentials_path: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", ""
        )
        self._access_token = ""
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    def is_streaming(self) -> bool:
        return False

    async def initialize(self) -> None:
        """Load Google service account credentials and refresh an access token."""
        if not self.credentials_path:
            log.warning("Google credentials path not set")
            return

        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            self._access_token = credentials.token or ""
            log.info("Google Speech backend initialized")
        except Exception as e:
            log.error("Failed to load Google credentials: %s", e)

    def _map_language(self, language: str) -> str:
        """Map app language codes to the BCP-47 codes expected by Google."""
        if language in _LANGUAGE_MAP:
            return _LANGUAGE_MAP[language]
        return f"{language}-{language.upper()}"

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        try:
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
            response = await self._client.post(
                GOOGLE_SPEECH_URL,
                json={
                    "config": {
                        "encoding": "LINEAR16",
                        "sampleRateHertz": SAMPLE_RATE,
                        "languageCode": self._map_language(language),
                    },
                    "audio": {"content": audio_b64},
                },
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                return ""
            return "".join(
                result["alternatives"][0]["transcript"] for result in results
            ).strip()
        except Exception as e:
            log.error("Google Speech API error: %s", e)
            return ""

    async def cleanup(self) -> None:
        await self._client.aclose()

    def is_ready(self) -> bool:
        return bool(self.credentials_path)

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="google",
            model_id="google-stt-default",
            capabilities=BackendCapabilities(
                supports_streaming=False,
                requires_network=True,
            ),
        )

    def create_session(self, language: str) -> Session:
        return _GoogleSession(self, language)

    async def shutdown(self) -> None:
        await self._client.aclose()


class _GoogleSession(Session):
    def __init__(self, backend: GoogleSpeechBackend, language: str) -> None:
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
                str(e), user_message=f"Google STT error: {str(e)[:80]}"
            ) from e
