"""STT backend abstraction and factory."""
from __future__ import annotations

import logging

from voice_input.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)


def create_backend(config: dict) -> TranscriptionBackend:
    """Create a transcription backend from app config."""
    stt_cfg = config.get("stt", {})
    backend_name = stt_cfg.get("backend", "local")

    if backend_name == "local":
        from voice_input.backends.local_whisper import LocalWhisperBackend

        whisper_cfg = config.get("whisper", {})
        return LocalWhisperBackend(
            model_name=whisper_cfg.get("model", "medium"),
            device=whisper_cfg.get("device", "auto"),
        )

    if backend_name == "openai":
        from voice_input.backends.openai_whisper import OpenAIWhisperBackend

        openai_cfg = stt_cfg.get("openai", {})
        try:
            import keyring

            api_key = keyring.get_password("voice-input", "stt-openai-api-key") or ""
        except Exception:
            api_key = ""
        return OpenAIWhisperBackend(
            api_base=openai_cfg.get("api_base", "https://api.openai.com/v1"),
            api_key=api_key,
            model=openai_cfg.get("model", "whisper-1"),
        )

    if backend_name == "google":
        from voice_input.backends.google_speech import GoogleSpeechBackend

        google_cfg = stt_cfg.get("google", {})
        return GoogleSpeechBackend(
            credentials_path=google_cfg.get("credentials_path", ""),
        )

    if backend_name == "volcengine":
        from voice_input.backends.volcengine_speech import VolcengineSpeechBackend

        volc_cfg = stt_cfg.get("volcengine", {})
        try:
            import keyring

            access_key = (
                keyring.get_password("voice-input", "stt-volcengine-access-key") or ""
            )
            secret_key = (
                keyring.get_password("voice-input", "stt-volcengine-secret-key") or ""
            )
        except Exception:
            access_key = ""
            secret_key = ""
        return VolcengineSpeechBackend(
            app_id=volc_cfg.get("app_id", ""),
            access_key=access_key,
            secret_key=secret_key,
        )

    raise ValueError(f"Unknown STT backend: {backend_name}")
