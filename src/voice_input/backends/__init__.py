"""STT backend abstraction and factory."""
from __future__ import annotations

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import AppConfig


def create_backend(config: AppConfig) -> TranscriptionBackend:
    """Create a transcription backend from app config."""
    stt_cfg = config.get("stt", {})
    backend_name = stt_cfg.get("backend", "sherpa")

    # Legacy 'local' alias maps to sherpa
    if backend_name in ("sherpa", "local"):
        from voice_input.backends.sherpa_backend import SherpaBackend
        return SherpaBackend(config)

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
            resource_id=volc_cfg.get("resource_id", "volc.seedasr.sauc.duration"),
        )

    raise ValueError(f"Unknown STT backend: {backend_name}")
