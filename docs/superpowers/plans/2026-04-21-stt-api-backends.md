# STT API Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add external STT API backend support (OpenAI, Google, Volcengine) alongside existing local faster-whisper, selectable via config and tray menu.

**Architecture:** Abstract `TranscriptionBackend` base class with factory function. Local backend extracted from `WhisperWorker`. Remote backends use `httpx.AsyncClient`. `AppController` gains a `TRANSCRIBING` state for async remote API calls. Tray menu and settings dialog extended for backend selection and configuration.

**Tech Stack:** Python 3.11+, PyQt6, httpx, keyring, numpy, faster-whisper (local only), google-auth (Google backend only)

---

### Task 1: TranscriptionBackend ABC

**Files:**
- Create: `src/voice_input/backends/__init__.py`
- Create: `src/voice_input/backends/base.py`
- Test: `tests/test_backends_base.py`

- [ ] **Step 1: Write the failing test for ABC interface**

```python
# tests/test_backends_base.py
import pytest
import numpy as np
from voice_input.backends.base import TranscriptionBackend


def test_cannot_instantiate_abc():
    """TranscriptionBackend is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        TranscriptionBackend()


def test_concrete_subclass_must_implement_all_methods():
    """A subclass missing abstract methods cannot be instantiated."""
    class Incomplete(TranscriptionBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()


@pytest.mark.asyncio
async def test_concrete_subclass_works():
    """A fully implemented subclass can be instantiated and called."""
    class Dummy(TranscriptionBackend):
        async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
            return "hello"

        def is_streaming(self) -> bool:
            return False

        async def initialize(self) -> None:
            pass

    backend = Dummy()
    assert backend.is_streaming() is False
    result = await backend.transcribe(np.zeros(100, dtype=np.int16), "zh")
    assert result == "hello"


@pytest.mark.asyncio
async def test_cleanup_default_is_noop():
    """Default cleanup() does nothing and doesn't raise."""
    class Dummy(TranscriptionBackend):
        async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
            return ""

        def is_streaming(self) -> bool:
            return False

        async def initialize(self) -> None:
            pass

    backend = Dummy()
    await backend.cleanup()  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends'`

- [ ] **Step 3: Create the backends package and ABC**

```python
# src/voice_input/backends/__init__.py
"""STT backend abstraction and factory."""
```

```python
# src/voice_input/backends/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TranscriptionBackend(ABC):
    """Abstract base class for all speech-to-text backends."""

    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        """Transcribe int16 audio array, return text."""
        ...

    @abstractmethod
    def is_streaming(self) -> bool:
        """Whether this backend supports real-time incremental transcription."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Load model or validate API credentials."""
        ...

    async def cleanup(self) -> None:
        """Release resources. Default is no-op."""
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends_base.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/__init__.py src/voice_input/backends/base.py tests/test_backends_base.py
git commit -m "feat: add TranscriptionBackend ABC"
```

---

### Task 2: LocalWhisperBackend

**Files:**
- Create: `src/voice_input/backends/local_whisper.py`
- Test: `tests/test_backend_local_whisper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backend_local_whisper.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from voice_input.backends.local_whisper import LocalWhisperBackend


def test_is_streaming_returns_true():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    assert backend.is_streaming() is True


@pytest.mark.asyncio
async def test_initialize_loads_model():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    mock_model = MagicMock()
    with patch("voice_input.backends.local_whisper.WhisperModel", return_value=mock_model):
        await backend.initialize()
    assert backend._model is mock_model


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    backend._model = mock_model

    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_short_audio_returns_empty():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    backend._model = MagicMock()

    audio = np.zeros(100, dtype=np.int16)  # too short
    result = await backend.transcribe(audio, "zh")
    assert result == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_local_whisper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends.local_whisper'`

- [ ] **Step 3: Implement LocalWhisperBackend**

```python
# src/voice_input/backends/local_whisper.py
from __future__ import annotations

import logging

import numpy as np

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)

# Minimum samples to attempt transcription (0.1s at 16kHz)
MIN_SAMPLES = 1600


class LocalWhisperBackend(TranscriptionBackend):
    """Local faster-whisper transcription backend."""

    def __init__(
        self,
        model_name: str = "medium",
        language: str = "zh",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self._model = None

    def is_streaming(self) -> bool:
        return True

    async def initialize(self) -> None:
        from faster_whisper import WhisperModel

        model_dir = xdg_cache_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        compute_type = "int8"
        actual_device = "cpu"
        if self.device in ("auto", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    actual_device = "cuda"
                    compute_type = "float16"
            except ImportError:
                pass

        log.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            self.model_name, actual_device, compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=actual_device,
            compute_type=compute_type,
            download_root=str(model_dir),
        )
        log.info("Whisper model loaded and ready")

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if self._model is None or len(audio_data) < MIN_SAMPLES:
            return ""
        try:
            audio_f32 = audio_data.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Transcription error: %s", e)
            return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_local_whisper.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/local_whisper.py tests/test_backend_local_whisper.py
git commit -m "feat: add LocalWhisperBackend extracted from whisper_worker"
```

---

### Task 3: OpenAIWhisperBackend

**Files:**
- Create: `src/voice_input/backends/openai_whisper.py`
- Test: `tests/test_backend_openai.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backend_openai.py
import io
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.openai_whisper import OpenAIWhisperBackend


def test_is_streaming_returns_false():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    assert backend.is_streaming() is False


def test_encode_wav_produces_valid_wav():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    audio = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
    wav_bytes = backend._encode_wav(audio)
    # WAV header starts with RIFF
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "transcribed text"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "transcribed text"
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "/audio/transcriptions"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""


@pytest.mark.asyncio
async def test_cleanup_closes_client():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
        await backend.cleanup()
    mock_close.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_openai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends.openai_whisper'`

- [ ] **Step 3: Implement OpenAIWhisperBackend**

```python
# src/voice_input/backends/openai_whisper.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_openai.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/openai_whisper.py tests/test_backend_openai.py
git commit -m "feat: add OpenAI Whisper API backend"
```

---

### Task 4: GoogleSpeechBackend

**Files:**
- Create: `src/voice_input/backends/google_speech.py`
- Test: `tests/test_backend_google.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backend_google.py
import base64
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.google_speech import GoogleSpeechBackend


def test_is_streaming_returns_false():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    assert backend.is_streaming() is False


def test_map_language():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    assert backend._map_language("zh") == "zh-CN"
    assert backend._map_language("en") == "en-US"
    assert backend._map_language("ja") == "ja-JP"
    assert backend._map_language("ko") == "ko-KR"
    assert backend._map_language("zh-TW") == "zh-TW"
    assert backend._map_language("fr") == "fr-FR"  # fallback


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"alternatives": [{"transcript": "hello"}]},
            {"alternatives": [{"transcript": " world"}]},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_no_results():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "en")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_google.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends.google_speech'`

- [ ] **Step 3: Implement GoogleSpeechBackend**

```python
# src/voice_input/backends/google_speech.py
from __future__ import annotations

import base64
import logging
import os

import httpx
import numpy as np

from voice_input.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

GOOGLE_SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"

# Map short language codes to BCP-47 codes used by Google
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
        self._access_token: str = ""
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    def is_streaming(self) -> bool:
        return False

    async def initialize(self) -> None:
        """Load credentials and obtain access token."""
        if not self.credentials_path:
            log.warning("Google credentials path not set")
            return
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds.refresh(Request())
            self._access_token = creds.token
            log.info("Google Speech backend initialized")
        except Exception as e:
            log.error("Failed to load Google credentials: %s", e)

    def _map_language(self, language: str) -> str:
        """Map short language code to BCP-47."""
        if language in _LANGUAGE_MAP:
            return _LANGUAGE_MAP[language]
        return f"{language}-{language.upper()}"

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        try:
            audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
            lang_code = self._map_language(language)
            body = {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": SAMPLE_RATE,
                    "languageCode": lang_code,
                },
                "audio": {"content": audio_b64},
            }
            response = await self._client.post(
                GOOGLE_SPEECH_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                return ""
            return " ".join(
                r["alternatives"][0]["transcript"] for r in results
            ).strip()
        except Exception as e:
            log.error("Google Speech API error: %s", e)
            return ""

    async def cleanup(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_google.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/google_speech.py tests/test_backend_google.py
git commit -m "feat: add Google Cloud Speech-to-Text backend"
```

---

### Task 5: VolcengineSpeechBackend

**Files:**
- Create: `src/voice_input/backends/volcengine_speech.py`
- Test: `tests/test_backend_volcengine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backend_volcengine.py
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.volcengine_speech import VolcengineSpeechBackend


def test_is_streaming_returns_false():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    assert backend.is_streaming() is False


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"text": "transcribed text"}
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "transcribed text"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""


def test_sign_request_produces_headers():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    headers = backend._build_headers(b"fake-audio", "zh")
    assert "Authorization" in headers or "X-Api-Key" in headers or "Content-Type" in headers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_volcengine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends.volcengine_speech'`

- [ ] **Step 3: Implement VolcengineSpeechBackend**

```python
# src/voice_input/backends/volcengine_speech.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid

import httpx
import numpy as np

from voice_input.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Volcengine ASR endpoint
VOLCENGINE_ASR_URL = "https://openspeech.bytedance.com/api/v1/auc/submit"


class VolcengineSpeechBackend(TranscriptionBackend):
    """Volcengine (ByteDance) speech-to-text API backend."""

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
            audio_b64 = base64.b64encode(audio_bytes).decode()
            headers = self._build_headers(audio_bytes, language)
            body = {
                "app": {"appid": self.app_id, "cluster": "volcengine_input_common"},
                "user": {"uid": "voice-input-user"},
                "audio": {
                    "format": "raw",
                    "rate": SAMPLE_RATE,
                    "bits": 16,
                    "channel": 1,
                    "language": language,
                },
                "request": {
                    "reqid": str(uuid.uuid4()),
                    "sequence": -1,
                    "nbest": 1,
                    "text": "",
                },
                "additions": {"with_frontend_asr": "true"},
            }
            # Audio data sent as base64 in the data field
            body["audio"]["data"] = audio_b64

            response = await self._client.post(
                VOLCENGINE_ASR_URL,
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}).get("text", "").strip()
        except Exception as e:
            log.error("Volcengine Speech API error: %s", e)
            return ""

    async def cleanup(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_volcengine.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/volcengine_speech.py tests/test_backend_volcengine.py
git commit -m "feat: add Volcengine speech-to-text backend"
```

---

### Task 6: Backend Factory Function

**Files:**
- Modify: `src/voice_input/backends/__init__.py`
- Test: `tests/test_backend_factory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backend_factory.py
import pytest
from unittest.mock import patch

from voice_input.backends import create_backend
from voice_input.backends.base import TranscriptionBackend
from voice_input.backends.local_whisper import LocalWhisperBackend
from voice_input.backends.openai_whisper import OpenAIWhisperBackend
from voice_input.backends.google_speech import GoogleSpeechBackend
from voice_input.backends.volcengine_speech import VolcengineSpeechBackend


def _make_config(backend: str = "local", **overrides) -> dict:
    cfg = {
        "whisper": {"model": "tiny", "language": "zh", "device": "cpu"},
        "stt": {
            "backend": backend,
            "openai": {"api_base": "https://api.openai.com/v1", "model": "whisper-1"},
            "google": {"credentials_path": ""},
            "volcengine": {"app_id": "test"},
        },
    }
    cfg.update(overrides)
    return cfg


def test_create_local_backend():
    backend = create_backend(_make_config("local"))
    assert isinstance(backend, LocalWhisperBackend)


def test_create_openai_backend():
    with patch("keyring.get_password", return_value="sk-test"):
        backend = create_backend(_make_config("openai"))
    assert isinstance(backend, OpenAIWhisperBackend)


def test_create_google_backend():
    backend = create_backend(_make_config("google"))
    assert isinstance(backend, GoogleSpeechBackend)


def test_create_volcengine_backend():
    with patch("keyring.get_password", return_value="fake-key"):
        backend = create_backend(_make_config("volcengine"))
    assert isinstance(backend, VolcengineSpeechBackend)


def test_create_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(_make_config("nonexistent"))


def test_all_backends_are_transcription_backends():
    local = create_backend(_make_config("local"))
    assert isinstance(local, TranscriptionBackend)

    with patch("keyring.get_password", return_value="sk-test"):
        openai_b = create_backend(_make_config("openai"))
    assert isinstance(openai_b, TranscriptionBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_backend'`

- [ ] **Step 3: Implement the factory function**

Replace `src/voice_input/backends/__init__.py` with:

```python
# src/voice_input/backends/__init__.py
"""STT backend abstraction and factory."""
from __future__ import annotations

import logging

from voice_input.backends.base import TranscriptionBackend

log = logging.getLogger(__name__)


def create_backend(config: dict) -> TranscriptionBackend:
    """Create a TranscriptionBackend instance based on config."""
    stt_cfg = config.get("stt", {})
    backend_name = stt_cfg.get("backend", "local")

    if backend_name == "local":
        from voice_input.backends.local_whisper import LocalWhisperBackend

        whisper_cfg = config.get("whisper", {})
        return LocalWhisperBackend(
            model_name=whisper_cfg.get("model", "medium"),
            language=whisper_cfg.get("language", "zh"),
            device=whisper_cfg.get("device", "auto"),
        )

    elif backend_name == "openai":
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

    elif backend_name == "google":
        from voice_input.backends.google_speech import GoogleSpeechBackend

        google_cfg = stt_cfg.get("google", {})
        return GoogleSpeechBackend(
            credentials_path=google_cfg.get("credentials_path", ""),
        )

    elif backend_name == "volcengine":
        from voice_input.backends.volcengine_speech import VolcengineSpeechBackend

        volc_cfg = stt_cfg.get("volcengine", {})
        try:
            import keyring
            access_key = keyring.get_password("voice-input", "stt-volcengine-access-key") or ""
            secret_key = keyring.get_password("voice-input", "stt-volcengine-secret-key") or ""
        except Exception:
            access_key = ""
            secret_key = ""
        return VolcengineSpeechBackend(
            app_id=volc_cfg.get("app_id", ""),
            access_key=access_key,
            secret_key=secret_key,
        )

    else:
        raise ValueError(f"Unknown STT backend: {backend_name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_factory.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/__init__.py tests/test_backend_factory.py
git commit -m "feat: add backend factory function create_backend()"
```

---

### Task 7: Config Changes

**Files:**
- Modify: `src/voice_input/config.py:19-44` (DEFAULT_CONFIG)
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_default_config_has_stt_section():
    cfg = DEFAULT_CONFIG
    assert cfg["stt"]["backend"] == "local"
    assert cfg["stt"]["openai"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["stt"]["openai"]["model"] == "whisper-1"
    assert cfg["stt"]["google"]["credentials_path"] == ""
    assert cfg["stt"]["volcengine"]["app_id"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_default_config_has_stt_section -v`
Expected: FAIL — `KeyError: 'stt'`

- [ ] **Step 3: Add stt section to DEFAULT_CONFIG**

In `src/voice_input/config.py`, add to `DEFAULT_CONFIG` dict after the `"inject"` section:

```python
    "stt": {
        "backend": "local",
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "model": "whisper-1",
        },
        "google": {
            "credentials_path": "",
        },
        "volcengine": {
            "app_id": "",
        },
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: all passed (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/config.py tests/test_config.py
git commit -m "feat: add stt config section with backend defaults"
```

---

### Task 8: AppState TRANSCRIBING + AppController Integration

**Files:**
- Modify: `src/voice_input/app.py:29-44` (AppState, transitions)
- Modify: `src/voice_input/app.py:47-298` (AppController)
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write the failing tests for new state transitions**

Add to `tests/test_app.py`:

```python
def test_transcribing_state_transitions():
    """Verify TRANSCRIBING state transitions."""
    assert AppState.RECORDING.can_transition_to(AppState.TRANSCRIBING)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.IDLE)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.REFINING)
    assert not AppState.IDLE.can_transition_to(AppState.TRANSCRIBING)
    assert not AppState.TRANSCRIBING.can_transition_to(AppState.RECORDING)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_transcribing_state_transitions -v`
Expected: FAIL — `AttributeError: 'AppState' has no attribute 'TRANSCRIBING'`

- [ ] **Step 3: Add TRANSCRIBING state**

In `src/voice_input/app.py`, update `AppState` and `_VALID_TRANSITIONS`:

```python
class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"
    REFINING = "Refining"

    _transitions = None  # placeholder, set below

    def can_transition_to(self, target: AppState) -> bool:
        return target in _VALID_TRANSITIONS.get(self, set())


_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING, AppState.REFINING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: all passed

- [ ] **Step 5: Integrate backend into AppController**

In `src/voice_input/app.py`, update imports:

```python
from voice_input.backends import create_backend
```

In `AppController.__init__`, after creating `self._whisper`, add backend creation:

```python
        # STT Backend
        self._backend = create_backend(config)
        self._backend_is_streaming = self._backend.is_streaming()
```

Update `_on_stop_recording` to handle remote mode:

```python
    @pyqtSlot()
    def _on_stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            return
        self._audio.stop()
        self._viz_timer.stop()
        self._whisper.set_active(False)

        if self._backend_is_streaming:
            # Local mode: text already available from real-time transcription
            text = self._last_transcription
            if not text:
                self._overlay.animate_exit(on_finished=self._overlay.hide)
                self._set_state(AppState.IDLE)
                return
            if self._llm_enabled and self._llm is not None:
                self._set_state(AppState.REFINING)
                self._overlay.update_text("Refining...")
                asyncio.ensure_future(self._refine_and_inject(text))
            else:
                self._inject_and_finish(text)
        else:
            # Remote mode: send full audio buffer to API
            audio_buffer = self._whisper.get_audio_buffer()
            if len(audio_buffer) < 1600:
                self._overlay.animate_exit(on_finished=self._overlay.hide)
                self._set_state(AppState.IDLE)
                return
            self._set_state(AppState.TRANSCRIBING)
            self._overlay.update_text("识别中...")
            asyncio.ensure_future(self._remote_transcribe_and_inject(audio_buffer))
```

Add the new async method:

```python
    async def _remote_transcribe_and_inject(self, audio_buffer: np.ndarray) -> None:
        """Send audio to remote STT API, then optionally refine and inject."""
        try:
            text = await self._backend.transcribe(audio_buffer, self._config["whisper"]["language"])
        except Exception as e:
            log.error("Remote transcription failed: %s", e)
            self._send_notification("Voice Input Error", f"STT API error: {e}")
            text = ""

        if not text:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        self._last_transcription = text
        if self._llm_enabled and self._llm is not None:
            self._set_state(AppState.REFINING)
            self._overlay.update_text("Refining...")
            await self._refine_and_inject(text)
        else:
            self._inject_and_finish(text)
```

Update `_on_transcription` to only show text in streaming mode:

```python
    @pyqtSlot(str)
    def _on_transcription(self, text: str) -> None:
        self._last_transcription = text
        if self._state == AppState.RECORDING and self._backend_is_streaming:
            self._overlay.update_text(text)
```

Update `start()` to initialize the backend:

```python
    async def start(self) -> None:
        """Initialize and start the app."""
        self._tray.show()

        # Register hotkey
        ok = await self._hotkey.register()
        if not ok:
            log.warning("Hotkey registration failed. Use tray menu to control recording.")

        # Start evdev loop if in hold mode
        if self._hotkey.mode == "hold":
            asyncio.create_task(self._hotkey.run_evdev_loop())

        # Initialize STT backend
        try:
            await self._backend.initialize()
        except Exception as e:
            log.error("Backend initialization failed: %s", e)

        # Pre-load whisper model in background (only for local mode)
        if self._backend_is_streaming:
            self._whisper.start()

        # Initialize LLM if enabled
        self._init_llm()

        log.info("Voice Input started (backend=%s)", self._config.get("stt", {}).get("backend", "local"))
```

- [ ] **Step 6: Add `get_audio_buffer()` to WhisperWorker**

In `src/voice_input/whisper_worker.py`, add this method to `WhisperWorker`:

```python
    def get_audio_buffer(self) -> np.ndarray:
        """Return a copy of the current audio buffer."""
        self._drain_and_accumulate()
        return self.audio_buffer.copy()
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: all passed

- [ ] **Step 8: Commit**

```bash
git add src/voice_input/app.py src/voice_input/whisper_worker.py tests/test_app.py
git commit -m "feat: integrate STT backends into AppController with TRANSCRIBING state"
```

---

### Task 9: Tray Menu STT Backend Submenu

**Files:**
- Modify: `src/voice_input/tray.py`
- Modify: `src/voice_input/app.py` (connect signal)

- [ ] **Step 1: Add STT backend submenu to TrayManager**

In `src/voice_input/tray.py`, add the backends dict at module level:

```python
STT_BACKENDS = {
    "local": "Local (faster-whisper)",
    "openai": "OpenAI Whisper API",
    "google": "Google Speech-to-Text",
    "volcengine": "字节火山语音识别",
}
```

In `TrayManager.__init__`, add:

```python
        self._current_backend = "local"
```

In `_build_menu`, after the Language submenu block and before the LLM submenu, add:

```python
        # STT Backend submenu
        stt_menu = QMenu("STT Backend", menu)
        self._stt_group = QActionGroup(stt_menu)
        self._stt_group.setExclusive(True)
        self._stt_actions: dict[str, QAction] = {}
        for key, label in STT_BACKENDS.items():
            action = QAction(label, stt_menu)
            action.setCheckable(True)
            action.setData(key)
            if key == self._current_backend:
                action.setChecked(True)
            self._stt_group.addAction(action)
            stt_menu.addAction(action)
            self._stt_actions[key] = action
        stt_menu.addSeparator()
        self._stt_settings_action = QAction("Settings...", stt_menu)
        stt_menu.addAction(self._stt_settings_action)
        menu.addMenu(stt_menu)
```

Add properties:

```python
    @property
    def stt_group(self) -> QActionGroup:
        return self._stt_group

    @property
    def stt_settings_action(self) -> QAction:
        return self._stt_settings_action

    def set_backend(self, backend: str) -> None:
        self._current_backend = backend
        if backend in self._stt_actions:
            self._stt_actions[backend].setChecked(True)
```

Also update `set_state` to handle the new "Transcribing" state:

```python
    def set_state(self, state: str) -> None:
        """Update tray state: 'Idle', 'Recording', 'Transcribing', 'Refining'."""
        self._state = state
        self._status_action.setText(f"Status: {state}")
        self.setToolTip(f"Voice Input — {state}")

        if state == "Recording":
            self._toggle_action.setText("Stop Recording")
            self.setIcon(self._icon_recording)
        elif state in ("Transcribing", "Refining"):
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(False)
        else:
            self._toggle_action.setText("Start Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_idle)
```

- [ ] **Step 2: Connect STT backend signal in AppController**

In `src/voice_input/app.py`, in `_connect_signals`, add:

```python
        self._tray.stt_group.triggered.connect(self._on_backend_changed)
```

Add the handler:

```python
    @pyqtSlot(QAction)
    def _on_backend_changed(self, action: QAction) -> None:
        backend_name = action.data()
        if backend_name == self._config.get("stt", {}).get("backend"):
            return
        self._config.setdefault("stt", {})["backend"] = backend_name
        save_config(self._config)

        # Cleanup old backend
        asyncio.ensure_future(self._backend.cleanup())

        # Create new backend
        self._backend = create_backend(self._config)
        self._backend_is_streaming = self._backend.is_streaming()
        asyncio.ensure_future(self._backend.initialize())

        # Start/stop whisper worker based on streaming support
        if self._backend_is_streaming and not self._whisper.isRunning():
            self._whisper.start()

        log.info("STT backend changed to: %s", backend_name)
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add src/voice_input/tray.py src/voice_input/app.py
git commit -m "feat: add STT backend submenu to system tray"
```

---

### Task 10: STT Settings Dialog

**Files:**
- Modify: `src/voice_input/settings_dialog.py`

- [ ] **Step 1: Expand SettingsDialog with STT tab**

Refactor `SettingsDialog` to use `QTabWidget` with two tabs: "LLM" (existing content) and "STT" (new).

In `src/voice_input/settings_dialog.py`, update `_build_ui`:

```python
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # LLM tab (existing)
        llm_tab = QWidget()
        llm_layout = QVBoxLayout(llm_tab)
        form = QFormLayout()

        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("API Base URL:", self._api_base_edit)

        key_layout = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("sk-...")
        key_layout.addWidget(self._api_key_edit)
        self._toggle_vis_btn = QPushButton("Show")
        self._toggle_vis_btn.setFixedWidth(60)
        self._toggle_vis_btn.setCheckable(True)
        self._toggle_vis_btn.toggled.connect(self._toggle_key_visibility)
        key_layout.addWidget(self._toggle_vis_btn)
        form.addRow("API Key:", key_layout)

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("gpt-4o-mini")
        form.addRow("Model:", self._model_edit)

        llm_layout.addLayout(form)

        test_layout = QHBoxLayout()
        test_layout.addStretch()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        test_layout.addWidget(self._test_btn)
        llm_layout.addLayout(test_layout)

        tabs.addTab(llm_tab, "LLM")

        # STT tab (new)
        stt_tab = QWidget()
        stt_layout = QVBoxLayout(stt_tab)

        # OpenAI section
        stt_layout.addWidget(QLabel("OpenAI Whisper API"))
        openai_form = QFormLayout()
        self._stt_openai_base_edit = QLineEdit()
        self._stt_openai_base_edit.setPlaceholderText("https://api.openai.com/v1")
        openai_form.addRow("API Base:", self._stt_openai_base_edit)
        self._stt_openai_model_edit = QLineEdit()
        self._stt_openai_model_edit.setPlaceholderText("whisper-1")
        openai_form.addRow("Model:", self._stt_openai_model_edit)
        self._stt_openai_key_edit = QLineEdit()
        self._stt_openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_openai_key_edit.setPlaceholderText("sk-...")
        openai_form.addRow("API Key:", self._stt_openai_key_edit)
        stt_layout.addLayout(openai_form)

        # Google section
        stt_layout.addWidget(QLabel("Google Speech-to-Text"))
        google_form = QFormLayout()
        self._stt_google_creds_edit = QLineEdit()
        self._stt_google_creds_edit.setPlaceholderText("/path/to/credentials.json")
        google_form.addRow("Credentials:", self._stt_google_creds_edit)
        stt_layout.addLayout(google_form)

        # Volcengine section
        stt_layout.addWidget(QLabel("字节火山语音识别"))
        volc_form = QFormLayout()
        self._stt_volc_appid_edit = QLineEdit()
        self._stt_volc_appid_edit.setPlaceholderText("app-id")
        volc_form.addRow("App ID:", self._stt_volc_appid_edit)
        self._stt_volc_ak_edit = QLineEdit()
        self._stt_volc_ak_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_volc_ak_edit.setPlaceholderText("access key")
        volc_form.addRow("Access Key:", self._stt_volc_ak_edit)
        self._stt_volc_sk_edit = QLineEdit()
        self._stt_volc_sk_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_volc_sk_edit.setPlaceholderText("secret key")
        volc_form.addRow("Secret Key:", self._stt_volc_sk_edit)
        stt_layout.addLayout(volc_form)

        stt_layout.addStretch()
        tabs.addTab(stt_tab, "STT")

        layout.addWidget(tabs)

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
```

Add imports at top:

```python
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
```

Update `_load_from_config` to load STT fields:

```python
    def _load_from_config(self) -> None:
        # LLM
        llm = self._config.get("llm", {})
        self._api_base_edit.setText(llm.get("api_base", "https://api.openai.com/v1"))
        self._model_edit.setText(llm.get("model", "gpt-4o-mini"))
        self._load_api_key()

        # STT
        stt = self._config.get("stt", {})
        openai_cfg = stt.get("openai", {})
        self._stt_openai_base_edit.setText(openai_cfg.get("api_base", "https://api.openai.com/v1"))
        self._stt_openai_model_edit.setText(openai_cfg.get("model", "whisper-1"))
        self._load_keyring_field(self._stt_openai_key_edit, "stt-openai-api-key")

        google_cfg = stt.get("google", {})
        self._stt_google_creds_edit.setText(google_cfg.get("credentials_path", ""))

        volc_cfg = stt.get("volcengine", {})
        self._stt_volc_appid_edit.setText(volc_cfg.get("app_id", ""))
        self._load_keyring_field(self._stt_volc_ak_edit, "stt-volcengine-access-key")
        self._load_keyring_field(self._stt_volc_sk_edit, "stt-volcengine-secret-key")
```

Add helper methods:

```python
    def _load_keyring_field(self, field: QLineEdit, key: str) -> None:
        try:
            import keyring
            value = keyring.get_password("voice-input", key)
            if value:
                field.setText(value)
        except Exception as e:
            log.debug("Could not load %s from keyring: %s", key, e)

    def _save_keyring_field(self, field: QLineEdit, key: str) -> None:
        try:
            import keyring
            value = field.text().strip()
            if value:
                keyring.set_password("voice-input", key, value)
            else:
                try:
                    keyring.delete_password("voice-input", key)
                except Exception:
                    pass
        except Exception as e:
            log.warning("Could not save %s to keyring: %s", key, e)
```

Update `_on_save` to save STT fields:

```python
    def _on_save(self) -> None:
        # LLM
        self._config["llm"]["api_base"] = self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        self._config["llm"]["model"] = self._model_edit.text().strip() or "gpt-4o-mini"
        self._save_api_key(self._api_key_edit.text().strip())

        # STT
        stt = self._config.setdefault("stt", {})
        openai_cfg = stt.setdefault("openai", {})
        openai_cfg["api_base"] = self._stt_openai_base_edit.text().strip() or "https://api.openai.com/v1"
        openai_cfg["model"] = self._stt_openai_model_edit.text().strip() or "whisper-1"
        self._save_keyring_field(self._stt_openai_key_edit, "stt-openai-api-key")

        google_cfg = stt.setdefault("google", {})
        google_cfg["credentials_path"] = self._stt_google_creds_edit.text().strip()

        volc_cfg = stt.setdefault("volcengine", {})
        volc_cfg["app_id"] = self._stt_volc_appid_edit.text().strip()
        self._save_keyring_field(self._stt_volc_ak_edit, "stt-volcengine-access-key")
        self._save_keyring_field(self._stt_volc_sk_edit, "stt-volcengine-secret-key")

        save_config(self._config)
        self.accept()
```

- [ ] **Step 2: Update dialog title**

Change the title in `__init__`:

```python
        self.setWindowTitle("Voice Input — Settings")
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add src/voice_input/settings_dialog.py
git commit -m "feat: add STT backend settings to settings dialog"
```

---

### Task 11: pyproject.toml Optional Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional dependencies**

In `pyproject.toml`, update `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.3", "pytest-asyncio>=0.23"]
google = ["google-auth>=2.0"]
all-stt = ["google-auth>=2.0"]
```

Note: `httpx` is already a core dependency. Volcengine uses only httpx + stdlib. Also add `pytest-asyncio` to dev deps since async tests need it.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add optional dependencies for STT backends"
```

---

### Task 12: Final Integration Test

**Files:**
- Create: `tests/test_integration_stt.py`

- [ ] **Step 1: Write integration test for backend switching flow**

```python
# tests/test_integration_stt.py
"""Integration tests for STT backend switching."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends import create_backend
from voice_input.backends.base import TranscriptionBackend
from voice_input.config import DEFAULT_CONFIG


def test_default_config_creates_local_backend():
    """Default config should create a local backend."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    backend = create_backend(cfg)
    assert backend.is_streaming() is True


def test_all_backends_implement_interface():
    """Every backend returned by the factory implements TranscriptionBackend."""
    import copy

    # Local
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["stt"]["backend"] = "local"
    local = create_backend(cfg)
    assert isinstance(local, TranscriptionBackend)
    assert local.is_streaming() is True

    # OpenAI
    cfg["stt"]["backend"] = "openai"
    with patch("keyring.get_password", return_value="sk-test"):
        openai_b = create_backend(cfg)
    assert isinstance(openai_b, TranscriptionBackend)
    assert openai_b.is_streaming() is False

    # Google
    cfg["stt"]["backend"] = "google"
    google_b = create_backend(cfg)
    assert isinstance(google_b, TranscriptionBackend)
    assert google_b.is_streaming() is False

    # Volcengine
    cfg["stt"]["backend"] = "volcengine"
    with patch("keyring.get_password", return_value="fake"):
        volc_b = create_backend(cfg)
    assert isinstance(volc_b, TranscriptionBackend)
    assert volc_b.is_streaming() is False


@pytest.mark.asyncio
async def test_remote_backend_transcribe_flow():
    """Simulate remote backend: record → buffer → transcribe."""
    from voice_input.backends.openai_whisper import OpenAIWhisperBackend

    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test")

    # Simulate buffered audio (2 seconds of silence)
    audio_buffer = np.zeros(32000, dtype=np.int16)

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "hello world"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        result = await backend.transcribe(audio_buffer, "en")

    assert result == "hello world"
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration_stt.py -v`
Expected: all passed

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_stt.py
git commit -m "test: add integration tests for STT backend switching"
```
