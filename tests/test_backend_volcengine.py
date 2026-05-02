import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.base import BackendCapabilities, BackendDescriptor, RecognitionError
from voice_input.backends.volcengine_speech import (
    VOLCENGINE_STREAM_RESOURCE_ID,
    VOLCENGINE_STREAM_URL,
    VolcengineSpeechBackend,
    _VolcSession,
)


def test_is_streaming_returns_false():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    assert backend.is_streaming() is False


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = VolcengineSpeechBackend(
        app_id="test-app",
        access_key="ak",
        secret_key="sk",
        resource_id="volc.test.resource",
    )

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.responses = [
                backend._build_server_response({"result": {"text": ""}}, final=False),
                backend._build_server_response(
                    {"result": {"text": "transcribed text"}}, final=True
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            return self.responses.pop(0)

    ws = FakeWebSocket()
    with patch.object(backend, "_connect", return_value=ws) as connect:
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "transcribed text"
    connect.assert_called_once()
    assert connect.call_args.args[0]["X-Api-App-Key"] == "test-app"
    assert connect.call_args.args[0]["X-Api-Access-Key"] == "ak"
    assert connect.call_args.args[0]["X-Api-Resource-Id"] == "volc.test.resource"
    assert connect.call_args.args[1] == VOLCENGINE_STREAM_URL

    request = backend._parse_server_payload_like_client_request(ws.sent[0])
    assert request["audio"] == {
        "format": "pcm",
        "codec": "raw",
        "rate": 16000,
        "bits": 16,
        "channel": 1,
        "language": "zh-CN",
    }
    assert request["request"]["model_name"] == "bigmodel"
    assert ws.sent[-1][1] == 0x22


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")

    with patch.object(backend, "_connect", side_effect=Exception("network error")):
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""


def test_sign_request_produces_headers():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    headers = backend._build_headers()
    assert headers["X-Api-App-Key"] == "test-app"
    assert headers["X-Api-Access-Key"] == "ak"
    assert headers["X-Api-Resource-Id"] == VOLCENGINE_STREAM_RESOURCE_ID
    assert "X-Api-Connect-Id" in headers


def test_parse_server_response_decompresses_json():
    backend = VolcengineSpeechBackend()
    frame = backend._build_server_response(
        {"result": {"text": "最终文本"}},
        final=True,
    )
    payload, final = backend._parse_server_response(frame)
    assert payload["result"]["text"] == "最终文本"
    assert final is True


def test_normalize_language_for_bigmodel():
    backend = VolcengineSpeechBackend()
    assert backend._normalize_language("zh") == "zh-CN"
    assert backend._normalize_language("en") == "en-US"
    assert backend._normalize_language("ja-JP") == "ja-JP"


# --- New Session interface tests ---


def test_is_ready_with_credentials():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    assert backend.is_ready() is True


def test_is_ready_missing_credentials():
    backend = VolcengineSpeechBackend(app_id="", access_key="ak", secret_key="sk")
    assert backend.is_ready() is False

    backend = VolcengineSpeechBackend(app_id="test-app", access_key="", secret_key="sk")
    assert backend.is_ready() is False

    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="")
    assert backend.is_ready() is False


def test_describe_returns_descriptor():
    backend = VolcengineSpeechBackend(
        app_id="test-app", access_key="ak", secret_key="sk",
        resource_id="volc.test.resource",
    )
    desc = backend.describe()
    assert isinstance(desc, BackendDescriptor)
    assert desc.backend_id == "volcengine"
    assert desc.model_id == "volc.test.resource"
    assert desc.capabilities.supports_streaming is True
    assert desc.capabilities.requires_network is True
    assert desc.capabilities.supports_vad is False


def test_create_session_returns_session():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = backend.create_session("zh")
    assert isinstance(session, _VolcSession)
    assert session._language == "zh"


@pytest.mark.asyncio
async def test_shutdown_is_noop():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    await backend.shutdown()  # should not raise


@pytest.mark.asyncio
async def test_session_finish_returns_text():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    audio = np.zeros(16000, dtype=np.int16)
    session.push_audio(audio)

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="hello"):
        result = await session.finish()
    assert result == "hello"


@pytest.mark.asyncio
async def test_session_finish_concatenates_chunks():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.push_audio(np.zeros(200, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="ok") as mock_t:
        await session.finish()
    assert mock_t.call_args[0][0].shape == (300,)


@pytest.mark.asyncio
async def test_session_finish_empty_returns_empty():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_cancel_returns_empty():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.cancel()
    assert session._cancelled is True
    assert len(session._buffer) == 0
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_finish_raises_recognition_error():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, side_effect=Exception("ws down")):
        with pytest.raises(RecognitionError) as exc_info:
            await session.finish()
    assert "Volcengine error" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_session_finish_reraises_recognition_error():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))

    original_error = RecognitionError("original", user_message="original msg")
    with patch.object(backend, "transcribe", new_callable=AsyncMock, side_effect=original_error):
        with pytest.raises(RecognitionError, match="original"):
            await session.finish()


@pytest.mark.asyncio
async def test_session_push_audio_while_cancelled():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    session = _VolcSession(backend, "zh")
    session.cancel()
    session.push_audio(np.zeros(100, dtype=np.int16))
    assert len(session._buffer) == 0
