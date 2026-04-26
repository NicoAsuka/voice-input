import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.volcengine_speech import (
    VOLCENGINE_STREAM_RESOURCE_ID,
    VOLCENGINE_STREAM_URL,
    VolcengineSpeechBackend,
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
