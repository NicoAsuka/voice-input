from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.asr.model_manager import ModelInfo
from voice_input.backends.base import RecognitionError
from voice_input.backends.sherpa_backend import SherpaBackend, SherpaSession


def make_config(model_id="sherpa-onnx-paraformer-zh-2024-03-09", vad=True):
    return {
        "stt": {
            "sherpa": {
                "model_id": model_id,
                "vad_enabled": vad,
                "num_threads": 2,
                "provider": "cpu",
            }
        }
    }


def make_model_info(tmp_path):
    return ModelInfo(
        model_id="sherpa-onnx-paraformer-zh-2024-03-09",
        family="paraformer",
        paths={"model": tmp_path / "model.onnx", "tokens": tmp_path / "tokens.txt"},
        language="zh-en",
        size_bytes=237_000_000,
    )


def test_is_ready_false_before_initialize():
    backend = SherpaBackend(make_config())
    assert backend.is_ready() is False


def test_describe_returns_descriptor_with_model_id():
    backend = SherpaBackend(make_config())
    desc = backend.describe()
    assert desc.backend_id == "sherpa"
    assert desc.model_id == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert desc.capabilities.supports_vad is True
    assert desc.capabilities.requires_network is False


@pytest.mark.asyncio
async def test_initialize_loads_model_and_vad(tmp_path):
    backend = SherpaBackend(make_config())
    info = make_model_info(tmp_path)

    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.VadTrimmer"
    ) as VT, patch("voice_input.backends.sherpa_backend.sherpa_onnx") as SO:
        mgr_instance = MM.return_value
        mgr_instance.ensure_model = AsyncMock(return_value=info)
        mgr_instance.ensure_vad_model = AsyncMock(
            return_value=tmp_path / "silero_vad.onnx"
        )
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()
        vad_instance = VT.return_value

        await backend.initialize()

    assert backend.is_ready() is True
    mgr_instance.ensure_model.assert_awaited_once()
    mgr_instance.ensure_vad_model.assert_awaited_once()
    vad_instance.load.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_skips_vad_when_disabled(tmp_path):
    backend = SherpaBackend(make_config(vad=False))
    info = make_model_info(tmp_path)

    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.VadTrimmer"
    ) as VT, patch("voice_input.backends.sherpa_backend.sherpa_onnx") as SO:
        mgr_instance = MM.return_value
        mgr_instance.ensure_model = AsyncMock(return_value=info)
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()

        await backend.initialize()

    VT.assert_not_called()
    mgr_instance.ensure_vad_model.assert_not_called()


@pytest.mark.asyncio
async def test_create_session_returns_sherpa_session(tmp_path):
    backend = SherpaBackend(make_config(vad=False))
    info = make_model_info(tmp_path)
    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.sherpa_onnx"
    ) as SO:
        MM.return_value.ensure_model = AsyncMock(return_value=info)
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()
        await backend.initialize()
    session = backend.create_session(language="zh")
    assert isinstance(session, SherpaSession)


@pytest.mark.asyncio
async def test_session_finish_returns_text():
    fake_recognizer = MagicMock()
    fake_stream = MagicMock()
    fake_stream.result.text = "你好世界"
    fake_recognizer.create_stream.return_value = fake_stream

    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    pcm = (np.ones(16000, dtype=np.float32) * 0.5 * 32768).astype(np.int16)
    session.push_audio(pcm)
    text = await session.finish()
    assert text == "你好世界"


@pytest.mark.asyncio
async def test_session_finish_returns_empty_for_no_audio():
    fake_recognizer = MagicMock()
    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    text = await session.finish()
    assert text == ""


@pytest.mark.asyncio
async def test_session_finish_raises_recognition_error_on_failure():
    fake_recognizer = MagicMock()
    fake_recognizer.create_stream.side_effect = RuntimeError("inference crashed")

    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    pcm = np.ones(16000, dtype=np.int16)
    session.push_audio(pcm)
    with pytest.raises(RecognitionError) as exc_info:
        await session.finish()
    assert "识别" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_session_with_vad_filters_silence():
    fake_recognizer = MagicMock()
    fake_stream = MagicMock()
    fake_stream.result.text = "filtered"
    fake_recognizer.create_stream.return_value = fake_stream

    fake_vad = MagicMock()
    fake_vad.available.return_value = True
    fake_vad.trim.return_value = np.array([], dtype=np.float32)

    session = SherpaSession(recognizer=fake_recognizer, vad=fake_vad, language="zh")
    pcm = np.zeros(16000, dtype=np.int16)
    session.push_audio(pcm)
    text = await session.finish()
    assert text == ""
