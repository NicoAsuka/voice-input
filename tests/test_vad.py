from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from voice_input.asr.vad import VadTrimmer


def test_unavailable_when_not_loaded():
    vad = VadTrimmer()
    assert vad.available() is False


def test_trim_returns_input_when_unavailable():
    vad = VadTrimmer()
    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    # 未加载时直接透传
    assert np.array_equal(out, samples)


def test_load_initializes_vad():
    vad = VadTrimmer()
    fake_vad = MagicMock()
    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))
    assert vad.available() is True
    fake_sherpa.VoiceActivityDetector.assert_called_once()


def test_load_disables_vad_when_config_construction_fails(caplog):
    vad = VadTrimmer()

    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.SileroVadModelConfig.side_effect = RuntimeError("bad config")
        vad.load(Path("/tmp/silero_vad.onnx"))

    assert vad.available() is False
    assert "Failed to load VAD: bad config" in caplog.text


def test_trim_concatenates_speech_segments():
    vad = VadTrimmer()
    fake_vad = MagicMock()

    # Stub VAD to emit two segments
    seg1 = MagicMock()
    seg1.samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    seg2 = MagicMock()
    seg2.samples = np.array([0.4, 0.5], dtype=np.float32)

    # Simulate VAD streaming API: accept_waveform → flush → is_empty/front/pop
    states = {"emitted": [seg1, seg2], "flushed": False}

    def is_empty():
        return len(states["emitted"]) == 0

    def front():
        return states["emitted"][0]

    def pop():
        states["emitted"].pop(0)

    fake_vad.is_empty.side_effect = is_empty
    fake_vad.front.side_effect = front
    fake_vad.pop.side_effect = pop

    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))

    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    assert out.dtype == np.float32
    assert np.allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_trim_returns_empty_for_no_speech():
    vad = VadTrimmer()
    fake_vad = MagicMock()
    fake_vad.is_empty.return_value = True

    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))

    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    assert len(out) == 0
