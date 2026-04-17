# tests/test_audio.py
import queue
import numpy as np

from voice_input.audio import compute_rms, AudioRecorder


def test_compute_rms_silence():
    """RMS of silence should be 0."""
    samples = np.zeros(256, dtype=np.int16)
    assert compute_rms(samples) == 0.0


def test_compute_rms_known_signal():
    """RMS of a constant signal should equal abs(value) / 32768."""
    value = 1000
    samples = np.full(256, value, dtype=np.int16)
    expected = value / 32768.0
    assert abs(compute_rms(samples) - expected) < 1e-6


def test_audio_recorder_init():
    """AudioRecorder should be constructable with queues."""
    wq = queue.Queue()
    vq = queue.Queue()
    rec = AudioRecorder(whisper_queue=wq, viz_queue=vq, device="default", sample_rate=16000)
    assert rec.sample_rate == 16000
    assert not rec.is_recording
