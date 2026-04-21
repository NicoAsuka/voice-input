# tests/test_whisper_worker.py
import queue
import numpy as np
from unittest.mock import MagicMock

from voice_input.whisper_worker import WhisperWorker


def _worker(wq: queue.Queue) -> WhisperWorker:
    return WhisperWorker(whisper_queue=wq, backend=MagicMock(), language="zh")


def test_drain_queue_concatenates_chunks():
    """drain_queue should concatenate all available chunks."""
    wq = queue.Queue()
    worker = _worker(wq)
    chunk1 = np.zeros(512, dtype=np.int16)
    chunk2 = np.ones(256, dtype=np.int16)
    wq.put(chunk1)
    wq.put(chunk2)
    result = worker.drain_queue()
    assert len(result) == 768
    assert result.dtype == np.int16


def test_drain_queue_empty():
    """drain_queue on empty queue returns empty array."""
    wq = queue.Queue()
    worker = _worker(wq)
    result = worker.drain_queue()
    assert len(result) == 0


def test_buffer_accumulates():
    """Audio buffer should accumulate across drain calls."""
    wq = queue.Queue()
    worker = _worker(wq)
    wq.put(np.zeros(100, dtype=np.int16))
    worker.accumulate()
    assert len(worker.audio_buffer) == 100
    wq.put(np.ones(200, dtype=np.int16))
    worker.accumulate()
    assert len(worker.audio_buffer) == 300


def test_reset_clears_buffer():
    wq = queue.Queue()
    worker = _worker(wq)
    wq.put(np.zeros(100, dtype=np.int16))
    worker.accumulate()
    worker.reset()
    assert len(worker.audio_buffer) == 0
