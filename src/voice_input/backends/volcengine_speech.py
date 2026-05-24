from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import logging
import struct
import uuid
from typing import Any

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
VOLCENGINE_STREAM_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
VOLCENGINE_STREAM_RESOURCE_ID = "volc.seedasr.sauc.duration"

_PROTOCOL_VERSION = 0x1
_HEADER_SIZE_WORDS = 0x1
_SERIALIZATION_NONE = 0x0
_SERIALIZATION_JSON = 0x1
_COMPRESSION_NONE = 0x0
_COMPRESSION_GZIP = 0x1

_MSG_FULL_CLIENT_REQUEST = 0x1
_MSG_AUDIO_ONLY_REQUEST = 0x2
_MSG_FULL_SERVER_RESPONSE = 0x9
_MSG_ERROR = 0xF

_FLAG_NO_SEQUENCE = 0x0
_FLAG_POS_SEQUENCE = 0x1
_FLAG_FINAL_NO_SEQUENCE = 0x2
_FLAG_FINAL_NEG_SEQUENCE = 0x3


class VolcengineSpeechBackend(TranscriptionBackend):
    """Volcengine bigmodel streaming speech-to-text API backend."""

    def __init__(
        self,
        app_id: str = "",
        access_key: str = "",
        secret_key: str = "",
        resource_id: str = VOLCENGINE_STREAM_RESOURCE_ID,
        timeout: float = 30.0,
        websocket_url: str = VOLCENGINE_STREAM_URL,
    ) -> None:
        self.app_id = app_id
        self.access_key = access_key
        self.secret_key = secret_key
        self.resource_id = resource_id
        self.timeout = timeout
        self.websocket_url = websocket_url

    def is_streaming(self) -> bool:
        return False

    async def initialize(self) -> None:
        if not self.app_id or not self.access_key:
            log.warning("Volcengine credentials not fully configured")
            return
        log.info(
            "Volcengine Speech backend initialized (app_id=%s, resource_id=%s)",
            self.app_id,
            self.resource_id,
        )

    def _build_headers(self) -> dict[str, str]:
        return {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def _connect(self, headers: dict[str, str], url: str):
        try:
            import websockets
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "websockets is not installed. Install with: pip install websockets"
            ) from exc

        kwargs: dict[str, Any] = {
            "open_timeout": self.timeout,
            "close_timeout": self.timeout,
        }
        header_param = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )
        kwargs[header_param] = headers
        return websockets.connect(url, **kwargs)

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if not self.app_id or not self.access_key:
            log.error("Volcengine Speech credentials are not configured")
            return ""

        try:
            async with self._connect(self._build_headers(), self.websocket_url) as ws:
                await ws.send(self._build_full_client_request(language))
                await self._send_audio(ws, audio_data)
                return await self._receive_text(ws)
        except Exception as e:
            log.error("Volcengine Speech API error: %s", e)
            return ""

    def _build_full_client_request(self, language: str) -> bytes:
        payload = {
            "user": {"uid": self.app_id},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
                "language": self._normalize_language(language),
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": False,
            },
        }
        return self._build_payload_frame(
            _MSG_FULL_CLIENT_REQUEST,
            _FLAG_NO_SEQUENCE,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            serialization=_SERIALIZATION_JSON,
            compression=_COMPRESSION_GZIP,
        )

    async def _send_audio(self, ws: Any, audio_data: np.ndarray) -> None:
        audio = np.asarray(audio_data, dtype=np.int16)
        samples_per_chunk = SAMPLE_RATE // 5  # 200ms
        if audio.size == 0:
            await ws.send(self._build_audio_frame(b"", final=True))
            return

        for start in range(0, audio.size, samples_per_chunk):
            end = min(start + samples_per_chunk, audio.size)
            final = end >= audio.size
            await ws.send(self._build_audio_frame(audio[start:end].tobytes(), final))

    async def _receive_text(self, ws: Any) -> str:
        latest_text = ""
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            payload, final = self._parse_server_response(message)
            text = self._extract_text(payload)
            if text:
                latest_text = text
            if final:
                return latest_text

    def _build_audio_frame(self, audio_bytes: bytes, final: bool) -> bytes:
        return self._build_payload_frame(
            _MSG_AUDIO_ONLY_REQUEST,
            _FLAG_FINAL_NO_SEQUENCE if final else _FLAG_NO_SEQUENCE,
            audio_bytes,
            serialization=_SERIALIZATION_NONE,
            compression=_COMPRESSION_GZIP,
        )

    def _build_payload_frame(
        self,
        message_type: int,
        flags: int,
        payload: bytes,
        *,
        serialization: int,
        compression: int,
    ) -> bytes:
        encoded = gzip.compress(payload) if compression == _COMPRESSION_GZIP else payload
        header = bytes(
            [
                (_PROTOCOL_VERSION << 4) | _HEADER_SIZE_WORDS,
                (message_type << 4) | flags,
                (serialization << 4) | compression,
                0x00,
            ]
        )
        return header + struct.pack(">I", len(encoded)) + encoded

    def _parse_server_response(self, frame: bytes) -> tuple[dict[str, Any], bool]:
        if len(frame) < 8:
            raise ValueError("Volcengine response frame is too short")

        header_size = (frame[0] & 0x0F) * 4
        message_type = frame[1] >> 4
        flags = frame[1] & 0x0F
        serialization = frame[2] >> 4
        compression = frame[2] & 0x0F
        offset = header_size

        if message_type == _MSG_ERROR:
            if len(frame) < offset + 8:
                raise ValueError("Volcengine error frame is too short")
            code = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            size = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            message = frame[offset : offset + size].decode("utf-8", errors="replace")
            raise RuntimeError(f"Volcengine error {code}: {message}")

        if message_type != _MSG_FULL_SERVER_RESPONSE:
            raise ValueError(f"Unexpected Volcengine message type: {message_type}")

        if flags in (_FLAG_POS_SEQUENCE, _FLAG_FINAL_NEG_SEQUENCE):
            if len(frame) < offset + 4:
                raise ValueError("Volcengine response frame is missing sequence")
            offset += 4

        if len(frame) < offset + 4:
            raise ValueError("Volcengine response frame is missing payload size")
        size = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4
        payload = frame[offset : offset + size]
        if len(payload) != size:
            raise ValueError("Volcengine response payload is truncated")

        if compression == _COMPRESSION_GZIP:
            payload = gzip.decompress(payload)
        if serialization == _SERIALIZATION_JSON:
            data = json.loads(payload.decode("utf-8"))
        elif serialization == _SERIALIZATION_NONE:
            data = {"payload": payload.decode("utf-8", errors="replace")}
        else:
            raise ValueError(f"Unsupported Volcengine serialization: {serialization}")

        return data, flags in (_FLAG_FINAL_NO_SEQUENCE, _FLAG_FINAL_NEG_SEQUENCE)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        result = payload.get("result")
        if isinstance(result, dict):
            text = result.get("text")
            return text.strip() if isinstance(text, str) else ""
        if isinstance(result, list):
            texts = [item.get("text", "") for item in result if isinstance(item, dict)]
            return "".join(texts).strip()
        return ""

    def _normalize_language(self, language: str) -> str:
        mapping = {
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "cn": "zh-CN",
            "en": "en-US",
            "en-us": "en-US",
            "ja": "ja-JP",
            "yue": "yue-CN",
        }
        normalized = mapping.get(language.strip().lower())
        return normalized or language

    def is_ready(self) -> bool:
        return bool(self.app_id and self.access_key and self.secret_key)

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="volcengine",
            model_id=self.resource_id,
            capabilities=BackendCapabilities(
                supports_streaming=True,
                requires_network=True,
                supports_vad=False,
            ),
        )

    def create_session(self, language: str) -> Session:
        return _VolcSession(self, language)

    async def shutdown(self) -> None:
        pass


class _VolcSession(Session):
    def __init__(self, backend: VolcengineSpeechBackend, language: str) -> None:
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
        except RecognitionError:
            raise
        except Exception as e:
            raise RecognitionError(
                str(e), user_message=f"Volcengine error: {str(e)[:80]}"
            ) from e
