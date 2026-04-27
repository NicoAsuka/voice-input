from __future__ import annotations

import enum
import hashlib
import json


class RegistryState(enum.Enum):
    LOADING = "loading"
    READY = "ready"
    RELOADING = "reloading"
    ERROR = "error"


def compute_signature(config: dict) -> str:
    """Compute a stable fingerprint for STT-relevant config only."""
    stt = config.get("stt", {})
    relevant = {
        "backend": stt.get("backend"),
        "sherpa": stt.get("sherpa"),
        "volcengine": stt.get("volcengine"),
        "google": stt.get("google"),
        "openai": stt.get("openai"),
    }
    serialized = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
