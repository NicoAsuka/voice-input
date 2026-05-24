# src/voice_input/keyring_helper.py
"""Thin wrapper around keyring with graceful fallback when unavailable."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SERVICE = "voice-input"


def get_secret(key: str) -> str:
    """Retrieve a secret from keyring. Returns empty string on failure."""
    try:
        import keyring
        value = keyring.get_password(SERVICE, key)
        return value or ""
    except Exception as e:
        log.debug("keyring get_password(%s) failed: %s", key, e)
        return ""


def set_secret(key: str, value: str) -> None:
    """Store a secret in keyring. Deletes the entry when *value* is empty."""
    try:
        import keyring
        if value:
            keyring.set_password(SERVICE, key, value)
        else:
            try:
                keyring.delete_password(SERVICE, key)
            except keyring.errors.PasswordDeleteError as e:
                # Not-found is benign (e.g. first run); other errors are not.
                log.debug("keyring delete_password(%s): %s", key, e)
    except Exception as e:
        log.warning("keyring set_password(%s) failed: %s", key, e)
