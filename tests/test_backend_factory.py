import pytest


def test_create_backend_local():
    from voice_input.backends import create_backend

    config = {
        "stt": {
            "backend": "local",
            "local": {
                "engine": "whisper",
                "model": "tiny",
                "language": "zh",
                "device": "cpu",
            },
        },
    }
    backend = create_backend(config)
    from voice_input.backends.local import LocalBackend

    assert isinstance(backend, LocalBackend)


def test_create_backend_unknown_raises():
    from voice_input.backends import create_backend

    config = {"stt": {"backend": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(config)


def test_create_backend_default_is_local():
    from voice_input.backends import create_backend

    config = {
        "stt": {
            "local": {
                "engine": "whisper",
                "model": "tiny",
                "language": "zh",
                "device": "cpu",
            },
        },
    }
    backend = create_backend(config)
    from voice_input.backends.local import LocalBackend

    assert isinstance(backend, LocalBackend)
