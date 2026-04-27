from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.postprocess.llm import LLMRefiner, SYSTEM_PROMPT


def test_system_prompt_content() -> None:
    assert "speech recognition error corrector" in SYSTEM_PROMPT.lower()
    assert "DO NOT rewrite" in SYSTEM_PROMPT
    assert "Return ONLY the corrected text" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_refine_uses_provided_prompt() -> None:
    refiner = LLMRefiner(
        api_base="https://api.test.com/v1",
        api_key="sk-test",
        model="gpt-test",
    )
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "corrected text"}}]
    }
    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        result = await refiner.refine("raw text", prompt="custom prompt")

    assert result == "corrected text"
    assert mock_client.post.await_args.kwargs["json"]["messages"][0]["content"] == "custom prompt"
    await refiner.close()


@pytest.mark.asyncio
async def test_refine_returns_original_on_failure() -> None:
    refiner = LLMRefiner(api_base="https://api.test.com/v1", api_key="sk-test", model="m")
    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("boom"))
        result = await refiner.refine("raw text", prompt="p")

    assert result == "raw text"
    await refiner.close()


@pytest.mark.asyncio
async def test_refine_returns_original_on_empty_input() -> None:
    refiner = LLMRefiner(api_base="x", api_key="x", model="m")
    assert await refiner.refine("", prompt="p") == ""
    await refiner.close()


def test_is_configured_true_with_key() -> None:
    refiner = LLMRefiner(api_base="x", api_key="sk-x", model="m")
    assert refiner.is_configured() is True


def test_is_configured_false_without_key() -> None:
    refiner = LLMRefiner(api_base="x", api_key="", model="m")
    assert refiner.is_configured() is False
