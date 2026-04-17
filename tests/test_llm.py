# tests/test_llm.py
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from voice_input.llm import LLMRefiner, SYSTEM_PROMPT


def test_system_prompt_content():
    """System prompt should instruct conservative correction only."""
    assert "speech recognition error corrector" in SYSTEM_PROMPT.lower()
    assert "DO NOT rewrite" in SYSTEM_PROMPT
    assert "Return ONLY the corrected text" in SYSTEM_PROMPT


def test_refiner_init():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    assert refiner.api_base == "https://api.openai.com/v1"
    assert refiner.model == "gpt-4o-mini"
    assert refiner.timeout == 5.0


@pytest.mark.asyncio
async def test_refine_returns_corrected_text():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "corrected text"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        result = await refiner.refine("original text")
        assert result == "corrected text"


@pytest.mark.asyncio
async def test_refine_returns_original_on_timeout():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        result = await refiner.refine("original text")
        assert result == "original text"


@pytest.mark.asyncio
async def test_refine_empty_input():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    result = await refiner.refine("")
    assert result == ""
