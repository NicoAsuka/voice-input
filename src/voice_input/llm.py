# src/voice_input/llm.py
from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a speech recognition error corrector. ONLY fix obvious transcription errors, especially:\n"
    "- Chinese homophone errors from ASR\n"
    "- English technical terms mis-transcribed as Chinese phonetics "
    "(e.g. 配森→Python, 杰森→JSON, 锐克特→React, 瑞迪斯→Redis, 多克→Docker, 哥拉格→GraphQL)\n"
    "- Mixed Chinese-English where English terms got corrupted\n"
    "Rules (HARD):\n"
    "- DO NOT rewrite, polish, paraphrase, or expand anything\n"
    "- DO NOT change punctuation unless it's clearly wrong\n"
    "- DO NOT add explanation, quotes, or markdown\n"
    "- If the input looks correct, return it EXACTLY as-is\n"
    "- Return ONLY the corrected text."
)


class LLMRefiner:
    """Calls an OpenAI-compatible API to correct ASR transcription errors."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 5.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            trust_env=False,
        )

    async def refine(self, text: str) -> str:
        """Send text for LLM correction. Returns corrected text, or original on failure."""
        if not text:
            return text
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": len(text) * 3 + 100,
                },
            )
            response.raise_for_status()
            data = response.json()
            corrected = data["choices"][0]["message"]["content"].strip()
            if corrected:
                return corrected
            return text
        except Exception as e:
            log.warning("LLM refinement failed, using original text: %s", e)
            return text

    async def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity. Returns (success, message)."""
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            response.raise_for_status()
            return True, "Connection successful"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return False, str(e)

    async def close(self) -> None:
        await self._client.aclose()
