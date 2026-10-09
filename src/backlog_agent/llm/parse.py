"""Shared JSON recovery for provider responses."""

from __future__ import annotations

import json

from backlog_agent.llm.exceptions import AIProviderError


def parse_json_object(raw_text: str, provider_name: str) -> dict:
    """Parse a JSON object, allowing markdown fences or a short prefix."""

    if not isinstance(raw_text, str):
        raise AIProviderError(f"{provider_name} returned an unexpected JSON value.")

    text = raw_text.strip()
    if not text:
        raise AIProviderError(f"{provider_name} returned an empty response.")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise AIProviderError(
                f"{provider_name} returned JSON, but the root value is not an object."
            )
        return result
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    preview = text[:500].replace("\n", "\\n")
    raise AIProviderError(
        f"{provider_name} returned invalid JSON. Response preview: {preview}"
    )
