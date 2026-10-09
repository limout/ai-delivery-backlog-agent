from __future__ import annotations

import pytest

from backlog_agent.llm.exceptions import AIProviderError
from backlog_agent.llm.parse import parse_json_object


def test_parse_plain_object() -> None:
    assert parse_json_object('{"items": []}', "Gemini") == {"items": []}


def test_parse_markdown_fenced_json() -> None:
    raw = """```json
{"ok": true}
```"""
    assert parse_json_object(raw, "Gemini") == {"ok": True}


def test_parse_recovers_object_after_prefix() -> None:
    assert parse_json_object('Here you go: {"ok": true}', "Gemini") == {"ok": True}


def test_parse_rejects_empty() -> None:
    with pytest.raises(AIProviderError, match="empty"):
        parse_json_object("  ", "Gemini")


def test_parse_rejects_array_root() -> None:
    with pytest.raises(AIProviderError, match="root value is not an object"):
        parse_json_object("[1, 2]", "Gemini")


def test_parse_rejects_malformed() -> None:
    with pytest.raises(AIProviderError, match="invalid JSON"):
        parse_json_object("not json at all", "Gemini")
