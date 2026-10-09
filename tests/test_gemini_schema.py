from __future__ import annotations

import json
from typing import Any

from backlog_agent.backlog.proposal import BacklogProposal, proposal_json_schema
from backlog_agent.llm.gemini import GeminiProvider
from backlog_agent.llm.gemini_schema import (
    GEMINI_UNSUPPORTED_SCHEMA_KEYS,
    schema_contains_unsupported_keys,
    to_gemini_response_schema,
)
from tests.test_gemini_provider import _FakeClient, _FakeModels, _provider


def _nested_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "uniqueItems": True,
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "local_id": {"type": "string"},
                        "count": {"const": "one"},
                        "provenance": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/Provenance",
                            },
                        },
                        "note": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                    },
                    "required": ["local_id"],
                    "oneOf": [{"required": ["local_id"]}],
                },
            }
        },
        "required": ["items"],
        "$defs": {
            "Provenance": {
                "type": "object",
                "additionalProperties": False,
                "unevaluatedProperties": False,
                "properties": {
                    "json_path": {"type": "string", "minLength": 1},
                    "excerpt": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["json_path"],
            }
        },
    }


def test_nested_unsupported_keywords_are_removed() -> None:
    original = _nested_schema()
    transformed = to_gemini_response_schema(original)

    assert original["additionalProperties"] is False
    assert original["properties"]["items"]["items"]["additionalProperties"] is False
    assert original["$defs"]["Provenance"]["additionalProperties"] is False
    assert "uniqueItems" in original["properties"]["items"]
    assert "$schema" in original

    assert schema_contains_unsupported_keys(transformed) == set()
    assert "$defs" not in transformed
    assert '"$ref"' not in json.dumps(transformed)
    assert transformed["properties"]["items"]["minItems"] == 1


def test_refs_are_inlined_and_null_anyof_becomes_nullable() -> None:
    transformed = to_gemini_response_schema(_nested_schema())
    item = transformed["properties"]["items"]["items"]
    provenance_items = item["properties"]["provenance"]["items"]

    assert provenance_items["type"] == "object"
    assert provenance_items["properties"]["json_path"]["type"] == "string"
    assert provenance_items["properties"]["json_path"]["minLength"] == 1
    excerpt = provenance_items["properties"]["excerpt"]
    assert excerpt.get("type") == "string"
    assert excerpt.get("nullable") is True
    assert "anyOf" not in excerpt

    note = item["properties"]["note"]
    assert note.get("type") == "string"
    assert note.get("nullable") is True

    assert item["properties"]["count"]["enum"] == ["one"]
    assert "const" not in item["properties"]["count"]


def test_canonical_proposal_schema_keeps_pydantic_rules() -> None:
    canonical = proposal_json_schema()
    assert canonical["additionalProperties"] is False
    assert canonical["$defs"]["ProposedItem"]["additionalProperties"] is False
    assert canonical["$defs"]["ProposedProvenance"]["additionalProperties"] is False
    assert canonical["properties"]["items"]["minItems"] == 1
    assert canonical["$defs"]["ProposedItem"]["properties"]["provenance"]["minItems"] == 1
    assert BacklogProposal.model_json_schema()["additionalProperties"] is False

    gemini_schema = to_gemini_response_schema(canonical)
    assert schema_contains_unsupported_keys(gemini_schema) == set()
    assert gemini_schema["properties"]["items"]["minItems"] == 1
    item_schema = gemini_schema["properties"]["items"]["items"]
    assert item_schema["properties"]["provenance"]["minItems"] == 1
    assert item_schema["properties"]["type"]["enum"] == [
        "epic",
        "feature",
        "user_story",
        "task",
        "subtask",
    ]
    assert item_schema["properties"]["title_origin"]["enum"] == ["source", "generated"]
    assert '"additionalProperties"' not in json.dumps(gemini_schema)


def test_gemini_client_receives_compatible_schema_not_canonical() -> None:
    canonical = proposal_json_schema()
    models = _FakeModels(text='{"items":[{"local_id":"e1"}]}')
    provider = _provider(models)

    provider.generate_json("prompt", canonical)

    sent = models.calls[0]["config"]["response_schema"]
    assert schema_contains_unsupported_keys(sent) == set()
    assert '"additionalProperties"' not in json.dumps(sent)
    assert "$defs" not in sent
    assert sent["properties"]["items"]["minItems"] == 1
    assert canonical["additionalProperties"] is False
    assert canonical is not sent


def test_simple_object_schema_still_round_trips() -> None:
    models = _FakeModels(text='{"items": [{"local_id": "e1"}]}')
    parsed = GeminiProvider(client=_FakeClient(models), retry_delay=0).generate_json(
        "prompt", {"type": "object"}
    )
    assert parsed == {"items": [{"local_id": "e1"}]}
    assert models.calls[0]["config"]["response_schema"] == {"type": "object"}
