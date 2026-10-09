"""Gemini Developer API response_schema compatibility.

Pydantic JSON Schema is the application contract. Gemini's Schema protobuf
rejects several JSON Schema keywords (notably ``additionalProperties``, even
when false). Transform only at the Gemini provider boundary; do not change
Pydantic models to match the vendor.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

# Keywords the Gemini Developer API Schema object has rejected or does not
# document as generation_config.response_schema fields. extra=forbid in
# Pydantic emits additionalProperties:false, which the API 400s on (the SDK
# only raises for truthy additionalProperties).
GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "additionalProperties",
        "additional_properties",
        "$schema",
        "$id",
        "$comment",
        "unevaluatedProperties",
        "unevaluatedItems",
        "patternProperties",
        "propertyNames",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "not",
        "allOf",
        "oneOf",
        "one_of",
        "uniqueItems",
        "unique_items",
        "prefixItems",
        "prefix_items",
        "contentMediaType",
        "contentEncoding",
        "deprecated",
        "readOnly",
        "writeOnly",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "examples",
    }
)


def to_gemini_response_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Gemini-safe copy of ``schema`` without mutating the original."""

    root = deepcopy(dict(schema))
    defs = root.pop("$defs", None) or root.pop("defs", None) or {}
    resolved: dict[str, Any] = {name: deepcopy(value) for name, value in defs.items()}
    for name, value in list(resolved.items()):
        resolved[name] = _transform_node(value, resolved)
    return _transform_node(root, resolved)


def schema_contains_unsupported_keys(schema: Mapping[str, Any] | list[Any] | Any) -> set[str]:
    """Return unsupported keys found anywhere in a schema tree."""

    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            found.update(key for key in node if key in GEMINI_UNSUPPORTED_SCHEMA_KEYS)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return found


def _transform_node(node: Any, defs: Mapping[str, Any]) -> Any:
    if not isinstance(node, dict):
        return node

    current = dict(node)
    ref = current.pop("$ref", None) or current.pop("ref", None)
    if ref is not None:
        name = str(ref).rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"Gemini schema $ref {ref!r} does not resolve")
        inlined = deepcopy(defs[name])
        if not isinstance(inlined, dict):
            raise ValueError(f"Gemini schema $ref {ref!r} did not resolve to an object")
        overlay = {key: value for key, value in current.items()}
        current = {**inlined, **overlay}

    _flatten_nullable_any_of(current)

    const = current.pop("const", None)
    if const is not None:
        current["enum"] = [const]

    for key in list(current):
        if key in GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            del current[key]

    if isinstance(current.get("properties"), dict):
        current["properties"] = {
            name: _transform_node(value, defs) for name, value in current["properties"].items()
        }
        if len(current["properties"]) > 1 and "propertyOrdering" not in current:
            current["propertyOrdering"] = list(current["properties"].keys())

    if "items" in current:
        current["items"] = _transform_node(current["items"], defs)

    if isinstance(current.get("anyOf"), list):
        current["anyOf"] = [_transform_node(value, defs) for value in current["anyOf"]]

    return current


def _flatten_nullable_any_of(node: dict[str, Any]) -> None:
    variants = node.get("anyOf")
    if not isinstance(variants, list):
        return
    non_null: list[Any] = []
    nullable = False
    for variant in variants:
        if isinstance(variant, dict) and variant.get("type") == "null":
            nullable = True
        else:
            non_null.append(variant)
    if not nullable:
        return
    node["nullable"] = True
    if len(non_null) == 1 and isinstance(non_null[0], dict):
        for key, value in non_null[0].items():
            node.setdefault(key, value)
        del node["anyOf"]
    elif non_null:
        node["anyOf"] = non_null
    else:
        del node["anyOf"]
