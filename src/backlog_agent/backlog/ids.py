"""Canonical identifier generation, independent of tracker IDs.

Generated backlog **items** use RFC 4122 UUID version 5 (SHA-1, name-based):

- Namespace UUID: ``uuid5(NAMESPACE_URL, "https://limout.ai/contracts/canonical.backlog.v1")``
- Name: content-addressed identity
  ``{base_json_path}|{normalized_text}|{occurrence}``
  where ``occurrence`` is the 0-based count of earlier entries in the same
  list with the same normalized text. Array indexes are not part of the name,
  so inserting unrelated rows does not change an existing item's ID.
- Format: ``cbl_`` + the UUID5 string

Provenance ``json_path`` values remain index-based (for example
``$.requirements.functional_requirements[1]``) so they point at the live
envelope. Those paths are not identity.

``backlog_id`` is an envelope fingerprint (UUID5 of a SHA-256 of canonical
JSON). It is not item identity and changes if any envelope field changes.

This scheme does not claim cross-version identity: changing the algorithm,
namespace, or item text mints a new ID.

``new_canonical_id()`` remains available for hand-built test graphs (UUID4).
The generator never calls it for Copilot-derived items.
"""

from __future__ import annotations

import uuid

CANONICAL_ID_PREFIX = "cbl"
CANONICAL_ID_ALGORITHM = "rfc4122-uuid5-sha1"
CANONICAL_ID_NAMESPACE_NAME = "https://limout.ai/contracts/canonical.backlog.v1"
CANONICAL_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, CANONICAL_ID_NAMESPACE_NAME)


def new_canonical_id() -> str:
    """Return a newly generated random canonical ID for hand-built items."""

    return f"{CANONICAL_ID_PREFIX}_{uuid.uuid4()}"


def normalize_source_text(value: str) -> str:
    """Collapse whitespace and case-fold source text for identity keys."""

    return " ".join(value.casefold().split())


def item_identity_key(base_path: str, text: str, occurrence: int) -> str:
    """Return the UUID5 name for a generated item (not a JSONPath)."""

    if occurrence < 0:
        raise ValueError("occurrence must be >= 0")
    return f"{base_path}|{normalize_source_text(text)}|{occurrence}"


def canonical_id_from_source_key(source_key: str) -> str:
    """Return a deterministic canonical ID derived from a stable source key."""

    if not source_key or not source_key.strip():
        raise ValueError("source_key must be a non-empty string")
    return f"{CANONICAL_ID_PREFIX}_{uuid.uuid5(CANONICAL_ID_NAMESPACE, source_key)}"


def canonical_id_from_item_identity(base_path: str, text: str, occurrence: int) -> str:
    return canonical_id_from_source_key(item_identity_key(base_path, text, occurrence))
