from __future__ import annotations

from backlog_agent.source.request_headings import (
    heading_appears_represented,
    parse_labeled_request_outcomes,
    parse_narrative_sections,
)


def test_parse_labeled_headings_ignores_bullets_and_paragraphs() -> None:
    text = (
        "Build a portal.\n"
        "- FR-style bullet about widgets\n"
        "FR-04 is mentioned in a sentence without a heading.\n"
        "### FR-01: Customer Authentication\n"
        "Details about login.\n"
        "## NFR-07: Completely Unique Widget Enumeration\n"
        "### FR-08: Renewal Status\n"
        "**NFR-02: Reliability And Recovery**\n"
        "FR-10: Audit Trail\n"
    )
    headings = parse_labeled_request_outcomes(text)
    labels = [item.label for item in headings]
    assert labels == ["FR-01", "NFR-07", "FR-08", "NFR-02", "FR-10"]
    assert headings[3].title == "Reliability And Recovery"
    assert headings[0].title == "Customer Authentication"
    assert not any(item.label == "FR-04" for item in headings)


def test_narrative_section_excerpt_is_an_exact_slice() -> None:
    text = (
        "Intro.\n"
        "### FR-04: Renewal Eligibility\n"
        "An ineligible policy cannot proceed.\n"
        "### NFR-04: Performance and Availability\n"
        "Performance targets have not been defined.\n"
    )
    sections = parse_narrative_sections(text)
    by_label = {section.label: section for section in sections}
    assert set(by_label) == {"FR-04", "NFR-04"}
    assert by_label["FR-04"].evidence_kind == "outcome"
    assert by_label["NFR-04"].evidence_kind == "unconfirmed"
    assert by_label["FR-04"].json_path == "$.user_request"
    assert by_label["FR-04"].excerpt in text
    assert by_label["FR-04"].excerpt.startswith("### FR-04: Renewal Eligibility")
    assert "cannot proceed" in by_label["FR-04"].excerpt
    assert "NFR-04" not in by_label["FR-04"].excerpt
    assert by_label["FR-04"].excerpt != text


def test_heading_token_heuristic_true_positive_and_false_negative() -> None:
    assert heading_appears_represented(
        "Renewal Status",
        "Customers can view available renewal status after authentication.",
    )
    assert heading_appears_represented(
        "Catalog Search",
        "The system shall let a member search the catalog by title or author.",
    )
    assert heading_appears_represented(
        "Customer Authentication",
        "A customer can authenticate through the existing identity provider.",
    )
    assert heading_appears_represented(
        "Customer Authentication",
        "The system must integrate with the existing identity provider for customer authentication.",
    )
    # False negative limitation: semantic coverage without shared tokens is missed.
    assert not heading_appears_represented(
        "Customer Authentication",
        "Members sign in through the existing identity provider.",
    )
    # Shared short prefixes are not treated as the same outcome.
    assert not heading_appears_represented(
        "Data Integrity",
        "The portal completes integration with the catalog service.",
    )
    assert heading_appears_represented(
        "Policy Listing",
        "Review Personal Policies. A customer reviews personal policies and inspects coverage.",
    )
    assert not heading_appears_represented(
        "Policy Listing",
        "A member reviews renewal terms before confirming a change.",
    )
    assert not heading_appears_represented(
        "Catalog Listing",
        "A member searches the catalog by title or author.",
    )
    # Distinctive unused heading is not treated as represented.
    assert not heading_appears_represented(
        "Completely Unique Widget Enumeration",
        "Members search the catalog and place a hold.",
    )
