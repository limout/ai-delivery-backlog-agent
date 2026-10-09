"""Supported input adapters. Decomposition consumes NormalizedProjectSource only."""

from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.source.copilot import from_copilot
from backlog_agent.source.model import NormalizedProjectSource, SourceFact

__all__ = [
    "NormalizedProjectSource",
    "SourceFact",
    "UnsupportedInputError",
    "load_project_source",
]


class UnsupportedInputError(ValueError):
    """Raised when the payload is not a supported project-input contract."""


def load_project_source(data: object) -> NormalizedProjectSource:
    """Load the first supported input contract. Does not accept arbitrary JSON."""

    if isinstance(data, NormalizedProjectSource):
        return data
    try:
        envelope = (
            data
            if isinstance(data, CopilotWorkflowResponseV1)
            else CopilotWorkflowResponseV1.model_validate(data)
        )
    except Exception as exc:
        detail = str(exc).split("\n", 1)[0]
        raise UnsupportedInputError(
            "Unsupported or insufficient input. This agent currently accepts a COMPLETE "
            "copilot.workflow_response.v1 /analyze JSON envelope (not tracker exports, "
            "plain text, or arbitrary JSON). Fix the input contract or add an adapter. "
            f"Adapter detail: {detail}"
        ) from exc
    return from_copilot(envelope)
