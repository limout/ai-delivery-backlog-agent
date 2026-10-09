export const CANONICAL_BACKLOG_V1 = "canonical.backlog.v1";
export const GENERATION_FINDINGS_V1 = "canonical.generation_findings.v1";

export const WORK_ITEM_TYPES = [
  "epic",
  "feature",
  "user_story",
  "task",
  "subtask",
] as const;

export const WORK_ITEM_STATUSES = [
  "todo",
  "in_progress",
  "done",
  "blocked",
  "cancelled",
] as const;

export const APPROVAL_STATES = ["not_approved", "approved"] as const;

export const PRIORITIES = ["critical", "high", "medium", "low"] as const;

export const FINDING_SEVERITIES = ["error", "warning", "info"] as const;

export const FINDING_CODES = [
  "UNCOVERED_FUNCTIONAL_REQUIREMENT",
  "COPILOT_CONTRADICTION",
  "SCOPE_CONFLICT",
  "AMBIGUOUS_DECOMPOSITION",
  "OUT_OF_SCOPE_EXCLUDED",
  "ID_COLLISION",
  "LLM_VALIDATION_FAILED",
  "UNGROUNDED_PROVENANCE",
  "GENERATED_CONTENT",
  "UNRESOLVED_PREREQUISITE",
  "UNCOVERED_NON_FUNCTIONAL_REQUIREMENT",
  "PLANNING_CONSTRAINT",
  "ATOMIC_STORY",
] as const;

export const KNOWN_ITEM_FIELDS = [
  "canonical_id",
  "title",
  "description",
  "status",
  "approval_state",
  "priority",
  "parent_id",
  "child_ids",
  "acceptance_criteria",
  "test_requirements",
  "dependencies",
  "provenance",
  "effort",
  "type",
] as const;

export const KNOWN_BACKLOG_FIELDS = [
  "contract_id",
  "backlog_id",
  "items",
  "source_contract_id",
  "source_provenance",
] as const;

export const KNOWN_ENVELOPE_FIELDS = ["backlog", "findings"] as const;

export const KNOWN_FINDING_FIELDS = [
  "severity",
  "code",
  "message",
  "source_references",
  "canonical_ids",
] as const;

export const KNOWN_FINDINGS_FIELDS = ["contract_id", "items"] as const;

export const WORK_ITEM_TYPE_LABELS: Record<(typeof WORK_ITEM_TYPES)[number], string> = {
  epic: "Epic",
  feature: "Feature",
  user_story: "User Story",
  task: "Task",
  subtask: "Subtask",
};
