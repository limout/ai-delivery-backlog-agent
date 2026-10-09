import {
  CANONICAL_BACKLOG_V1,
  GENERATION_FINDINGS_V1,
  KNOWN_BACKLOG_FIELDS,
  KNOWN_ENVELOPE_FIELDS,
  KNOWN_FINDING_FIELDS,
  KNOWN_FINDINGS_FIELDS,
  KNOWN_ITEM_FIELDS,
  WORK_ITEM_TYPES,
} from "./constants";
import type {
  CanonicalProposal,
  CanonicalWorkItem,
  Effort,
  Finding,
  JsonObject,
  ParseResult,
  SourceReference,
  StructuralIssue,
} from "./types";

const ITEM_FIELD_SET = new Set<string>(KNOWN_ITEM_FIELDS);
const BACKLOG_FIELD_SET = new Set<string>(KNOWN_BACKLOG_FIELDS);
const ENVELOPE_FIELD_SET = new Set<string>(KNOWN_ENVELOPE_FIELDS);
const FINDING_FIELD_SET = new Set<string>(KNOWN_FINDING_FIELDS);
const FINDINGS_FIELD_SET = new Set<string>(KNOWN_FINDINGS_FIELDS);
const TYPE_SET = new Set<string>(WORK_ITEM_TYPES);
const SOURCE_REF_FIELDS = new Set([
  "contract_id",
  "json_path",
  "field_name",
  "excerpt",
]);
const EFFORT_FIELDS = new Set([
  "story_points",
  "original_estimate_hours",
  "remaining_hours",
  "textual_estimate",
]);

export function parseProposalText(text: string, filename: string): ParseResult {
  const trimmed = text.trim();
  if (!trimmed) {
    return { ok: false, errors: ["The file is empty."] };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown JSON error";
    return { ok: false, errors: [`Malformed JSON: ${detail}`] };
  }

  return parseProposalJson(parsed, filename);
}

export function parseProposalJson(parsed: unknown, filename: string): ParseResult {
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      ok: false,
      errors: ["Unsupported format: expected a JSON object with a canonical backlog."],
    };
  }

  const root = parsed as JsonObject;
  const errors: string[] = [];
  const issues: StructuralIssue[] = [];

  let backlogRaw: JsonObject;
  let findingsRaw: unknown;
  let envelopeUnknown: JsonObject = {};

  if (isObject(root.backlog)) {
    backlogRaw = root.backlog;
    findingsRaw = root.findings;
    envelopeUnknown = pickUnknown(root, ENVELOPE_FIELD_SET);
  } else if (typeof root.contract_id === "string") {
    backlogRaw = root;
    findingsRaw = undefined;
  } else {
    return {
      ok: false,
      errors: [
        "Unsupported format: expected { backlog, findings } from the agent CLI, or a canonical.backlog.v1 object.",
      ],
    };
  }

  const contractId = asString(backlogRaw.contract_id);
  if (!contractId) {
    errors.push("Missing required field backlog.contract_id.");
  } else if (contractId !== CANONICAL_BACKLOG_V1) {
    errors.push(
      `Unsupported contract version ${contractId}. This review UI supports ${CANONICAL_BACKLOG_V1}.`,
    );
  }

  if (!Array.isArray(backlogRaw.items)) {
    errors.push("Missing required field backlog.items (must be an array).");
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const itemsRaw = backlogRaw.items as unknown[];
  const items: CanonicalWorkItem[] = [];
  itemsRaw.forEach((entry, index) => {
    const parsedItem = parseItem(entry, index, errors, issues);
    if (parsedItem) {
      items.push(parsedItem);
    }
  });

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const backlogId = asString(backlogRaw.backlog_id) ?? "";
  if (!backlogId) {
    issues.push({
      severity: "warning",
      code: "MISSING_BACKLOG_ID",
      message: "backlog_id is missing.",
    });
  } else if (!backlogId.startsWith("cbl_")) {
    issues.push({
      severity: "warning",
      code: "UNEXPECTED_BACKLOG_ID",
      message: `backlog_id does not use the canonical cbl_ prefix: ${backlogId}`,
    });
  }

  const sourceProvenance = parseSourceRefs(backlogRaw.source_provenance, "backlog.source_provenance", issues);

  let findings: Finding[] = [];
  let findingsContractId: string | null = null;
  if (findingsRaw !== undefined) {
    const parsedFindings = parseFindings(findingsRaw, errors, issues);
    findings = parsedFindings.items;
    findingsContractId = parsedFindings.contractId;
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  collectReferenceIssues(items, issues);

  const document: CanonicalProposal = {
    filename,
    contract_id: contractId ?? CANONICAL_BACKLOG_V1,
    backlog_id: backlogId,
    source_contract_id: asString(backlogRaw.source_contract_id),
    source_provenance: sourceProvenance,
    items,
    findings,
    findingsContractId,
    unknownFields: {
      ...envelopeUnknown,
      ...prefixUnknown("backlog", pickUnknown(backlogRaw, BACKLOG_FIELD_SET)),
    },
    structuralIssues: issues,
  };

  return { ok: true, document };
}

function parseItem(
  entry: unknown,
  index: number,
  errors: string[],
  issues: StructuralIssue[],
): CanonicalWorkItem | null {
  const label = `items[${index}]`;
  if (!isObject(entry)) {
    errors.push(`${label} must be an object.`);
    return null;
  }

  const canonicalId = asString(entry.canonical_id);
  const title = asString(entry.title);
  const type = asString(entry.type);
  if (!canonicalId) {
    errors.push(`${label} is missing required field canonical_id.`);
  }
  if (!title) {
    errors.push(`${label} is missing required field title.`);
  }
  if (!type) {
    errors.push(`${label} is missing required field type.`);
  }
  if (!canonicalId || !title || !type) {
    return null;
  }

  if (!canonicalId.startsWith("cbl_")) {
    issues.push({
      severity: "warning",
      code: "UNEXPECTED_ITEM_ID",
      message: `${canonicalId} does not use the canonical cbl_ prefix.`,
      canonicalId,
    });
  }
  if (!TYPE_SET.has(type)) {
    issues.push({
      severity: "warning",
      code: "UNKNOWN_ITEM_TYPE",
      message: `${canonicalId} has unrecognized type ${type}.`,
      canonicalId,
    });
  }

  const childIds = parseIdList(entry.child_ids, `${label}.child_ids`, issues, canonicalId);
  const dependencies = parseIdList(entry.dependencies, `${label}.dependencies`, issues, canonicalId);

  return {
    canonical_id: canonicalId,
    title,
    description: asString(entry.description) ?? "",
    type,
    status: asString(entry.status) ?? "todo",
    approval_state: asString(entry.approval_state) ?? "not_approved",
    priority: asString(entry.priority),
    parent_id: asString(entry.parent_id),
    child_ids: childIds,
    acceptance_criteria: parseStringList(entry.acceptance_criteria, `${label}.acceptance_criteria`, issues, canonicalId),
    test_requirements: parseStringList(entry.test_requirements, `${label}.test_requirements`, issues, canonicalId),
    dependencies,
    provenance: parseSourceRefs(entry.provenance, `${label}.provenance`, issues, canonicalId),
    effort: parseEffort(entry.effort, `${label}.effort`, issues, canonicalId),
    unknownFields: pickUnknown(entry, ITEM_FIELD_SET),
  };
}

function parseFindings(
  raw: unknown,
  errors: string[],
  issues: StructuralIssue[],
): { items: Finding[]; contractId: string | null } {
  if (!isObject(raw)) {
    errors.push("findings must be an object when present.");
    return { items: [], contractId: null };
  }
  const contractId = asString(raw.contract_id);
  if (contractId && contractId !== GENERATION_FINDINGS_V1) {
    issues.push({
      severity: "warning",
      code: "UNSUPPORTED_FINDINGS_CONTRACT",
      message: `Findings contract ${contractId} is not ${GENERATION_FINDINGS_V1}; findings are shown as-is.`,
    });
  }
  if (raw.items !== undefined && !Array.isArray(raw.items)) {
    errors.push("findings.items must be an array.");
    return { items: [], contractId };
  }
  const items = Array.isArray(raw.items) ? raw.items : [];
  const findings: Finding[] = [];
  items.forEach((entry, index) => {
    if (!isObject(entry)) {
      issues.push({
        severity: "warning",
        code: "INVALID_FINDING",
        message: `findings.items[${index}] is not an object and was skipped.`,
      });
      return;
    }
    const message = asString(entry.message);
    if (!message) {
      issues.push({
        severity: "warning",
        code: "INVALID_FINDING",
        message: `findings.items[${index}] is missing a message.`,
      });
    }
    findings.push({
      severity: asString(entry.severity) ?? "info",
      code: asString(entry.code) ?? "UNKNOWN",
      message: message ?? "(missing message)",
      source_references: parseSourceRefs(entry.source_references, `findings.items[${index}].source_references`, issues),
      canonical_ids: parseIdList(entry.canonical_ids, `findings.items[${index}].canonical_ids`, issues),
      unknownFields: pickUnknown(entry, FINDING_FIELD_SET),
      index,
    });
  });
  const extra = pickUnknown(raw, FINDINGS_FIELD_SET);
  if (Object.keys(extra).length > 0) {
    issues.push({
      severity: "info",
      code: "UNKNOWN_FIELDS",
      message: `Findings envelope contains extra fields: ${Object.keys(extra).join(", ")}.`,
    });
  }
  return { items: findings, contractId };
}

function parseSourceRefs(
  raw: unknown,
  path: string,
  issues: StructuralIssue[],
  canonicalId?: string,
): SourceReference[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  if (!Array.isArray(raw)) {
    issues.push({
      severity: "warning",
      code: "INVALID_PROVENANCE",
      message: `${path} must be an array.`,
      canonicalId,
    });
    return [];
  }
  const refs: SourceReference[] = [];
  raw.forEach((entry, index) => {
    if (!isObject(entry)) {
      issues.push({
        severity: "warning",
        code: "INVALID_PROVENANCE",
        message: `${path}[${index}] is not an object.`,
        canonicalId,
      });
      return;
    }
    const jsonPath = asString(entry.json_path);
    const contractId = asString(entry.contract_id) ?? "";
    if (!jsonPath) {
      issues.push({
        severity: "warning",
        code: "INVALID_PROVENANCE",
        message: `${path}[${index}] is missing json_path.`,
        canonicalId,
      });
    }
    refs.push({
      contract_id: contractId,
      json_path: jsonPath ?? "",
      field_name: asString(entry.field_name),
      excerpt: asString(entry.excerpt),
      unknownFields: pickUnknown(entry, SOURCE_REF_FIELDS),
    });
  });
  return refs;
}

function parseEffort(
  raw: unknown,
  path: string,
  issues: StructuralIssue[],
  canonicalId: string,
): Effort | null {
  if (raw === undefined || raw === null) {
    return null;
  }
  if (!isObject(raw)) {
    issues.push({
      severity: "warning",
      code: "INVALID_EFFORT",
      message: `${path} must be an object.`,
      canonicalId,
    });
    return null;
  }
  return {
    story_points: asNumber(raw.story_points),
    original_estimate_hours: asNumber(raw.original_estimate_hours),
    remaining_hours: asNumber(raw.remaining_hours),
    textual_estimate: asString(raw.textual_estimate),
    unknownFields: pickUnknown(raw, EFFORT_FIELDS),
  };
}

function parseIdList(
  raw: unknown,
  path: string,
  issues: StructuralIssue[],
  canonicalId?: string,
): string[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  if (!Array.isArray(raw)) {
    issues.push({
      severity: "warning",
      code: "INVALID_ID_LIST",
      message: `${path} must be an array of strings.`,
      canonicalId,
    });
    return [];
  }
  const ids: string[] = [];
  const seen = new Set<string>();
  raw.forEach((entry, index) => {
    const value = asString(entry);
    if (!value) {
      issues.push({
        severity: "warning",
        code: "INVALID_ID_LIST",
        message: `${path}[${index}] is not a non-empty string.`,
        canonicalId,
      });
      return;
    }
    if (seen.has(value)) {
      issues.push({
        severity: "warning",
        code: "DUPLICATE_ID",
        message: `${path} contains duplicate id ${value}.`,
        canonicalId,
      });
      return;
    }
    seen.add(value);
    ids.push(value);
  });
  return ids;
}

function parseStringList(
  raw: unknown,
  path: string,
  issues: StructuralIssue[],
  canonicalId: string,
): string[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  if (!Array.isArray(raw)) {
    issues.push({
      severity: "warning",
      code: "INVALID_STRING_LIST",
      message: `${path} must be an array of strings.`,
      canonicalId,
    });
    return [];
  }
  return raw.map((entry, index) => {
    const value = asString(entry);
    if (value === null) {
      issues.push({
        severity: "warning",
        code: "INVALID_STRING_LIST",
        message: `${path}[${index}] is not a string.`,
        canonicalId,
      });
      return String(entry);
    }
    return value;
  });
}

function collectReferenceIssues(items: CanonicalWorkItem[], issues: StructuralIssue[]): void {
  const byId = new Map(items.map((item) => [item.canonical_id, item]));
  const seenIds = new Set<string>();
  for (const item of items) {
    if (seenIds.has(item.canonical_id)) {
      issues.push({
        severity: "error",
        code: "DUPLICATE_CANONICAL_ID",
        message: `Duplicate canonical_id ${item.canonical_id}.`,
        canonicalId: item.canonical_id,
      });
    }
    seenIds.add(item.canonical_id);

    if (Object.keys(item.unknownFields).length > 0) {
      issues.push({
        severity: "info",
        code: "UNKNOWN_FIELDS",
        message: `${item.canonical_id} contains extra fields: ${Object.keys(item.unknownFields).join(", ")}.`,
        canonicalId: item.canonical_id,
      });
    }

    if (item.parent_id) {
      const parent = byId.get(item.parent_id);
      if (!parent) {
        issues.push({
          severity: "error",
          code: "BROKEN_PARENT",
          message: `${item.canonical_id} parent_id ${item.parent_id} does not exist.`,
          canonicalId: item.canonical_id,
        });
      }
    } else if (item.type !== "epic") {
      issues.push({
        severity: "warning",
        code: "UNPARENTED_ITEM",
        message: `${item.type} ${item.canonical_id} has no parent.`,
        canonicalId: item.canonical_id,
      });
    }

    for (const childId of item.child_ids) {
      const child = byId.get(childId);
      if (!child) {
        issues.push({
          severity: "error",
          code: "BROKEN_CHILD",
          message: `${item.canonical_id} child_id ${childId} does not exist.`,
          canonicalId: item.canonical_id,
        });
      }
    }

    for (const dependencyId of item.dependencies) {
      if (!byId.has(dependencyId)) {
        issues.push({
          severity: "error",
          code: "BROKEN_DEPENDENCY",
          message: `${item.canonical_id} dependency ${dependencyId} does not exist.`,
          canonicalId: item.canonical_id,
        });
      }
    }
  }

  detectCycles(items, issues);
}

function detectCycles(items: CanonicalWorkItem[], issues: StructuralIssue[]): void {
  const byId = new Map(items.map((item) => [item.canonical_id, item]));
  const visiting = new Set<string>();
  const visited = new Set<string>();

  const walk = (id: string): boolean => {
    if (visited.has(id) || !byId.has(id)) {
      return false;
    }
    if (visiting.has(id)) {
      return true;
    }
    visiting.add(id);
    const item = byId.get(id);
    if (item) {
      for (const childId of item.child_ids) {
        if (walk(childId)) {
          return true;
        }
      }
    }
    visiting.delete(id);
    visited.add(id);
    return false;
  };

  for (const item of items) {
    if (walk(item.canonical_id)) {
      issues.push({
        severity: "error",
        code: "PARENT_CYCLE",
        message: "The backlog contains a parent/child cycle.",
        canonicalId: item.canonical_id,
      });
      break;
    }
  }
}

function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function pickUnknown(object: JsonObject, known: Set<string>): JsonObject {
  const extra: JsonObject = {};
  for (const [key, value] of Object.entries(object)) {
    if (!known.has(key)) {
      extra[key] = value;
    }
  }
  return extra;
}

function prefixUnknown(prefix: string, extra: JsonObject): JsonObject {
  if (Object.keys(extra).length === 0) {
    return {};
  }
  return { [prefix]: extra };
}
