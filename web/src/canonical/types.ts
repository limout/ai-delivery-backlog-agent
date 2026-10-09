import type {
  APPROVAL_STATES,
  FINDING_CODES,
  FINDING_SEVERITIES,
  PRIORITIES,
  WORK_ITEM_STATUSES,
  WORK_ITEM_TYPES,
} from "./constants";

export type WorkItemType = (typeof WORK_ITEM_TYPES)[number];
export type WorkItemStatus = (typeof WORK_ITEM_STATUSES)[number];
export type ApprovalState = (typeof APPROVAL_STATES)[number];
export type Priority = (typeof PRIORITIES)[number];
export type FindingSeverity = (typeof FINDING_SEVERITIES)[number];
export type FindingCode = (typeof FINDING_CODES)[number];

export type JsonObject = Record<string, unknown>;

export type SourceReference = {
  contract_id: string;
  json_path: string;
  field_name: string | null;
  excerpt: string | null;
  unknownFields: JsonObject;
};

export type Effort = {
  story_points: number | null;
  original_estimate_hours: number | null;
  remaining_hours: number | null;
  textual_estimate: string | null;
  unknownFields: JsonObject;
};

export type CanonicalWorkItem = {
  canonical_id: string;
  title: string;
  description: string;
  type: string;
  status: string;
  approval_state: string;
  priority: string | null;
  parent_id: string | null;
  child_ids: string[];
  acceptance_criteria: string[];
  test_requirements: string[];
  dependencies: string[];
  provenance: SourceReference[];
  effort: Effort | null;
  unknownFields: JsonObject;
};

export type Finding = {
  severity: string;
  code: string;
  message: string;
  source_references: SourceReference[];
  canonical_ids: string[];
  unknownFields: JsonObject;
  index: number;
};

export type StructuralIssue = {
  severity: FindingSeverity;
  code: string;
  message: string;
  canonicalId?: string;
};

export type CanonicalProposal = {
  filename: string;
  contract_id: string;
  backlog_id: string;
  source_contract_id: string | null;
  source_provenance: SourceReference[];
  items: CanonicalWorkItem[];
  findings: Finding[];
  findingsContractId: string | null;
  unknownFields: JsonObject;
  structuralIssues: StructuralIssue[];
};

export type ParseFailure = {
  ok: false;
  errors: string[];
};

export type ParseSuccess = {
  ok: true;
  document: CanonicalProposal;
};

export type ParseResult = ParseSuccess | ParseFailure;
