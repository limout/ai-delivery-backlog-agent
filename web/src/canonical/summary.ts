import type { CanonicalProposal, Finding, FindingSeverity, StructuralIssue } from "./types";

export type ValidationStatus = "errors" | "warnings" | "ready";

export type CoverageSummary = {
  uncoveredFunctional: Finding[];
  uncoveredNonFunctional: Finding[];
  unresolvedPrerequisites: Finding[];
  planningConstraints: Finding[];
  atomicStories: Finding[];
  generatedContent: Finding[];
};

export type FindingCategory =
  | "all"
  | "error"
  | "warning"
  | "uncovered"
  | "prerequisites"
  | "planning"
  | "generated"
  | "atomic"
  | "other";

export function overallStatus(
  document: CanonicalProposal,
): { status: ValidationStatus; label: string } {
  const hasError =
    document.structuralIssues.some((issue) => issue.severity === "error") ||
    document.findings.some((finding) => finding.severity === "error");
  if (hasError) {
    return { status: "errors", label: "Errors" };
  }
  const hasWarning =
    document.structuralIssues.some((issue) => issue.severity === "warning") ||
    document.findings.some((finding) => finding.severity === "warning");
  if (hasWarning) {
    return { status: "warnings", label: "Warnings" };
  }
  return { status: "ready", label: "Ready to review" };
}

export function coverageSummary(document: CanonicalProposal): CoverageSummary {
  return {
    uncoveredFunctional: document.findings.filter((item) => item.code === "UNCOVERED_FUNCTIONAL_REQUIREMENT"),
    uncoveredNonFunctional: document.findings.filter(
      (item) => item.code === "UNCOVERED_NON_FUNCTIONAL_REQUIREMENT",
    ),
    unresolvedPrerequisites: document.findings.filter((item) => item.code === "UNRESOLVED_PREREQUISITE"),
    planningConstraints: document.findings.filter((item) => item.code === "PLANNING_CONSTRAINT"),
    atomicStories: document.findings.filter((item) => item.code === "ATOMIC_STORY"),
    generatedContent: document.findings.filter((item) => item.code === "GENERATED_CONTENT"),
  };
}

export function countBySeverity(items: Array<Finding | StructuralIssue>): Record<FindingSeverity, number> {
  const counts: Record<FindingSeverity, number> = { error: 0, warning: 0, info: 0 };
  for (const item of items) {
    if (item.severity === "error" || item.severity === "warning" || item.severity === "info") {
      counts[item.severity] += 1;
    }
  }
  return counts;
}

export function findingCategory(finding: Finding): FindingCategory {
  if (finding.severity === "error") {
    return "error";
  }
  if (
    finding.code === "UNCOVERED_FUNCTIONAL_REQUIREMENT" ||
    finding.code === "UNCOVERED_NON_FUNCTIONAL_REQUIREMENT"
  ) {
    return "uncovered";
  }
  if (finding.code === "UNRESOLVED_PREREQUISITE") {
    return "prerequisites";
  }
  if (finding.code === "PLANNING_CONSTRAINT") {
    return "planning";
  }
  if (finding.code === "GENERATED_CONTENT") {
    return "generated";
  }
  if (finding.code === "ATOMIC_STORY") {
    return "atomic";
  }
  if (finding.severity === "warning") {
    return "warning";
  }
  return "other";
}

export function matchesFindingFilter(
  finding: Finding,
  severity: string | "all",
  category: FindingCategory,
  query: string,
): boolean {
  if (severity !== "all" && finding.severity !== severity) {
    return false;
  }
  if (category !== "all" && findingCategory(finding) !== category) {
    return false;
  }
  const haystack = [finding.code, finding.message, finding.severity, ...finding.canonical_ids]
    .join(" ")
    .toLowerCase();
  return !query || haystack.includes(query.trim().toLowerCase());
}
