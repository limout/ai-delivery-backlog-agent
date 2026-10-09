import { Badge } from "@/components/ui/badge";
import type { FindingCategory } from "@/canonical/summary";
import { matchesFindingFilter } from "@/canonical/summary";
import type { CanonicalProposal, Finding } from "@/canonical/types";
import type { ReviewIndex } from "@/canonical/tree";
import { displayTitle } from "@/canonical/tree";
import { truncate } from "@/lib/utils";

type FindingsPanelProps = {
  document: CanonicalProposal;
  index: ReviewIndex;
  severity: string | "all";
  category: FindingCategory;
  query: string;
  onSelectItem: (id: string) => void;
};

const CATEGORIES: { id: FindingCategory; label: string }[] = [
  { id: "all", label: "All" },
  { id: "error", label: "Errors" },
  { id: "warning", label: "Warnings" },
  { id: "uncovered", label: "Uncovered" },
  { id: "prerequisites", label: "Prerequisites" },
  { id: "planning", label: "Planning" },
  { id: "generated", label: "Generated" },
  { id: "atomic", label: "Atomic stories" },
  { id: "other", label: "Other" },
];

export function FindingsPanel({
  document,
  index,
  severity,
  category,
  query,
  onSelectItem,
}: FindingsPanelProps) {
  const filtered = document.findings.filter((finding) =>
    matchesFindingFilter(finding, severity, category, query),
  );
  const structural = document.structuralIssues.filter((issue) => {
    if (severity !== "all" && issue.severity !== severity) {
      return false;
    }
    if (category === "generated" || category === "atomic" || category === "uncovered" || category === "prerequisites" || category === "planning") {
      return false;
    }
    if (category === "error" && issue.severity !== "error") {
      return false;
    }
    if (category === "warning" && issue.severity !== "warning") {
      return false;
    }
    const haystack = `${issue.code} ${issue.message}`.toLowerCase();
    return !query || haystack.includes(query.trim().toLowerCase());
  });

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="findings-panel">
      <div className="flex flex-wrap gap-1 border-b px-3 py-2">
        {CATEGORIES.map((entry) => (
          <span key={entry.id} className="sr-only">
            {entry.label}
          </span>
        ))}
        <p className="text-xs text-muted-foreground">
          {filtered.length} findings
          {structural.length ? ` · ${structural.length} structural issues` : ""}
        </p>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-3 py-2">
        {filtered.length === 0 && structural.length === 0 ? (
          <p className="py-8 text-sm text-muted-foreground">No findings match the current filters.</p>
        ) : null}
        {filtered.map((finding) => (
          <FindingCard
            key={`finding-${finding.index}`}
            finding={finding}
            index={index}
            onSelectItem={onSelectItem}
          />
        ))}
        {structural.map((issue, idx) => (
          <article key={`issue-${issue.code}-${idx}`} className="mb-2 rounded-md border p-3 text-sm">
            <div className="flex flex-wrap gap-1.5">
              <SeverityBadge severity={issue.severity} />
              <Badge variant="outline">{issue.code}</Badge>
            </div>
            <p className="mt-1">{issue.message}</p>
            {issue.canonicalId && index.byId.has(issue.canonicalId) ? (
              <button
                type="button"
                className="mt-2 text-xs underline"
                onClick={() => onSelectItem(issue.canonicalId as string)}
              >
                Open {displayTitle(index.byId.get(issue.canonicalId)!)}
              </button>
            ) : null}
          </article>
        ))}
      </div>
    </div>
  );
}

function FindingCard({
  finding,
  index,
  onSelectItem,
}: {
  finding: Finding;
  index: ReviewIndex;
  onSelectItem: (id: string) => void;
}) {
  return (
    <article className="mb-2 rounded-md border p-3 text-sm">
      <div className="flex flex-wrap gap-1.5">
        <SeverityBadge severity={finding.severity} />
        <Badge variant="outline">{finding.code}</Badge>
      </div>
      <p className="mt-1">{finding.message}</p>
      {finding.canonical_ids.length > 0 ? (
        <ul className="mt-2 space-y-1">
          {finding.canonical_ids.map((id) => {
            const item = index.byId.get(id);
            return (
              <li key={id}>
                {item ? (
                  <button type="button" className="text-left text-xs underline" onClick={() => onSelectItem(id)}>
                    Open {displayTitle(item)}
                  </button>
                ) : (
                  <span className="text-xs text-destructive">Unknown item {id}</span>
                )}
              </li>
            );
          })}
        </ul>
      ) : null}
      {finding.source_references.length > 0 ? (
        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
          {finding.source_references.map((ref, idx) => (
            <li key={`${ref.json_path}:${idx}`}>
              <span className="font-mono">{ref.json_path}</span>
              {ref.excerpt ? ` — ${truncate(ref.excerpt, 180)}` : ""}
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const variant =
    severity === "error" ? "critical" : severity === "warning" ? "warning" : "info";
  return <Badge variant={variant}>{severity}</Badge>;
}
