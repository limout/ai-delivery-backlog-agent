import { FolderOpen } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { coverageSummary, overallStatus } from "@/canonical/summary";
import { typeCounts } from "@/canonical/tree";
import type { CanonicalProposal } from "@/canonical/types";
import { formatType } from "@/lib/utils";

type ReviewHeaderProps = {
  document: CanonicalProposal;
  onNewFile: () => void;
};

export function ReviewHeader({ document, onNewFile }: ReviewHeaderProps) {
  const status = overallStatus(document);
  const coverage = coverageSummary(document);
  const counts = typeCounts(document.items);
  const titleItem = document.items.find((item) => item.type === "epic") ?? document.items[0];
  const title = titleItem
    ? titleItem.title.split(/\r?\n/, 1)[0]?.trim() || "Canonical backlog"
    : "Canonical backlog";
  const statusVariant =
    status.status === "errors" ? "critical" : status.status === "warnings" ? "warning" : "healthy";

  return (
    <header className="space-y-4 border-b bg-card px-6 py-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Backlog review
          </p>
          <h1 className="truncate text-xl font-semibold tracking-tight" title={title}>
            {title}
          </h1>
          <p className="text-sm text-muted-foreground">
            {document.filename}
            {document.backlog_id ? ` · ${document.backlog_id}` : ""}
            {document.source_contract_id ? ` · source ${document.source_contract_id}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={statusVariant}>{status.label}</Badge>
          <Button variant="outline" size="sm" onClick={onNewFile}>
            <FolderOpen className="size-4" />
            Load another file
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {counts
          .filter(({ count }) => count > 0)
          .map(({ type, count }) => (
          <Badge key={type} variant="secondary">
            {formatType(type)} {count}
          </Badge>
        ))}
        <Badge variant="outline">{document.items.length} items</Badge>
        {coverage.uncoveredFunctional.length > 0 ? (
          <Badge variant="critical">
            Uncovered functional {coverage.uncoveredFunctional.length}
          </Badge>
        ) : null}
        {coverage.uncoveredNonFunctional.length > 0 ? (
          <Badge variant="warning">
            Uncovered non-functional {coverage.uncoveredNonFunctional.length}
          </Badge>
        ) : null}
        {coverage.unresolvedPrerequisites.length > 0 ? (
          <Badge variant="warning">
            Unresolved prerequisites {coverage.unresolvedPrerequisites.length}
          </Badge>
        ) : null}
        {document.findings.filter((item) => item.severity === "error").length > 0 ? (
          <Badge variant="critical">
            Finding errors {document.findings.filter((item) => item.severity === "error").length}
          </Badge>
        ) : null}
        {document.findings.filter((item) => item.severity === "warning").length > 0 ? (
          <Badge variant="warning">
            Finding warnings {document.findings.filter((item) => item.severity === "warning").length}
          </Badge>
        ) : null}
        {document.structuralIssues.filter((item) => item.severity === "error").length > 0 ? (
          <Badge variant="critical">
            Structural errors{" "}
            {document.structuralIssues.filter((item) => item.severity === "error").length}
          </Badge>
        ) : null}
      </div>
    </header>
  );
}
