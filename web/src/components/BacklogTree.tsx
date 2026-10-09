import { ChevronDown, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { ReviewIndex, TreeNode } from "@/canonical/tree";
import { displayTitle } from "@/canonical/tree";
import { formatType } from "@/lib/utils";

type BacklogTreeProps = {
  nodes: TreeNode[];
  orphans: TreeNode[];
  missingChildren: { parentId: string; childId: string }[];
  index: ReviewIndex;
  selectedId: string | null;
  expanded: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
};

export function BacklogTree({
  nodes,
  orphans,
  missingChildren,
  index,
  selectedId,
  expanded,
  onSelect,
  onToggle,
}: BacklogTreeProps) {
  return (
    <div className="h-full overflow-auto px-2 py-2" data-testid="backlog-tree">
      {nodes.length === 0 && orphans.length === 0 ? (
        <p className="px-3 py-8 text-sm text-muted-foreground">No work items match the current filters.</p>
      ) : null}
      {nodes.map((node) => (
        <TreeRow
          key={node.item.canonical_id}
          node={node}
          index={index}
          selectedId={selectedId}
          expanded={expanded}
          onSelect={onSelect}
          onToggle={onToggle}
        />
      ))}
      {orphans.length > 0 ? (
        <section className="mt-3 border-t px-2 pt-3">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Unparented items
          </p>
          {orphans.map((node) => (
            <TreeRow
              key={node.item.canonical_id}
              node={node}
              index={index}
              selectedId={selectedId}
              expanded={expanded}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          ))}
        </section>
      ) : null}
      {missingChildren.length > 0 ? (
        <section className="mt-3 border-t px-2 pt-3">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Broken child references
          </p>
          {missingChildren.map((entry) => (
            <p key={`${entry.parentId}:${entry.childId}`} className="px-2 py-1 text-xs text-destructive">
              {entry.parentId} points at missing child {entry.childId}
            </p>
          ))}
        </section>
      ) : null}
    </div>
  );
}

function TreeRow({
  node,
  index,
  selectedId,
  expanded,
  onSelect,
  onToggle,
}: {
  node: TreeNode;
  index: ReviewIndex;
  selectedId: string | null;
  expanded: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
}) {
  const id = node.item.canonical_id;
  const hasChildren = node.children.length > 0;
  const isOpen = expanded.has(id);
  const selected = selectedId === id;
  const findings = index.findingsByItem.get(id) ?? [];
  const issues = index.issuesByItem.get(id) ?? [];
  const hasError =
    findings.some((item) => item.severity === "error") ||
    issues.some((item) => item.severity === "error");
  const hasWarning =
    findings.some((item) => item.severity === "warning") ||
    issues.some((item) => item.severity === "warning");

  return (
    <div>
      <div
        className={`flex items-start gap-1 rounded-md px-1 py-1 ${selected ? "bg-accent" : "hover:bg-accent/70"}`}
        style={{ paddingLeft: `${node.depth * 12 + 4}px` }}
      >
        <button
          type="button"
          className="mt-0.5 size-5 shrink-0 rounded text-muted-foreground"
          aria-label={hasChildren ? (isOpen ? "Collapse" : "Expand") : "No children"}
          disabled={!hasChildren}
          onClick={() => onToggle(id)}
        >
          {hasChildren ? (
            isOpen ? (
              <ChevronDown className="size-4" />
            ) : (
              <ChevronRight className="size-4" />
            )
          ) : (
            <span className="block size-4" />
          )}
        </button>
        <button
          type="button"
          data-testid={`select-${id}`}
          className="min-w-0 flex-1 text-left"
          onClick={() => onSelect(id)}
        >
          <span className="flex flex-wrap items-center gap-1.5">
            <Badge variant="outline">{formatType(node.item.type)}</Badge>
            {node.item.status && node.item.status !== "todo" ? (
              <Badge variant="neutral">{formatType(node.item.status)}</Badge>
            ) : null}
            {node.item.priority ? <Badge variant="secondary">{node.item.priority}</Badge> : null}
            {hasError ? <Badge variant="critical">error</Badge> : hasWarning ? <Badge variant="warning">warning</Badge> : null}
            {findings.some((item) => item.code === "GENERATED_CONTENT") ? (
              <Badge variant="info">generated</Badge>
            ) : null}
            {findings.some((item) => item.code === "ATOMIC_STORY") ? (
              <Badge variant="neutral">atomic</Badge>
            ) : null}
          </span>
          <span className="mt-0.5 block truncate text-sm" title={node.item.title}>
            {displayTitle(node.item)}
          </span>
        </button>
      </div>
      {isOpen
        ? node.children.map((child) => (
            <TreeRow
              key={child.item.canonical_id}
              node={child}
              index={index}
              selectedId={selectedId}
              expanded={expanded}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          ))
        : null}
    </div>
  );
}
