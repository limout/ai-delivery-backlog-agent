import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import type { CanonicalWorkItem, SourceReference } from "@/canonical/types";
import type { ReviewIndex } from "@/canonical/tree";
import { displayTitle } from "@/canonical/tree";
import { formatType, truncate } from "@/lib/utils";

type ItemDetailProps = {
  item: CanonicalWorkItem | null;
  index: ReviewIndex;
  onSelect: (id: string) => void;
};

export function ItemDetail({ item, index, onSelect }: ItemDetailProps) {
  if (!item) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-sm text-muted-foreground">
        Select a work item to inspect its details.
      </div>
    );
  }

  const parent = item.parent_id ? index.byId.get(item.parent_id) : undefined;
  const children = item.child_ids
    .map((id) => ({ id, item: index.byId.get(id) }))
    .filter(Boolean);
  const dependents = (index.dependents.get(item.canonical_id) ?? [])
    .map((id) => ({ id, item: index.byId.get(id) }));
  const findings = index.findingsByItem.get(item.canonical_id) ?? [];
  const issues = index.issuesByItem.get(item.canonical_id) ?? [];
  const generated = findings.filter((finding) => finding.code === "GENERATED_CONTENT");

  return (
    <div className="h-full overflow-auto px-5 py-4" data-testid="item-detail">
      <div className="space-y-1">
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline">{formatType(item.type)}</Badge>
          <Badge variant="neutral">{formatType(item.status)}</Badge>
          <Badge variant="secondary">{formatType(item.approval_state)}</Badge>
          {item.priority ? <Badge variant="info">{item.priority}</Badge> : null}
          {generated.length > 0 ? <Badge variant="info">Generated content</Badge> : null}
        </div>
        <h2 className="text-lg font-semibold tracking-tight">{displayTitle(item)}</h2>
        <p className="font-mono text-xs text-muted-foreground">{item.canonical_id}</p>
      </div>

      {item.description ? (
        <Section title="Description">
          <p className="whitespace-pre-wrap text-sm leading-6">{item.description}</p>
        </Section>
      ) : (
        <Section title="Description">
          <p className="text-sm text-muted-foreground">No description provided.</p>
        </Section>
      )}

      {item.title.includes("\n") ? (
        <Section title="Full title">
          <p className="whitespace-pre-wrap text-sm leading-6">{item.title}</p>
        </Section>
      ) : null}

      <Section title="Parent">
        {item.parent_id ? (
          parent ? (
            <ItemLink id={parent.canonical_id} label={displayTitle(parent)} type={parent.type} onSelect={onSelect} />
          ) : (
            <p className="text-sm text-destructive">Broken parent reference: {item.parent_id}</p>
          )
        ) : (
          <p className="text-sm text-muted-foreground">
            {item.type === "epic" ? "Root item (no parent)." : "No parent. Listed as unparented."}
          </p>
        )}
      </Section>

      {item.child_ids.length > 0 ? (
        <Section title="Children">
          <ul className="space-y-1">
            {children.map(({ id, item: child }) =>
              child ? (
                <li key={id}>
                  <ItemLink id={id} label={displayTitle(child)} type={child.type} onSelect={onSelect} />
                </li>
              ) : (
                <li key={id} className="text-sm text-destructive">
                  Missing child {id}
                </li>
              ),
            )}
          </ul>
        </Section>
      ) : item.type === "user_story" ? (
        <Section title="Children">
          <p className="text-sm text-muted-foreground">No child tasks. This can be a valid atomic story.</p>
        </Section>
      ) : null}

      {item.acceptance_criteria.length > 0 ? (
        <Section title="Acceptance criteria">
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {item.acceptance_criteria.map((criterion, index) => (
              <li key={`${index}:${criterion}`}>{criterion}</li>
            ))}
          </ul>
        </Section>
      ) : null}

      {item.test_requirements.length > 0 ? (
        <Section title="Test requirements">
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {item.test_requirements.map((requirement, index) => (
              <li key={`${index}:${requirement}`}>{requirement}</li>
            ))}
          </ul>
        </Section>
      ) : null}

      {item.dependencies.length > 0 ? (
        <Section title="Depends on">
          <ul className="space-y-1">
            {item.dependencies.map((id) => {
              const target = index.byId.get(id);
              return target ? (
                <li key={id}>
                  <ItemLink id={id} label={displayTitle(target)} type={target.type} onSelect={onSelect} />
                </li>
              ) : (
                <li key={id} className="text-sm text-destructive">
                  Missing dependency {id}
                </li>
              );
            })}
          </ul>
        </Section>
      ) : null}

      {dependents.length > 0 ? (
        <Section title="Dependents">
          <ul className="space-y-1">
            {dependents.map(({ id, item: dependent }) =>
              dependent ? (
                <li key={id}>
                  <ItemLink id={id} label={displayTitle(dependent)} type={dependent.type} onSelect={onSelect} />
                </li>
              ) : (
                <li key={id} className="text-sm text-muted-foreground">
                  {id}
                </li>
              ),
            )}
          </ul>
        </Section>
      ) : null}

      {item.provenance.length > 0 ? (
        <Section title="Provenance">
          <div className="space-y-2">
            {item.provenance.map((ref, index) => (
              <ProvenanceCard key={`${ref.json_path}:${index}`} reference={ref} />
            ))}
          </div>
        </Section>
      ) : (
        <Section title="Provenance">
          <p className="text-sm text-muted-foreground">No source references on this item.</p>
        </Section>
      )}

      {item.effort ? (
        <Section title="Effort">
          <p className="text-sm">
            {[
              item.effort.story_points != null ? `${item.effort.story_points} points` : null,
              item.effort.original_estimate_hours != null
                ? `${item.effort.original_estimate_hours}h original`
                : null,
              item.effort.remaining_hours != null ? `${item.effort.remaining_hours}h remaining` : null,
              item.effort.textual_estimate,
            ]
              .filter(Boolean)
              .join(" · ") || "Recorded with no numeric values."}
          </p>
        </Section>
      ) : null}

      {findings.length > 0 ? (
        <Section title="Findings">
          <ul className="space-y-2">
            {findings.map((finding) => (
              <li key={`f-${finding.index}`} className="rounded-md border bg-background p-2 text-sm">
                <div className="flex flex-wrap gap-1.5">
                  <SeverityBadge severity={finding.severity} />
                  <Badge variant="outline">{finding.code}</Badge>
                </div>
                <p className="mt-1">{finding.message}</p>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {issues.length > 0 ? (
        <Section title="Structural issues">
          <ul className="space-y-2">
            {issues.map((issue, idx) => (
              <li key={`${issue.code}:${idx}`} className="text-sm">
                <SeverityBadge severity={issue.severity} /> {issue.message}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {Object.keys(item.unknownFields).length > 0 ? (
        <Section title="Additional fields">
          <pre className="overflow-auto rounded-md bg-muted p-3 text-xs">
            {JSON.stringify(item.unknownFields, null, 2)}
          </pre>
        </Section>
      ) : null}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-5 space-y-2">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</h3>
      {children}
    </section>
  );
}

function ItemLink({
  id,
  label,
  type,
  onSelect,
}: {
  id: string;
  label: string;
  type: string;
  onSelect: (id: string) => void;
}) {
  return (
    <button type="button" className="text-left text-sm hover:underline" onClick={() => onSelect(id)}>
      <Badge variant="outline" className="mr-1.5">
        {formatType(type)}
      </Badge>
      {label}
    </button>
  );
}

function ProvenanceCard({ reference }: { reference: SourceReference }) {
  return (
    <article className="rounded-md border bg-background p-3 text-sm">
      <p className="font-mono text-xs text-muted-foreground">{reference.json_path || "(missing path)"}</p>
      <p className="text-xs text-muted-foreground">
        {[reference.contract_id, reference.field_name].filter(Boolean).join(" · ")}
      </p>
      {reference.excerpt ? (
        <p className="mt-2 whitespace-pre-wrap leading-5">{truncate(reference.excerpt, 600)}</p>
      ) : (
        <p className="mt-2 text-muted-foreground">No excerpt provided.</p>
      )}
    </article>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const variant =
    severity === "error" ? "critical" : severity === "warning" ? "warning" : "info";
  return <Badge variant={variant}>{severity}</Badge>;
}
