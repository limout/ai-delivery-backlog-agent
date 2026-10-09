import { useMemo, useState } from "react";

import { BacklogTree } from "@/components/BacklogTree";
import { FileDrop } from "@/components/FileDrop";
import { FindingsPanel } from "@/components/FindingsPanel";
import { ItemDetail } from "@/components/ItemDetail";
import { ReviewHeader } from "@/components/ReviewHeader";
import { parseProposalText } from "@/canonical/parseProposal";
import type { FindingCategory } from "@/canonical/summary";
import { ancestorsOf, buildReviewIndex, filterTree } from "@/canonical/tree";
import type { CanonicalProposal } from "@/canonical/types";
import { WORK_ITEM_TYPES } from "@/canonical/constants";
import { formatType } from "@/lib/utils";

type AppState =
  | { status: "empty" }
  | { status: "loading"; filename: string }
  | { status: "error"; filename?: string; errors: string[] }
  | { status: "ready"; document: CanonicalProposal };

export default function App() {
  const [state, setState] = useState<AppState>({ status: "empty" });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<string | "all">("all");
  const [findingSeverity, setFindingSeverity] = useState<string | "all">("all");
  const [findingCategory, setFindingCategory] = useState<FindingCategory>("all");
  const [findingQuery, setFindingQuery] = useState("");
  const [mainTab, setMainTab] = useState<"tree" | "findings">("tree");

  const index = useMemo(
    () => (state.status === "ready" ? buildReviewIndex(state.document) : null),
    [state],
  );

  const filteredRoots = useMemo(() => {
    if (!index) {
      return [];
    }
    return filterTree(index.roots, query, typeFilter);
  }, [index, query, typeFilter]);

  const filteredOrphans = useMemo(() => {
    if (!index) {
      return [];
    }
    return filterTree(index.orphans, query, typeFilter);
  }, [index, query, typeFilter]);

  const selectedItem =
    state.status === "ready" && selectedId && index ? (index.byId.get(selectedId) ?? null) : null;

  const loadFile = async (file: File) => {
    setState({ status: "loading", filename: file.name });
    try {
      const text = await readFileAsText(file);
      const result = parseProposalText(text, file.name);
      if (!result.ok) {
        setState({ status: "error", filename: file.name, errors: result.errors });
        return;
      }
      const next = result.document;
      const nextIndex = buildReviewIndex(next);
      const initialExpanded = new Set<string>();
      for (const root of nextIndex.roots) {
        initialExpanded.add(root.item.canonical_id);
      }
      setExpanded(initialExpanded);
      setSelectedId(next.items[0]?.canonical_id ?? null);
      setQuery("");
      setTypeFilter("all");
      setFindingSeverity("all");
      setFindingCategory("all");
      setFindingQuery("");
      setMainTab("tree");
      setState({ status: "ready", document: next });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not read the file.";
      setState({ status: "error", filename: file.name, errors: [message] });
    }
  };

  const selectItem = (id: string) => {
    if (!index) {
      return;
    }
    setSelectedId(id);
    setMainTab("tree");
    setExpanded((current) => {
      const next = new Set(current);
      next.add(id);
      for (const ancestor of ancestorsOf(id, index.byId)) {
        next.add(ancestor);
      }
      return next;
    });
  };

  const toggleExpand = (id: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      {state.status === "empty" || state.status === "error" ? (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-6 py-16">
          <div>
            <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
              AI Delivery Backlog Agent
            </p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight">Review a canonical backlog</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Load JSON produced by the agent. This UI inspects canonical items and findings; it does
              not generate a backlog from source documents.
            </p>
          </div>
          <FileDrop onFile={loadFile} />
          {state.status === "error" ? (
            <div
              className="rounded-lg border border-destructive/40 bg-[var(--status-critical-bg)] p-4 text-sm"
              data-testid="load-error"
            >
              <p className="font-medium">Could not load {state.filename ?? "the file"}.</p>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {state.errors.map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}

      {state.status === "loading" ? (
        <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
          Reading {state.filename}…
        </div>
      ) : null}

      {state.status === "ready" && index ? (
        <>
          <ReviewHeader document={state.document} onNewFile={() => setState({ status: "empty" })} />
          <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
            <section className="flex min-h-0 min-w-0 flex-1 flex-col border-b lg:border-r lg:border-b-0">
              <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
                <div className="flex rounded-md border p-0.5 text-xs">
                  <TabButton active={mainTab === "tree"} onClick={() => setMainTab("tree")}>
                    Hierarchy
                  </TabButton>
                  <TabButton active={mainTab === "findings"} onClick={() => setMainTab("findings")}>
                    Findings
                  </TabButton>
                </div>
                {mainTab === "tree" ? (
                  <>
                    <input
                      data-testid="tree-search"
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Search items"
                      className="h-8 min-w-[12rem] flex-1 rounded-md border bg-background px-3 text-sm"
                    />
                    <select
                      data-testid="type-filter"
                      value={typeFilter}
                      onChange={(event) => setTypeFilter(event.target.value)}
                      className="h-8 rounded-md border bg-background px-2 text-sm"
                    >
                      <option value="all">All types</option>
                      {WORK_ITEM_TYPES.map((type) => (
                        <option key={type} value={type}>
                          {formatType(type)}
                        </option>
                      ))}
                    </select>
                  </>
                ) : (
                  <>
                    <input
                      data-testid="finding-search"
                      value={findingQuery}
                      onChange={(event) => setFindingQuery(event.target.value)}
                      placeholder="Search findings"
                      className="h-8 min-w-[10rem] flex-1 rounded-md border bg-background px-3 text-sm"
                    />
                    <select
                      data-testid="finding-severity"
                      value={findingSeverity}
                      onChange={(event) => setFindingSeverity(event.target.value)}
                      className="h-8 rounded-md border bg-background px-2 text-sm"
                    >
                      <option value="all">All severities</option>
                      <option value="error">Error</option>
                      <option value="warning">Warning</option>
                      <option value="info">Info</option>
                    </select>
                    <select
                      data-testid="finding-category"
                      value={findingCategory}
                      onChange={(event) => setFindingCategory(event.target.value as FindingCategory)}
                      className="h-8 rounded-md border bg-background px-2 text-sm"
                    >
                      <option value="all">All categories</option>
                      <option value="error">Errors</option>
                      <option value="warning">Warnings</option>
                      <option value="uncovered">Uncovered requirements</option>
                      <option value="prerequisites">Prerequisites</option>
                      <option value="planning">Planning constraints</option>
                      <option value="generated">Generated content</option>
                      <option value="atomic">Atomic stories</option>
                      <option value="other">Other</option>
                    </select>
                  </>
                )}
              </div>
              <div className="min-h-0 flex-1">
                {mainTab === "tree" ? (
                  <BacklogTree
                    nodes={filteredRoots}
                    orphans={filteredOrphans}
                    missingChildren={index.missingChildren}
                    index={index}
                    selectedId={selectedId}
                    expanded={expanded}
                    onSelect={selectItem}
                    onToggle={toggleExpand}
                  />
                ) : (
                  <FindingsPanel
                    document={state.document}
                    index={index}
                    severity={findingSeverity}
                    category={findingCategory}
                    query={findingQuery}
                    onSelectItem={selectItem}
                  />
                )}
              </div>
            </section>
            <section className="min-h-0 min-w-0 flex-1 bg-card">
              <ItemDetail item={selectedItem} index={index} onSelect={selectItem} />
            </section>
          </div>
        </>
      ) : null}
    </div>
  );
}

function readFileAsText(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the file."));
    reader.readAsText(file);
  });
}

function TabButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded px-2.5 py-1 ${active ? "bg-accent font-medium" : "text-muted-foreground"}`}
    >
      {children}
    </button>
  );
}
