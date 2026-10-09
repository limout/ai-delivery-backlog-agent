import { WORK_ITEM_TYPES, type WORK_ITEM_TYPES as WorkTypes } from "./constants";
import type { CanonicalProposal, CanonicalWorkItem, Finding, StructuralIssue } from "./types";

export type TreeNode = {
  item: CanonicalWorkItem;
  children: TreeNode[];
  depth: number;
};

export type ReviewIndex = {
  byId: Map<string, CanonicalWorkItem>;
  roots: TreeNode[];
  orphans: TreeNode[];
  missingChildren: { parentId: string; childId: string }[];
  dependents: Map<string, string[]>;
  findingsByItem: Map<string, Finding[]>;
  issuesByItem: Map<string, StructuralIssue[]>;
};

export function buildReviewIndex(document: CanonicalProposal): ReviewIndex {
  const byId = new Map(document.items.map((item) => [item.canonical_id, item]));
  const placed = new Set<string>();
  const missingChildren: { parentId: string; childId: string }[] = [];

  const makeNode = (item: CanonicalWorkItem, depth: number): TreeNode => {
    placed.add(item.canonical_id);
    const children: TreeNode[] = [];
    for (const childId of item.child_ids) {
      const child = byId.get(childId);
      if (!child) {
        missingChildren.push({ parentId: item.canonical_id, childId });
        continue;
      }
      children.push(makeNode(child, depth + 1));
    }
    return { item, children, depth };
  };

  const roots: TreeNode[] = [];
  for (const item of document.items) {
    if (item.parent_id && byId.has(item.parent_id)) {
      continue;
    }
    if (placed.has(item.canonical_id)) {
      continue;
    }
    roots.push(makeNode(item, 0));
  }

  const orphans: TreeNode[] = [];
  for (const item of document.items) {
    if (placed.has(item.canonical_id)) {
      continue;
    }
    orphans.push(makeNode(item, 0));
  }

  const dependents = new Map<string, string[]>();
  for (const item of document.items) {
    for (const dependencyId of item.dependencies) {
      const list = dependents.get(dependencyId) ?? [];
      list.push(item.canonical_id);
      dependents.set(dependencyId, list);
    }
  }

  const findingsByItem = new Map<string, Finding[]>();
  for (const finding of document.findings) {
    for (const id of finding.canonical_ids) {
      const list = findingsByItem.get(id) ?? [];
      list.push(finding);
      findingsByItem.set(id, list);
    }
  }

  const issuesByItem = new Map<string, StructuralIssue[]>();
  for (const issue of document.structuralIssues) {
    if (!issue.canonicalId) {
      continue;
    }
    const list = issuesByItem.get(issue.canonicalId) ?? [];
    list.push(issue);
    issuesByItem.set(issue.canonicalId, list);
  }

  return { byId, roots, orphans, missingChildren, dependents, findingsByItem, issuesByItem };
}

export function typeCounts(items: CanonicalWorkItem[]): { type: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const type of WORK_ITEM_TYPES) {
    counts.set(type, 0);
  }
  for (const item of items) {
    counts.set(item.type, (counts.get(item.type) ?? 0) + 1);
  }
  return [...counts.entries()].map(([type, count]) => ({ type, count }));
}

export function displayTitle(item: CanonicalWorkItem): string {
  const firstLine = item.title.split(/\r?\n/, 1)[0]?.trim() ?? item.title;
  if (firstLine.length <= 120) {
    return firstLine || item.title;
  }
  return `${firstLine.slice(0, 117)}…`;
}

export function isKnownType(type: string): type is (typeof WorkTypes)[number] {
  return (WORK_ITEM_TYPES as readonly string[]).includes(type);
}

export function ancestorsOf(itemId: string, byId: Map<string, CanonicalWorkItem>): string[] {
  const chain: string[] = [];
  const seen = new Set<string>();
  let current = byId.get(itemId);
  while (current?.parent_id && !seen.has(current.parent_id)) {
    seen.add(current.parent_id);
    chain.push(current.parent_id);
    current = byId.get(current.parent_id);
  }
  return chain;
}

export function nodeMatchesQuery(node: TreeNode, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    node.item.title,
    node.item.description,
    node.item.type,
    node.item.canonical_id,
    ...node.item.acceptance_criteria,
  ]
    .join("\n")
    .toLowerCase();
  return haystack.includes(query);
}

export function filterTree(nodes: TreeNode[], query: string, typeFilter: string | "all"): TreeNode[] {
  const normalized = query.trim().toLowerCase();
  const walk = (node: TreeNode): TreeNode | null => {
    const children = node.children
      .map(walk)
      .filter((child): child is TreeNode => child !== null);
    const selfMatch =
      nodeMatchesQuery(node, normalized) &&
      (typeFilter === "all" || node.item.type === typeFilter);
    if (selfMatch || children.length > 0) {
      return { ...node, children };
    }
    return null;
  };
  return nodes.map(walk).filter((node): node is TreeNode => node !== null);
}
