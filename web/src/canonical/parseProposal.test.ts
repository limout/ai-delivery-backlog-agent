import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { parseProposalJson, parseProposalText } from "./parseProposal";
import { buildReviewIndex } from "./tree";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

function workItem(overrides: Record<string, unknown>) {
  return {
    canonical_id: "cbl_epic",
    title: "Delivery epic",
    description: "Ship the capability.",
    status: "todo",
    approval_state: "not_approved",
    priority: null,
    parent_id: null,
    child_ids: [],
    acceptance_criteria: [],
    test_requirements: [],
    dependencies: [],
    provenance: [],
    effort: null,
    type: "epic",
    ...overrides,
  };
}

function envelope(items: unknown[], findings: unknown[] = []) {
  return {
    backlog: {
      contract_id: "canonical.backlog.v1",
      backlog_id: "cbl_backlog",
      items,
      source_contract_id: "copilot.workflow_response.v1",
      source_provenance: [],
    },
    findings: {
      contract_id: "canonical.generation_findings.v1",
      items: findings,
    },
  };
}

describe("parseProposalText", () => {
  it("rejects empty input", () => {
    const result = parseProposalText("  ", "empty.json");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors[0]).toMatch(/empty/i);
    }
  });

  it("rejects malformed JSON", () => {
    const result = parseProposalText("{not json", "bad.json");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors[0]).toMatch(/Malformed JSON/);
    }
  });

  it("rejects unsupported contract versions", () => {
    const payload = envelope([workItem({})]);
    payload.backlog.contract_id = "canonical.backlog.v0";
    const result = parseProposalJson(payload, "old.json");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/Unsupported contract version/);
    }
  });

  it("rejects missing required item fields", () => {
    const result = parseProposalJson(
      envelope([{ ...workItem({}), title: undefined, canonical_id: undefined }]),
      "missing.json",
    );
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/canonical_id/);
      expect(result.errors.join(" ")).toMatch(/title/);
    }
  });

  it("loads a valid envelope and preserves extra fields", () => {
    const result = parseProposalJson(
      {
        ...envelope([
          workItem({
            review_note: "keep me",
            child_ids: ["cbl_story"],
          }),
          workItem({
            canonical_id: "cbl_story",
            type: "user_story",
            title: "Atomic story",
            parent_id: "cbl_epic",
            child_ids: [],
          }),
        ]),
        extra_envelope: 1,
      },
      "valid.json",
    );
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.document.filename).toBe("valid.json");
    expect(result.document.items).toHaveLength(2);
    expect(result.document.items[0]?.unknownFields.review_note).toBe("keep me");
    expect(result.document.unknownFields.extra_envelope).toBe(1);
    const story = result.document.items[1];
    expect(story?.type).toBe("user_story");
    expect(story?.child_ids).toEqual([]);
  });

  it("records broken parent and dependency references without discarding items", () => {
    const result = parseProposalJson(
      envelope([
        workItem({
          canonical_id: "cbl_story",
          type: "user_story",
          parent_id: "cbl_missing_parent",
          dependencies: ["cbl_missing_dep"],
        }),
      ]),
      "broken.json",
    );
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const codes = result.document.structuralIssues.map((issue) => issue.code);
    expect(codes).toContain("BROKEN_PARENT");
    expect(codes).toContain("BROKEN_DEPENDENCY");
    expect(result.document.items).toHaveLength(1);
  });

  it("builds a large hierarchy without flattening", () => {
    const items = [workItem({ child_ids: ["cbl_f0"] })];
    for (let index = 0; index < 40; index += 1) {
      items.push(
        workItem({
          canonical_id: `cbl_f${index}`,
          type: "feature",
          title: `Capability ${index}`,
          parent_id: "cbl_epic",
          child_ids: [`cbl_s${index}`],
        }),
        workItem({
          canonical_id: `cbl_s${index}`,
          type: "user_story",
          title: `Story ${index}`,
          parent_id: `cbl_f${index}`,
          child_ids: [],
        }),
      );
    }
    items[0] = workItem({
      child_ids: Array.from({ length: 40 }, (_, index) => `cbl_f${index}`),
    });
    const result = parseProposalJson(envelope(items), "large.json");
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const tree = buildReviewIndex(result.document);
    expect(tree.roots).toHaveLength(1);
    expect(tree.roots[0]?.children).toHaveLength(40);
    expect(result.document.items).toHaveLength(81);
  });

  it("parses checked-in agent outputs", () => {
    for (const relative of [
      "output/real-backlog.json",
      "output/llm-eval-backlog-v2.json",
      "output/llm-eval-backlog-v3c.json",
    ]) {
      const text = readFileSync(resolve(repoRoot, relative), "utf8");
      const result = parseProposalText(text, relative);
      expect(result.ok, relative).toBe(true);
      if (!result.ok) {
        continue;
      }
      expect(result.document.items.length).toBeGreaterThan(0);
      expect(result.document.contract_id).toBe("canonical.backlog.v1");
    }
  });

  it("keeps an atomic story without inventing child tasks", () => {
    const text = readFileSync(resolve(repoRoot, "output/llm-eval-backlog-v3c.json"), "utf8");
    const result = parseProposalText(text, "llm-eval-backlog-v3c.json");
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const atomic = result.document.items.find(
      (item) => item.type === "user_story" && item.child_ids.length === 0,
    );
    expect(atomic).toBeTruthy();
  });
});
