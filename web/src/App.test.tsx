import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import App from "@/App";

function validFile() {
  const payload = {
    backlog: {
      contract_id: "canonical.backlog.v1",
      backlog_id: "cbl_backlog",
      items: [
        {
          canonical_id: "cbl_epic",
          title: "Catalog holds",
          description: "Members search and place holds.",
          status: "todo",
          approval_state: "not_approved",
          priority: null,
          parent_id: null,
          child_ids: ["cbl_story"],
          acceptance_criteria: [],
          test_requirements: [],
          dependencies: [],
          provenance: [
            {
              contract_id: "example.contract.v1",
              json_path: "$.request",
              field_name: "request",
              excerpt: "Build catalog holds.",
            },
          ],
          effort: null,
          type: "epic",
        },
        {
          canonical_id: "cbl_story",
          title: "Place a hold",
          description: "A member reserves an available copy.",
          status: "todo",
          approval_state: "not_approved",
          priority: "high",
          parent_id: "cbl_epic",
          child_ids: [],
          acceptance_criteria: ["A hold appears on the member account."],
          test_requirements: ["Contract test for hold creation."],
          dependencies: [],
          provenance: [],
          effort: null,
          type: "user_story",
        },
      ],
      source_contract_id: "example.contract.v1",
      source_provenance: [],
    },
    findings: {
      contract_id: "canonical.generation_findings.v1",
      items: [
        {
          severity: "warning",
          code: "UNRESOLVED_PREREQUISITE",
          message: "Vendor feed access is not granted.",
          source_references: [],
          canonical_ids: ["cbl_story"],
        },
        {
          severity: "info",
          code: "ATOMIC_STORY",
          message: "User story has no child tasks.",
          source_references: [],
          canonical_ids: ["cbl_story"],
        },
      ],
    },
  };
  return new File([JSON.stringify(payload)], "sample-backlog.json", { type: "application/json" });
}

describe("App", () => {
  it("shows an empty load state", () => {
    render(<App />);
    expect(screen.getByText(/Review a canonical backlog/i)).toBeInTheDocument();
    expect(screen.getByTestId("file-input")).toBeInTheDocument();
  });

  it("shows actionable errors for malformed JSON", async () => {
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByTestId("file-input");
    const file = new File(["{nope"], "broken.json", { type: "application/json" });
    await user.upload(input, file);
    expect(await screen.findByTestId("load-error")).toHaveTextContent(/Malformed JSON/);
  });

  it("loads a backlog, shows hierarchy details, and navigates from a finding", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.upload(screen.getByTestId("file-input"), validFile());

    expect(await screen.findByText(/sample-backlog.json/)).toBeInTheDocument();
    expect(screen.getAllByText("Catalog holds").length).toBeGreaterThan(0);
    expect(screen.getByText(/User Story 1/)).toBeInTheDocument();
    expect(screen.getByText(/Unresolved prerequisites 1/)).toBeInTheDocument();

    await user.click(screen.getByTestId("select-cbl_story"));
    const detail = screen.getByTestId("item-detail");
    expect(detail).toHaveTextContent("Place a hold");
    expect(detail).toHaveTextContent("A hold appears on the member account.");
    expect(detail).toHaveTextContent("No child tasks");
    expect(detail).toHaveTextContent("UNRESOLVED_PREREQUISITE");

    await user.click(screen.getByRole("button", { name: "Findings" }));
    expect(screen.getByTestId("findings-panel")).toHaveTextContent("Vendor feed access is not granted.");
    await user.click(screen.getAllByRole("button", { name: /Open Place a hold/i })[0]!);
    expect(screen.getByTestId("item-detail")).toHaveTextContent("Place a hold");
  });

  it("filters the tree by type", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.upload(screen.getByTestId("file-input"), validFile());
    await screen.findByText(/sample-backlog.json/);
    await user.selectOptions(screen.getByTestId("type-filter"), "task");
    expect(screen.getByTestId("backlog-tree")).toHaveTextContent(/No work items match/);
  });
});
