import { useId, useState, type ChangeEvent, type DragEvent } from "react";
import { FileJson, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type FileDropProps = {
  disabled?: boolean;
  onFile: (file: File) => void;
};

export function FileDrop({ disabled, onFile }: FileDropProps) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) {
      onFile(file);
    }
  };

  const onDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (!disabled) {
      setDragging(true);
    }
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) {
      handleFiles(event.dataTransfer.files);
    }
  };

  return (
    <label
      htmlFor={inputId}
      onDragOver={onDragOver}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-12 text-center transition-colors",
        dragging ? "border-ring bg-accent" : "border-border bg-card hover:bg-accent/60",
        disabled && "pointer-events-none opacity-60",
      )}
    >
      <Upload className="size-8 text-muted-foreground" aria-hidden />
      <div className="space-y-1">
        <p className="text-sm font-medium">Load a canonical backlog JSON file</p>
        <p className="text-sm text-muted-foreground">
          Drop the agent output here, or choose a file. Expected shape is{" "}
          <code className="rounded bg-muted px-1 py-0.5 text-xs">
            {"{ backlog, findings }"}
          </code>
          .
        </p>
      </div>
      <Button type="button" size="sm" variant="outline">
        <FileJson className="size-4" />
        Choose file
      </Button>
      <input
        id={inputId}
        data-testid="file-input"
        type="file"
        accept="application/json,.json"
        className="sr-only"
        disabled={disabled}
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </label>
  );
}
