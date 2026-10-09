import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

const variants = {
  default: "bg-primary text-primary-foreground",
  secondary: "bg-secondary text-secondary-foreground",
  outline: "border-border text-foreground",
  healthy: "border-transparent bg-[var(--status-healthy-bg)] text-[var(--status-healthy)]",
  info: "border-transparent bg-[var(--status-info-bg)] text-[var(--status-info)]",
  warning: "border-transparent bg-[var(--status-warning-bg)] text-[var(--status-warning)]",
  critical: "border-transparent bg-[var(--status-critical-bg)] text-[var(--status-critical)]",
  neutral: "border-transparent bg-[var(--status-neutral-bg)] text-[var(--status-neutral)]",
} as const;

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  variant?: keyof typeof variants;
};

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex w-fit shrink-0 items-center rounded-full border border-transparent px-2 py-0.5 text-[11px] font-medium whitespace-nowrap",
        variants[variant],
        className,
      )}
      {...props}
    />
  );
}
