/**
 * Minimal fallback for tool call labels when no plan description is available.
 * Just cleans up the raw tool name — no hardcoded sentence templates.
 */
export function humanizeToolName(name: string): string {
  return name
    .replace(/^xweather_get_/, "")
    .replace(/_/g, " ");
}

const PHASE_LABELS: Record<string, string> = {
  plan: "Thinking...",
  fetch: "Searching...",
  research: "Searching...",
  calculate: "Analyzing...",
  visualize: "Creating visuals...",
  synthesize: "Writing...",
};

/**
 * Derive the header label from the current active step's op.
 */
export function getPhaseLabel(op: string | undefined): string {
  if (!op) return "Thinking...";
  return PHASE_LABELS[op] || "Thinking...";
}
