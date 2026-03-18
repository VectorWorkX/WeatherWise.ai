export interface ToolCall {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: "running" | "complete" | "error";
}

export interface ThinkingStep {
  id: string;
  label: string;
  status: "pending" | "active" | "done";
  op?: string;
  toolCall?: ToolCall;
}

export interface VisualReportData {
  title?: string;
  subtitle?: string;
  metrics?: { label: string; value: string; unit?: string; detail?: string }[];
  sections?: {
    heading: string;
    narrative: string;
    priority: number;
    echarts_option: Record<string, unknown>;
  }[];
  insights?: string[];
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  intent?: string;
  intentLabel?: string;
  toolCalls?: ToolCall[];
  thinkingSteps?: ThinkingStep[];
  visualReport?: VisualReportData;
  suggestions?: string[];
  isStreaming?: boolean;
  pendingApproval?: boolean;
  approvalMessage?: string;
  timestamp: Date;
}

export type WSMessageType =
  | "connected"
  | "intent"
  | "status"
  | "tool_call"
  | "tool_result"
  | "token"
  | "approval_required"
  | "done"
  | "suggestions"
  | "visual_report"
  | "error";

export interface WSMessage {
  type: WSMessageType;
  [key: string]: unknown;
}
