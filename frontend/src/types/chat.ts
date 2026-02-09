export interface ToolCall {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: "running" | "complete" | "error";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  intent?: string;
  intentLabel?: string;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
  timestamp: Date;
}

export type WSMessageType =
  | "connected"
  | "intent"
  | "tool_call"
  | "tool_result"
  | "token"
  | "done"
  | "error";

export interface WSMessage {
  type: WSMessageType;
  [key: string]: unknown;
}
