import { useState, useRef, useCallback, useEffect } from "react";
import type { ChatMessage, ThinkingStep, ToolCall, VisualReportData, WSMessage } from "../types/chat";
import { WS_URL } from "../config/constants";
import { humanizeToolName } from "../utils/humanize";

// Helper: find last streaming assistant message index
function findStreamingIdx(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant" && messages[i].isStreaming) return i;
  }
  return -1;
}

// Helper: add a thinking step to the streaming assistant message
function addThinkingStep(
  messages: ChatMessage[],
  step: ThinkingStep,
): ChatMessage[] {
  const updated = [...messages];
  const idx = findStreamingIdx(updated);
  if (idx === -1) return updated;
  const steps = updated[idx].thinkingSteps || [];
  updated[idx] = { ...updated[idx], thinkingSteps: [...steps, step] };
  return updated;
}

// Helper: update a thinking step by id
function updateThinkingStep(
  messages: ChatMessage[],
  stepId: string,
  patch: Partial<ThinkingStep>,
): ChatMessage[] {
  const updated = [...messages];
  const idx = findStreamingIdx(updated);
  if (idx === -1) return updated;
  const steps = (updated[idx].thinkingSteps || []).map((s) =>
    s.id === stepId ? { ...s, ...patch } : s,
  );
  updated[idx] = { ...updated[idx], thinkingSteps: steps };
  return updated;
}

export function useWeatherChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track pending plan steps that haven't been matched to tool calls yet
  const pendingPlanStepIds = useRef<string[]>([]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 3s
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      const data: WSMessage = JSON.parse(event.data);
      handleWSMessage(data);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const handleWSMessage = useCallback((data: WSMessage) => {
    switch (data.type) {
      case "connected":
        break;

      case "intent":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;
          // Set intent on the message
          updated[idx] = {
            ...updated[idx],
            intent: data.intent as string,
            intentLabel: data.label as string,
          };
          // Add reasoning as a thinking step
          const reasoning = data.reasoning as string;
          if (reasoning) {
            const steps = updated[idx].thinkingSteps || [];
            updated[idx] = {
              ...updated[idx],
              thinkingSteps: [
                ...steps,
                {
                  id: `intent-${Date.now()}`,
                  label: reasoning,
                  status: "done",
                  op: "intent",
                },
              ],
            };
          }
          return updated;
        });
        break;

      case "status": {
        const step = data.step as string;

        if (step === "planning") {
          setMessages((prev) =>
            addThinkingStep(prev, {
              id: "planning",
              label: "Planning approach...",
              status: "active",
              op: "plan",
            }),
          );
        } else if (step === "plan_ready") {
          setMessages((prev) => {
            // Mark planning step done
            let updated = updateThinkingStep(prev, "planning", { status: "done" });
            const idx = findStreamingIdx(updated);
            if (idx === -1) return updated;

            const existingSteps = updated[idx].thinkingSteps || [];
            const newSteps: ThinkingStep[] = [];

            // Add plan summary
            const summary = data.summary as string;
            if (summary) {
              newSteps.push({
                id: `summary-${Date.now()}`,
                label: summary,
                status: "done",
                op: "plan",
              });
            }

            // Add pending steps for each task
            const tasks = (data.tasks as { op: string; query: string; location: string; description: string }[]) || [];
            const pendingIds: string[] = [];
            tasks.forEach((task, i) => {
              const label = task.description || task.query || `${task.op} ${task.location}`.trim();
              const stepId = `task-${i}-${Date.now()}`;
              pendingIds.push(stepId);
              newSteps.push({
                id: stepId,
                label,
                status: "pending",
                op: task.op,
              });
            });
            pendingPlanStepIds.current = pendingIds;

            updated[idx] = {
              ...updated[idx],
              thinkingSteps: [...existingSteps, ...newSteps],
            };
            return updated;
          });
        } else if (step === "synthesizing") {
          setMessages((prev) =>
            addThinkingStep(prev, {
              id: `synthesizing-${Date.now()}`,
              label: "Putting it all together...",
              status: "active",
              op: "synthesize",
            }),
          );
        }
        break;
      }

      case "tool_call":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;

          const toolCall: ToolCall = {
            callId: data.call_id as string,
            name: data.name as string,
            args: (data.args as Record<string, unknown>) || {},
            status: "running",
          };

          // Add to toolCalls array (backward compat)
          const toolCalls = [...(updated[idx].toolCalls || []), toolCall];

          // Activate the next pending thinking step, or create a new one
          const steps = [...(updated[idx].thinkingSteps || [])];
          const pendingIds = pendingPlanStepIds.current;
          if (pendingIds.length > 0) {
            const nextId = pendingIds.shift()!;
            const stepIdx = steps.findIndex((s) => s.id === nextId);
            if (stepIdx !== -1) {
              steps[stepIdx] = { ...steps[stepIdx], status: "active", toolCall };
            }
          } else {
            // No plan step to match — create a new step from tool name
            steps.push({
              id: `tool-${toolCall.callId}`,
              label: humanizeToolName(toolCall.name),
              status: "active",
              op: "fetch",
              toolCall,
            });
          }

          updated[idx] = { ...updated[idx], toolCalls, thinkingSteps: steps };
          return updated;
        });
        break;

      case "tool_result":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;

          const callId = data.call_id as string;
          const resultContent = data.content as string;

          // Update toolCalls
          const toolCalls = (updated[idx].toolCalls || []).map((tc: ToolCall) =>
            tc.callId === callId
              ? { ...tc, result: resultContent, status: "complete" as const }
              : tc,
          );

          // Mark matching thinking step done
          const steps = (updated[idx].thinkingSteps || []).map((s) => {
            if (s.toolCall?.callId === callId) {
              return {
                ...s,
                status: "done" as const,
                toolCall: { ...s.toolCall, result: resultContent, status: "complete" as const },
              };
            }
            return s;
          });

          updated[idx] = { ...updated[idx], toolCalls, thinkingSteps: steps };
          return updated;
        });
        break;

      case "token":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;
          updated[idx] = {
            ...updated[idx],
            content: updated[idx].content + (data.content as string),
          };
          return updated;
        });
        break;

      case "approval_required":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;
          updated[idx] = {
            ...updated[idx],
            pendingApproval: true,
            approvalMessage: data.message as string,
          };
          return updated;
        });
        break;

      case "done":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;

          // Mark all thinking steps done
          const steps = (updated[idx].thinkingSteps || []).map((s) =>
            s.status !== "done" ? { ...s, status: "done" as const } : s,
          );

          updated[idx] = {
            ...updated[idx],
            content: (data.full_content as string) || updated[idx].content,
            isStreaming: false,
            thinkingSteps: steps,
          };
          // Reset pending plan step tracker
          pendingPlanStepIds.current = [];
          return updated;
        });
        setIsLoading(false);
        break;

      case "suggestions":
        setMessages((prev) => {
          const updated = [...prev];
          // Attach suggestions to the last assistant message
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant") {
              updated[i] = {
                ...updated[i],
                suggestions: data.suggestions as string[],
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "visual_report":
        setMessages((prev) => {
          const updated = [...prev];
          // Attach visual report to the last assistant message
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant") {
              updated[i] = {
                ...updated[i],
                visualReport: data.data as VisualReportData,
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "error":
        setMessages((prev) => {
          const updated = [...prev];
          const idx = findStreamingIdx(updated);
          if (idx === -1) return updated;
          updated[idx] = {
            ...updated[idx],
            content: `**Error:** ${data.message as string}`,
            isStreaming: false,
          };
          return updated;
        });
        setIsLoading(false);
        break;
    }
  }, []);

  const sendMessage = useCallback(
    (content: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (!content.trim()) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: new Date(),
      };

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        isStreaming: true,
        toolCalls: [],
        thinkingSteps: [],
        timestamp: new Date(),
      };

      setMessages((prev) => {
        const updated = [...prev, userMsg, assistantMsg];
        // Prevent unbounded growth — keep last 80 messages if over 100
        return updated.length > 100 ? updated.slice(-80) : updated;
      });
      setIsLoading(true);

      // Reset pending plan step tracker for new query
      pendingPlanStepIds.current = [];

      wsRef.current.send(
        JSON.stringify({ type: "user_message", content }),
      );
    },
    [],
  );

  const sendApproval = useCallback(
    (approved: boolean) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      // Clear the pending approval flag on the message
      setMessages((prev) => {
        const updated = [...prev];
        for (let i = updated.length - 1; i >= 0; i--) {
          if (updated[i].role === "assistant" && updated[i].pendingApproval) {
            updated[i] = {
              ...updated[i],
              pendingApproval: false,
              approvalMessage: undefined,
            };
            if (!approved) {
              updated[i] = {
                ...updated[i],
                content: "Response was not approved.",
                isStreaming: false,
              };
            }
            break;
          }
        }
        return updated;
      });

      if (!approved) {
        setIsLoading(false);
      }

      wsRef.current.send(
        JSON.stringify({ type: "user_approval", approved }),
      );
    },
    [],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, sendMessage, sendApproval, clearMessages, isConnected, isLoading };
}
