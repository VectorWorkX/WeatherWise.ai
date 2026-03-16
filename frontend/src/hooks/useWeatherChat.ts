import { useState, useRef, useCallback, useEffect } from "react";
import type { ChatMessage, ToolCall, VisualReportData, WSMessage } from "../types/chat";
import { WS_URL } from "../config/constants";

export function useWeatherChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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
          // Find the last assistant message (still streaming) and set its intent
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              updated[i] = {
                ...updated[i],
                intent: data.intent as string,
                intentLabel: data.label as string,
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "tool_call":
        setMessages((prev) => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              const toolCalls = updated[i].toolCalls || [];
              updated[i] = {
                ...updated[i],
                toolCalls: [
                  ...toolCalls,
                  {
                    callId: data.call_id as string,
                    name: data.name as string,
                    args: (data.args as Record<string, unknown>) || {},
                    status: "running",
                  },
                ],
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "tool_result":
        setMessages((prev) => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              const toolCalls = (updated[i].toolCalls || []).map((tc: ToolCall) =>
                tc.callId === (data.call_id as string)
                  ? { ...tc, result: data.content as string, status: "complete" as const }
                  : tc,
              );
              updated[i] = { ...updated[i], toolCalls };
              break;
            }
          }
          return updated;
        });
        break;

      case "token":
        setMessages((prev) => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              updated[i] = {
                ...updated[i],
                content: updated[i].content + (data.content as string),
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "approval_required":
        setMessages((prev) => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              updated[i] = {
                ...updated[i],
                pendingApproval: true,
                approvalMessage: data.message as string,
              };
              break;
            }
          }
          return updated;
        });
        break;

      case "done":
        setMessages((prev) => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              updated[i] = {
                ...updated[i],
                content: (data.full_content as string) || updated[i].content,
                isStreaming: false,
              };
              break;
            }
          }
          return updated;
        });
        setIsLoading(false);
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
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "assistant" && updated[i].isStreaming) {
              updated[i] = {
                ...updated[i],
                content: `**Error:** ${data.message as string}`,
                isStreaming: false,
              };
              break;
            }
          }
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
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsLoading(true);

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
