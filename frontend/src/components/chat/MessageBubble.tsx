import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../../types/chat";
import ApprovalPrompt from "./ApprovalPrompt";
import IntentBadge from "./IntentBadge";
import ToolCallStep from "./ToolCallStep";
import TypingIndicator from "./TypingIndicator";

interface Props {
  message: ChatMessage;
  onApproval?: (approved: boolean) => void;
}

export default function MessageBubble({ message, onApproval }: Props) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div className={`max-w-[85%] sm:max-w-[75%] ${isUser ? "" : "w-full sm:w-auto"}`}>
        {/* Intent badge */}
        {!isUser && message.intentLabel && (
          <div className="mb-2">
            <IntentBadge
              intent={message.intent || ""}
              label={message.intentLabel}
            />
          </div>
        )}

        {/* Tool call steps */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mb-2">
            {message.toolCalls.map((tc) => (
              <ToolCallStep key={tc.callId} toolCall={tc} />
            ))}
          </div>
        )}

        {/* Safety approval prompt */}
        {!isUser && message.pendingApproval && onApproval && (
          <div className="mb-2">
            <ApprovalPrompt
              message={message.approvalMessage || "Approve this response?"}
              onApprove={() => onApproval(true)}
              onReject={() => onApproval(false)}
            />
          </div>
        )}

        {/* Message bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "rounded-br-sm bg-red-100/80 backdrop-blur-md border border-red-300/50 text-slate-800 shadow-sm"
              : "rounded-bl-sm bg-violet-100/80 backdrop-blur-md border border-violet-300/50 text-slate-700 shadow-sm"
          }`}
        >
          {!isUser && message.isStreaming && !message.content ? (
            <TypingIndicator />
          ) : isUser ? (
            <p className="text-sm leading-relaxed">{message.content}</p>
          ) : (
            <div className="message-markdown text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <p
          className={`mt-1 text-[10px] text-slate-400 ${
            isUser ? "text-right" : "text-left"
          }`}
        >
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </motion.div>
  );
}
