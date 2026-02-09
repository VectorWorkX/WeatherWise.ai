import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../../types/chat";
import IntentBadge from "./IntentBadge";
import ToolCallStep from "./ToolCallStep";
import TypingIndicator from "./TypingIndicator";

interface Props {
  message: ChatMessage;
}

export default function MessageBubble({ message }: Props) {
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

        {/* Message bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "rounded-br-sm bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/15"
              : "rounded-bl-sm glass-light text-slate-700"
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
