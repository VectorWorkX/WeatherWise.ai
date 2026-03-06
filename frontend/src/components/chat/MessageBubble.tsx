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
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`${isUser ? "flex justify-end" : ""}`}
    >
      {isUser ? (
        <div className="max-w-[80%]">
          <div className="rounded-2xl rounded-br-md bg-white/[0.08] px-4 py-3">
            <p className="text-[15px] leading-relaxed text-neutral-200">{message.content}</p>
          </div>
        </div>
      ) : (
        <div className="max-w-full">
          {message.intentLabel && (
            <div className="mb-2">
              <IntentBadge intent={message.intent || ""} label={message.intentLabel} />
            </div>
          )}

          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mb-2">
              {message.toolCalls.map((tc) => (
                <ToolCallStep key={tc.callId} toolCall={tc} />
              ))}
            </div>
          )}

          {message.pendingApproval && onApproval && (
            <div className="mb-2">
              <ApprovalPrompt
                message={message.approvalMessage || "Approve this response?"}
                onApprove={() => onApproval(true)}
                onReject={() => onApproval(false)}
              />
            </div>
          )}

          <div>
            {message.isStreaming && !message.content ? (
              <TypingIndicator />
            ) : (
              <div className="message-markdown text-[15px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
