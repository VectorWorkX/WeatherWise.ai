import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../../types/chat";
import ApprovalPrompt from "./ApprovalPrompt";
import IntentBadge from "./IntentBadge";
import ThinkingTimeline from "./ThinkingTimeline";
import TypingIndicator from "./TypingIndicator";
import WeatherReport from "./WeatherReport";

/** Strip markdown image references that don't point to real URLs.
 *  LLM sometimes generates ![title](sandboxed.png) or ![title]() for
 *  charts that are rendered separately — these show as broken images. */
function stripBrokenImages(text: string): string {
  // Remove ![alt](src) where src is NOT an http/https URL
  return text.replace(/!\[[^\]]*\]\((?!https?:\/\/)[^)]*\)/g, "");
}

interface Props {
  message: ChatMessage;
  onApproval?: (approved: boolean) => void;
  onSuggestionClick?: (suggestion: string) => void;
}

export default function MessageBubble({ message, onApproval, onSuggestionClick }: Props) {
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

          {message.thinkingSteps && message.thinkingSteps.length > 0 && (
            <ThinkingTimeline steps={message.thinkingSteps} isStreaming={message.isStreaming} />
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
            ) : message.isStreaming ? (
              /* Raw text during streaming — avoids ReactMarkdown re-parsing
                 on every token. Markdown is applied once streaming completes. */
              <div className="message-markdown text-[15px]">
                <pre className="whitespace-pre-wrap font-[inherit] text-inherit leading-relaxed m-0 p-0 bg-transparent border-0">
                  {stripBrokenImages(message.content)}
                </pre>
              </div>
            ) : (
              <div className="message-markdown text-[15px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {stripBrokenImages(message.content)}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* Visual Report (charts, metrics, insights) */}
          {message.visualReport && message.visualReport.sections && message.visualReport.sections.length > 0 && (
            <WeatherReport report={message.visualReport} />
          )}

          {/* Follow-up suggestion chips */}
          {message.suggestions && message.suggestions.length > 0 && !message.isStreaming && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
              className="mt-4 flex flex-wrap gap-2"
            >
              {message.suggestions.map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => onSuggestionClick?.(suggestion)}
                  className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-neutral-400 transition-colors hover:bg-white/[0.06] hover:text-neutral-200 hover:border-white/[0.12]"
                >
                  {suggestion}
                </button>
              ))}
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
}
