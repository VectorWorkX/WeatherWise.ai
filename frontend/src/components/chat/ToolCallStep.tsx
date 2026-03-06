import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Loader2, CheckCircle2 } from "lucide-react";
import type { ToolCall } from "../../types/chat";

interface Props {
  toolCall: ToolCall;
}

export default function ToolCallStep({ toolCall }: Props) {
  const [expanded, setExpanded] = useState(false);
  const isRunning = toolCall.status === "running";

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      className="mb-1.5 overflow-hidden rounded-lg border border-white/[0.06] bg-white/[0.02]"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition hover:bg-white/[0.02]"
      >
        {isRunning ? (
          <Loader2 size={13} className="animate-spin text-neutral-400" />
        ) : (
          <CheckCircle2 size={13} className="text-emerald-500" />
        )}
        <span className="flex-1 text-[13px] text-neutral-500">
          {toolCall.name}
        </span>
        <ChevronDown
          size={13}
          className={`text-neutral-600 transition-transform ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-white/[0.06] px-3 py-2"
          >
            {Object.keys(toolCall.args).length > 0 && (
              <div className="mb-2">
                <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-neutral-600">
                  Input
                </p>
                <div className="rounded-md bg-[#111] px-3 py-2 font-mono text-xs text-neutral-400">
                  {Object.entries(toolCall.args).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-neutral-500">{k}:</span>{" "}
                      {String(v).slice(0, 200)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {toolCall.result && (
              <div>
                <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-neutral-600">
                  Output
                </p>
                <div className="max-h-40 overflow-auto rounded-md bg-[#111] px-3 py-2 font-mono text-xs text-neutral-400">
                  {toolCall.result.slice(0, 800)}
                  {toolCall.result.length > 800 && (
                    <span className="text-neutral-600"> ... truncated</span>
                  )}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
