import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wrench, ChevronDown, Loader2, CheckCircle2 } from "lucide-react";
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
      className="mb-2 overflow-hidden rounded-xl border-l-3 border-emerald-300 glass-light"
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition hover:bg-slate-100/60"
      >
        {isRunning ? (
          <Loader2 size={14} className="animate-spin text-emerald-400" />
        ) : (
          <CheckCircle2 size={14} className="text-emerald-400" />
        )}
        <Wrench size={14} className="text-slate-400" />
        <span className="flex-1 text-sm font-medium text-slate-600">
          {toolCall.name}
        </span>
        <ChevronDown
          size={14}
          className={`text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {/* Expanded details */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-slate-200/60 px-3 py-2.5"
          >
            {/* Args */}
            {Object.keys(toolCall.args).length > 0 && (
              <div className="mb-2">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Input
                </p>
                <div className="rounded-lg bg-slate-100/80 px-3 py-2 font-mono text-xs text-slate-500">
                  {Object.entries(toolCall.args).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-teal-600">{k}:</span>{" "}
                      {String(v).slice(0, 200)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Result */}
            {toolCall.result && (
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Output
                </p>
                <div className="max-h-40 overflow-auto rounded-lg bg-slate-100/80 px-3 py-2 font-mono text-xs text-slate-500">
                  {toolCall.result.slice(0, 800)}
                  {toolCall.result.length > 800 && (
                    <span className="text-slate-600"> ... truncated</span>
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
