import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Loader2, CheckCircle2, Circle } from "lucide-react";
import type { ThinkingStep } from "../../types/chat";
import { getPhaseLabel } from "../../utils/humanize";
import ToolCallStep from "./ToolCallStep";

interface Props {
  steps: ThinkingStep[];
  isStreaming?: boolean;
}

export default function ThinkingTimeline({ steps, isStreaming }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (steps.length === 0) return null;

  // Derive header label from the last active step's op
  const activeStep = [...steps].reverse().find((s) => s.status === "active");
  const allDone = !isStreaming && steps.every((s) => s.status === "done");
  const headerLabel = allDone ? "Thought process" : getPhaseLabel(activeStep?.op);

  return (
    <div className="mb-3">
      {/* Disclosure header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-[13px] text-neutral-400 hover:text-neutral-300 transition-colors"
      >
        <ChevronDown
          size={14}
          className={`transition-transform ${expanded ? "rotate-0" : "-rotate-90"}`}
        />
        <span className={allDone ? "" : "animate-pulse"}>
          {headerLabel}
        </span>
      </button>

      {/* Expanded thinking steps */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="ml-1.5 mt-2 border-l border-white/[0.06] pl-3 space-y-1.5">
              {steps.map((step) => (
                <ThinkingStepRow key={step.id} step={step} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ThinkingStepRow({ step }: { step: ThinkingStep }) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
    >
      <div className="flex items-start gap-2">
        {/* Status icon */}
        <div className="mt-0.5 shrink-0">
          {step.status === "active" ? (
            <Loader2 size={12} className="animate-spin text-neutral-400" />
          ) : step.status === "done" ? (
            <CheckCircle2 size={12} className="text-emerald-500/70" />
          ) : (
            <Circle size={12} className="text-neutral-700" />
          )}
        </div>

        {/* Label + optional details toggle */}
        <div className="flex-1 min-w-0">
          <p
            className={`text-[13px] leading-relaxed ${
              step.status === "pending"
                ? "text-neutral-600"
                : step.status === "active"
                  ? "text-neutral-400"
                  : "text-neutral-500"
            }`}
          >
            {step.label}
          </p>

          {/* Tool call details toggle */}
          {step.toolCall && (
            <button
              onClick={() => setDetailsOpen(!detailsOpen)}
              className="mt-0.5 text-[11px] text-neutral-600 hover:text-neutral-400 transition-colors"
            >
              {detailsOpen ? "hide details" : "show details"}
            </button>
          )}

          <AnimatePresence>
            {detailsOpen && step.toolCall && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.15 }}
                className="mt-1 overflow-hidden"
              >
                <ToolCallStep toolCall={step.toolCall} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}
