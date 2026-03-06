import { motion } from "framer-motion";
import { AlertTriangle, Check, X } from "lucide-react";

interface Props {
  message: string;
  onApprove: () => void;
  onReject: () => void;
}

export default function ApprovalPrompt({ message, onApprove, onReject }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/[0.06] px-4 py-3">
        <div className="flex items-start gap-3">
          <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-400" />
          <div className="flex-1">
            <p className="text-sm text-neutral-300">{message}</p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={onApprove}
                className="inline-flex items-center gap-1.5 rounded-lg bg-white/[0.1] px-3 py-1.5 text-xs font-medium text-neutral-200 transition hover:bg-white/[0.15]"
              >
                <Check size={13} />
                Approve
              </button>
              <button
                onClick={onReject}
                className="inline-flex items-center gap-1.5 rounded-lg bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-neutral-400 transition hover:bg-white/[0.08]"
              >
                <X size={13} />
                Dismiss
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
