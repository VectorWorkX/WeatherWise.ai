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
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="flex justify-start"
    >
      <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl border border-amber-400/60 bg-amber-50/80 backdrop-blur-md px-5 py-4 shadow-md">
        <div className="flex items-start gap-3">
          <AlertTriangle size={20} className="mt-0.5 shrink-0 text-amber-600" />
          <div className="flex-1">
            <p className="text-sm font-medium text-amber-900">{message}</p>
            <div className="mt-3 flex gap-2">
              <button
                onClick={onApprove}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-emerald-700"
              >
                <Check size={14} />
                Approve
              </button>
              <button
                onClick={onReject}
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-300"
              >
                <X size={14} />
                Dismiss
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
