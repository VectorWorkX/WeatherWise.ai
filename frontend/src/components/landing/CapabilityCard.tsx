import { motion } from "framer-motion";
import {
  Sun,
  ArrowLeftRight,
  Heart,
  ShieldAlert,
  FlaskConical,
  Calculator,
} from "lucide-react";
import type { Capability } from "../../types/weather";
import GlassCard from "../ui/GlassCard";

const ICON_MAP: Record<string, React.FC<{ size?: number; className?: string }>> = {
  Sun,
  ArrowLeftRight,
  Heart,
  ShieldAlert,
  FlaskConical,
  Calculator,
};

interface Props {
  capability: Capability;
  index: number;
  onHover: () => void;
  onLeave: () => void;
  onClick: () => void;
}

export default function CapabilityCard({
  capability,
  index,
  onHover,
  onLeave,
  onClick,
}: Props) {
  const Icon = ICON_MAP[capability.icon];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 + index * 0.1, duration: 0.5 }}
    >
      <GlassCard
        onClick={onClick}
        onMouseEnter={onHover}
        onMouseLeave={onLeave}
        className={`group h-full ${capability.bgTint}`}
      >
        {/* Icon + Title */}
        <div className="mb-3 flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${capability.gradient} shadow-lg`}
          >
            {Icon && <Icon size={20} className="text-white" />}
          </div>
          <h3 className="text-base font-semibold text-slate-800">
            {capability.title}
          </h3>
        </div>

        {/* Description */}
        <p className="mb-3 text-sm leading-relaxed text-slate-500">
          {capability.description}
        </p>

        {/* Example query */}
        <div className="rounded-lg bg-slate-100/80 px-3 py-2 transition-colors group-hover:bg-slate-100">
          <p className="text-xs font-medium text-slate-400">Try asking:</p>
          <p className="text-sm italic text-slate-600">
            &ldquo;{capability.example}&rdquo;
          </p>
        </div>
      </GlassCard>
    </motion.div>
  );
}
