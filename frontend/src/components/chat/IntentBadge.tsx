import { INTENT_COLORS } from "../../config/constants";

interface Props {
  intent: string;
  label: string;
}

export default function IntentBadge({ intent, label }: Props) {
  const color = INTENT_COLORS[intent] || "bg-slate-500/15 text-slate-400";

  return (
    <span className={`inline-block rounded-full px-3 py-0.5 text-xs font-medium ${color}`}>
      {label}
    </span>
  );
}
