import { INTENT_COLORS } from "../../config/constants";

interface Props {
  intent: string;
  label: string;
}

export default function IntentBadge({ intent, label }: Props) {
  const gradient = INTENT_COLORS[intent] || "from-slate-500 to-slate-600";

  return (
    <span
      className={`inline-block rounded-full bg-gradient-to-r ${gradient} px-3 py-0.5 text-xs font-semibold text-white shadow-sm`}
    >
      {label}
    </span>
  );
}
