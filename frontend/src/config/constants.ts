export const INTENT_LABELS: Record<string, string> = {
  real_time: "Real-Time Forecast",
  comparative: "Location Comparison",
  lifestyle: "Lifestyle Advisory",
  safety: "Safety & Alerts",
  explanation: "Atmospheric Science",
  math: "Weather Math",
};

export const INTENT_COLORS: Record<string, string> = {
  real_time: "bg-white/[0.06] text-neutral-300",
  comparative: "bg-white/[0.06] text-neutral-300",
  lifestyle: "bg-white/[0.06] text-neutral-300",
  safety: "bg-amber-500/10 text-amber-300",
  explanation: "bg-white/[0.06] text-neutral-300",
  math: "bg-white/[0.06] text-neutral-300",
};

export const WS_URL =
  import.meta.env.VITE_WS_URL || `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/chat`;
