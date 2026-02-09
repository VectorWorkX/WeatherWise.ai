export const INTENT_LABELS: Record<string, string> = {
  real_time: "Real-Time Forecast",
  comparative: "Location Comparison",
  lifestyle: "Lifestyle Advisory",
  safety: "Safety & Alerts",
  explanation: "Atmospheric Science",
  math: "Weather Math",
};

export const INTENT_COLORS: Record<string, string> = {
  real_time: "from-amber-500 to-orange-500",
  comparative: "from-blue-500 to-cyan-500",
  lifestyle: "from-pink-500 to-rose-500",
  safety: "from-red-500 to-orange-500",
  explanation: "from-violet-500 to-purple-500",
  math: "from-emerald-500 to-teal-500",
};

export const WS_URL =
  import.meta.env.VITE_WS_URL || `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/chat`;
