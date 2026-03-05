export const INTENT_LABELS: Record<string, string> = {
  real_time: "Real-Time Forecast",
  comparative: "Location Comparison",
  lifestyle: "Lifestyle Advisory",
  safety: "Safety & Alerts",
  explanation: "Atmospheric Science",
  math: "Weather Math",
};

export const INTENT_COLORS: Record<string, string> = {
  real_time: "from-blue-200 to-blue-300",
  comparative: "from-blue-200 to-blue-300",
  lifestyle: "from-blue-200 to-blue-300",
  safety: "from-blue-200 to-blue-300",
  explanation: "from-blue-200 to-blue-300",
  math: "from-blue-200 to-blue-300",
};

export const WS_URL =
  import.meta.env.VITE_WS_URL || `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/chat`;
