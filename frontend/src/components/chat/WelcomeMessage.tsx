import { motion } from "framer-motion";
import { CloudSun, MapPin, Thermometer, Wind } from "lucide-react";

const SUGGESTIONS = [
  { icon: CloudSun, query: "What's the weather in San Francisco right now?" },
  { icon: MapPin, query: "Compare weather between Tokyo and London" },
  { icon: Thermometer, query: "Is it hotter in Phoenix or Miami today?" },
  { icon: Wind, query: "Are there any severe weather alerts in Texas?" },
];

interface Props {
  onQueryClick: (query: string) => void;
}

export default function WelcomeMessage({ onQueryClick }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-3xl px-4"
    >
      <div className="mb-10 text-center">
        <img
          src="/weatherwise-icon.png"
          alt="WeatherWise"
          className="mx-auto mb-6 h-11 w-11 opacity-80"
          style={{ filter: "saturate(0.6) hue-rotate(180deg) brightness(1.3)" }}
        />
        <h2 className="mb-2 text-2xl font-medium text-neutral-200">
          What do you want to know?
        </h2>
        <p className="text-[15px] text-neutral-500">
          Weather forecasts, comparisons, alerts, and atmospheric science.
        </p>
      </div>

      <div className="grid gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, query }, i) => (
          <motion.button
            key={query}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + i * 0.05 }}
            onClick={() => onQueryClick(query)}
            className="group flex items-start gap-3 rounded-xl border border-white/[0.06] px-4 py-3.5 text-left transition hover:border-blue-500/20 hover:bg-white/[0.03]"
          >
            <Icon size={16} className="mt-0.5 shrink-0 text-neutral-600 transition group-hover:text-blue-400" />
            <p className="text-[14px] text-neutral-500 transition group-hover:text-neutral-300">
              {query}
            </p>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
