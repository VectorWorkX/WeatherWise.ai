import { motion } from "framer-motion";
import { CAPABILITIES } from "../../config/capabilities";

interface Props {
  onQueryClick: (query: string) => void;
}

export default function WelcomeMessage({ onQueryClick }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-3xl px-4 py-12"
    >
      <div className="text-center">
        <img
          src="/weatherwise-icon.png"
          alt="WeatherWise.ai"
          className="mx-auto mb-4 h-16 w-auto object-contain"
        />
        <h2 className="mb-2 text-xl font-bold text-slate-700">
          How can I help you today?
        </h2>
        <p className="mb-8 text-sm text-slate-500">
          Ask me anything about weather — from live forecasts to atmospheric
          science.
        </p>
      </div>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {CAPABILITIES.map((cap, i) => (
          <motion.button
            key={cap.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 + i * 0.05 }}
            onClick={() => onQueryClick(cap.example)}
            className={`group rounded-xl border-2 border-slate-400/40 ${cap.bgTint} px-3 py-3 text-left transition hover:border-emerald-300/60 hover:bg-white/80 hover:shadow-sm hover:shadow-emerald-500/5`}
          >
            <p className="mb-1 text-sm font-semibold text-slate-700 transition group-hover:text-slate-800">
              {cap.title}
            </p>
            <p className="text-sm text-slate-600 transition group-hover:text-slate-700">
              &ldquo;{cap.example}&rdquo;
            </p>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
