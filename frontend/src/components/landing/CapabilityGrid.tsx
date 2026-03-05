import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { CAPABILITIES } from "../../config/capabilities";
import CapabilityCard from "./CapabilityCard";
import type { WeatherMode } from "../../types/weather";

interface Props {
  onHoverMode: (mode: WeatherMode) => void;
  onLeaveMode: () => void;
}

export default function CapabilityGrid({ onHoverMode, onLeaveMode }: Props) {
  const navigate = useNavigate();

  return (
    <section className="mx-auto max-w-5xl px-4 pb-20">
      <motion.h2
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mb-8 text-center text-2xl font-bold text-slate-700"
      >
        What can I help you with?
      </motion.h2>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CAPABILITIES.map((cap, i) => (
          <CapabilityCard
            key={cap.id}
            capability={cap}
            index={i}
            onHover={() => onHoverMode(cap.weatherMode)}
            onLeave={onLeaveMode}
            onClick={() =>
              navigate(`/chat?q=${encodeURIComponent(cap.example)}`)
            }
          />
        ))}
      </div>
    </section>
  );
}
