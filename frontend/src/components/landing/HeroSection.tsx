import { motion } from "framer-motion";
import { MessageSquare } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="flex flex-col items-center pt-32 pb-12 text-center"
    >
      {/* Logo */}
      <motion.img
        layoutId="logo-hero"
        src="/weatherwise-icon.png"
        alt="WeatherWise.ai"
        className="mb-6 h-32 w-auto object-contain"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      />

      {/* Title */}
      <h1 className="mb-4 text-5xl font-extrabold tracking-tight text-stroke sm:text-6xl md:text-7xl">
        <span className="text-orange-350">Weather</span>
        <span className="text-emerald-500">Wise</span>
        <span className="text-emerald-500">.ai</span>
      </h1>

      {/* Tagline */}
      <p className="mb-8 max-w-xl text-lg text-slate-500 sm:text-xl">
        Your expert weather analyst, powered by AI.
        <br />
        <span className="text-slate-400">
          Real-time data. Deep science. Actionable insights.
        </span>
      </p>

      {/* CTA Button */}
      <motion.button
        onClick={() => navigate("/chat")}
        className="group flex items-center gap-2.5 rounded-2xl bg-emerald-500 px-8 py-3.5 text-lg font-semibold text-white shadow-xl shadow-emerald-500/25 transition-all hover:bg-emerald-600 hover:shadow-emerald-500/40"
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.98 }}
      >
        <MessageSquare size={20} />
        Ask Me Anything
      </motion.button>

      {/* Powered by */}
      <p className="mt-6 text-xs text-slate-400">
        Powered by Claude, Xweather MCP, and RAG Knowledge Base
      </p>
    </motion.section>
  );
}
