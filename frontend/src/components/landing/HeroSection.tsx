import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="flex min-h-screen flex-col items-center justify-center px-4 text-center"
    >
      {/* Logo — blue-tinted */}
      <motion.div
        className="mb-10"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.6 }}
      >
        <img
          src="/weatherwise-icon.png"
          alt="WeatherWise.ai"
          className="h-14 w-14"
          style={{ filter: "saturate(0.6) hue-rotate(180deg) brightness(1.3)" }}
        />
      </motion.div>

      {/* Title — smaller, gradient blue */}
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.7 }}
        className="mb-6 text-4xl font-semibold tracking-tight sm:text-5xl"
      >
        <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-teal-400 bg-clip-text text-transparent">
          WeatherWise
        </span>
        <span className="text-neutral-600">.ai</span>
      </motion.h1>

      {/* Tagline */}
      <motion.p
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7 }}
        className="mb-4 max-w-md text-lg text-neutral-400 sm:text-xl"
      >
        Your expert weather analyst, powered by AI.
      </motion.p>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.7 }}
        className="mb-14 text-sm text-neutral-600"
      >
        Real-time data &middot; Atmospheric science &middot; Actionable insights
      </motion.p>

      {/* CTA */}
      <motion.button
        onClick={() => navigate("/chat")}
        className="group flex items-center gap-3 rounded-full bg-gradient-to-r from-blue-600 to-cyan-600 px-7 py-3 text-[15px] font-medium text-white transition-all hover:from-blue-500 hover:to-cyan-500"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.5 }}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        Start a conversation
        <ArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
      </motion.button>

      {/* Powered by */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="mt-16 text-xs text-neutral-700"
      >
        Powered by Claude, Xweather MCP & RAG
      </motion.p>
    </motion.section>
  );
}
