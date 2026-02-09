import { Link, useLocation } from "react-router-dom";
import { ArrowLeft, MessageSquare } from "lucide-react";
import { motion } from "framer-motion";

export default function Navbar() {
  const location = useLocation();
  const isChat = location.pathname === "/chat";

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <div className="flex items-center gap-3">
          {isChat && (
            <Link
              to="/"
              className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
            >
              <ArrowLeft size={16} />
              <span className="hidden sm:inline">Home</span>
            </Link>
          )}
          <Link to="/" className="flex items-center gap-2.5">
            <motion.img
              layoutId="logo"
              src="/weatherwise-icon.png"
              alt="WeatherWise.ai"
              className="h-8 w-auto object-contain mix-blend-multiply"
            />
            <span className="text-lg font-bold tracking-tight text-slate-800">
              Weather<span className="text-blue-600">Wise</span>
              <span className="text-slate-400">.ai</span>
            </span>
          </Link>
        </div>

        {!isChat && (
          <Link
            to="/chat"
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-blue-500/20 transition hover:shadow-blue-500/30 hover:brightness-110"
          >
            <MessageSquare size={16} />
            Start Chatting
          </Link>
        )}
      </div>
    </nav>
  );
}
