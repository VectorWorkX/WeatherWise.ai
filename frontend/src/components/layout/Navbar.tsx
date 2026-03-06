import { Link } from "react-router-dom";
import { MessageSquare } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link to="/" className="flex items-center gap-2.5">
          <img
            src="/weatherwise-icon.png"
            alt="WeatherWise"
            className="h-7 w-7"
            style={{ filter: "saturate(0.6) hue-rotate(180deg) brightness(1.3)" }}
          />
          <span className="text-[15px] font-medium tracking-wide">
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">WeatherWise</span>
            <span className="text-neutral-600">.ai</span>
          </span>
        </Link>

        <Link
          to="/chat"
          className="flex items-center gap-2 rounded-full bg-white/[0.07] px-5 py-2 text-sm font-medium text-neutral-300 transition hover:bg-white/[0.12] hover:text-white"
        >
          <MessageSquare size={14} />
          Start Chatting
        </Link>
      </div>
    </nav>
  );
}
