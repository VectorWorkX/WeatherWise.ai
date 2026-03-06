import { useEffect, useRef } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useWeatherChat } from "../hooks/useWeatherChat";
import MessageList from "../components/chat/MessageList";
import ChatInput from "../components/chat/ChatInput";
import WelcomeMessage from "../components/chat/WelcomeMessage";

export default function ChatPage() {
  const { messages, sendMessage, sendApproval, isConnected, isLoading } = useWeatherChat();
  const [searchParams, setSearchParams] = useSearchParams();
  const autoSentRef = useRef(false);

  useEffect(() => {
    const q = searchParams.get("q");
    if (q && isConnected && !autoSentRef.current) {
      autoSentRef.current = true;
      const timer = setTimeout(() => {
        sendMessage(q);
        setSearchParams({}, { replace: true });
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [searchParams, isConnected, sendMessage, setSearchParams]);

  const handleQueryClick = (query: string) => {
    sendMessage(query);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="flex h-screen flex-col"
    >
      {/* Minimal top bar */}
      <div className="flex items-center justify-between px-5 py-3">
        <Link to="/" className="flex items-center gap-2">
          <img
            src="/weatherwise-icon.png"
            alt="WeatherWise"
            className="h-6 w-6"
            style={{ filter: "saturate(0.6) hue-rotate(180deg) brightness(1.3)" }}
          />
          <span className="text-[14px] font-medium tracking-wide">
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">WeatherWise</span>
            <span className="text-neutral-600">.ai</span>
          </span>
        </Link>
        <div className="flex items-center gap-2">
          <div
            className={`h-1.5 w-1.5 rounded-full ${
              isConnected ? "bg-emerald-400" : "bg-red-400 animate-pulse"
            }`}
          />
          <span className="text-xs text-neutral-600">
            {isConnected ? "Connected" : "Reconnecting..."}
          </span>
        </div>
      </div>

      {messages.length === 0 ? (
        <div className="flex flex-1 items-center justify-center overflow-y-auto">
          <WelcomeMessage onQueryClick={handleQueryClick} />
        </div>
      ) : (
        <MessageList messages={messages} onApproval={sendApproval} />
      )}

      <ChatInput
        onSend={sendMessage}
        disabled={!isConnected || isLoading}
      />
    </motion.div>
  );
}
