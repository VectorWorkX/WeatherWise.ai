import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Wifi, WifiOff } from "lucide-react";
import { useWeatherChat } from "../hooks/useWeatherChat";
import MessageList from "../components/chat/MessageList";
import ChatInput from "../components/chat/ChatInput";
import WelcomeMessage from "../components/chat/WelcomeMessage";

export default function ChatPage() {
  const { messages, sendMessage, isConnected, isLoading } = useWeatherChat();
  const [searchParams, setSearchParams] = useSearchParams();
  const autoSentRef = useRef(false);

  // Auto-send query from URL param (from landing page card clicks)
  useEffect(() => {
    const q = searchParams.get("q");
    if (q && isConnected && !autoSentRef.current) {
      autoSentRef.current = true;
      // Small delay to let WebSocket stabilize
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
      transition={{ duration: 0.4 }}
      className="relative z-10 flex h-screen flex-col pt-14"
    >
      {/* Connection status */}
      <div className="flex items-center justify-center gap-1.5 py-1 text-[10px]">
        {isConnected ? (
          <>
            <Wifi size={10} className="text-emerald-500" />
            <span className="text-emerald-500/70">Connected</span>
          </>
        ) : (
          <>
            <WifiOff size={10} className="text-red-400" />
            <span className="text-red-400/70">Reconnecting...</span>
          </>
        )}
      </div>

      {/* Messages or welcome */}
      {messages.length === 0 ? (
        <div className="flex-1 overflow-y-auto">
          <WelcomeMessage onQueryClick={handleQueryClick} />
        </div>
      ) : (
        <MessageList messages={messages} />
      )}

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        disabled={!isConnected || isLoading}
      />
    </motion.div>
  );
}
