import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import AtmosphericBackground from "./components/layout/AtmosphericBackground";
import LandingPage from "./pages/LandingPage";
import ChatPage from "./pages/ChatPage";

export default function App() {
  const location = useLocation();

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0a0a0a]">
      <AtmosphericBackground />
      <div className="relative z-10">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  );
}
