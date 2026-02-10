import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Navbar from "./components/layout/Navbar";
import WeatherElements from "./components/layout/WeatherElements";
import LandingPage from "./pages/LandingPage";
import ChatPage from "./pages/ChatPage";
import { useWeatherBackground } from "./hooks/useWeatherBackground";

export default function App() {
  const location = useLocation();
  const isChat = location.pathname === "/chat";
  const { setHoverMode, clearHoverMode } = useWeatherBackground(!isChat);

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-50">
      {/* Layer 1: Base gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-white via-blue-50/40 to-indigo-50/50" />

      {/* Layer 2: Weather particles */}
      <div className="fixed inset-0 z-[1]">
        <WeatherElements opacity={1} />
      </div>

      {/* Layer 3: Radial overlay for depth */}
      <div className="fixed inset-0 z-[2] bg-[radial-gradient(ellipse_at_center,transparent_0%,rgba(248,250,252,0.4)_100%)]" />

      {/* Layer 4: Content */}
      <div className="relative z-[3]">
        <Navbar />
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={
                <LandingPage
                  onHoverMode={setHoverMode}
                  onLeaveMode={clearHoverMode}
                />
              }
            />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  );
}
