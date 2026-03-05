import HeroSection from "../components/landing/HeroSection";
import CapabilityGrid from "../components/landing/CapabilityGrid";
import type { WeatherMode } from "../types/weather";

interface Props {
  onHoverMode: (mode: WeatherMode) => void;
  onLeaveMode: () => void;
}

export default function LandingPage({ onHoverMode, onLeaveMode }: Props) {
  return (
    <main className="relative z-10 min-h-screen">
      <HeroSection />
      <CapabilityGrid onHoverMode={onHoverMode} onLeaveMode={onLeaveMode} />
    </main>
  );
}
