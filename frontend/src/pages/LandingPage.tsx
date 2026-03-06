import Navbar from "../components/layout/Navbar";
import HeroSection from "../components/landing/HeroSection";

export default function LandingPage() {
  return (
    <main className="relative z-10 min-h-screen">
      <Navbar />
      <HeroSection />
    </main>
  );
}
