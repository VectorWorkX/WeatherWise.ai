import { useState, useEffect, useMemo } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import type { WeatherMode } from "../../types/weather";
import { weatherConfigs } from "../../config/weatherParticles";

interface Props {
  mode: WeatherMode;
  opacity?: number;
}

export default function ParticleBackground({ mode, opacity = 1 }: Props) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => setReady(true));
  }, []);

  const options = useMemo(() => weatherConfigs[mode], [mode]);

  if (!ready) return null;

  return (
    <div
      className="absolute inset-0 transition-opacity duration-1000"
      style={{ opacity }}
    >
      <Particles
        id={`particles-${mode}`}
        key={mode}
        options={options}
        className="absolute inset-0"
      />
      {mode === "lightning" && <div className="absolute inset-0 lightning-flash" />}
    </div>
  );
}
