import { useState, useEffect, useCallback, useRef } from "react";
import type { WeatherMode } from "../types/weather";
import { WEATHER_MODES } from "../config/weatherParticles";

/**
 * Manages the active weather particle mode.
 * Auto-cycles every 8s on the landing page; can be overridden by hover.
 */
export function useWeatherBackground(autoCycle = true) {
  const [mode, setMode] = useState<WeatherMode>("rain");
  const [isHovering, setIsHovering] = useState(false);
  const indexRef = useRef(0);

  useEffect(() => {
    if (!autoCycle || isHovering) return;

    const interval = setInterval(() => {
      indexRef.current = (indexRef.current + 1) % WEATHER_MODES.length;
      setMode(WEATHER_MODES[indexRef.current]);
    }, 8000);

    return () => clearInterval(interval);
  }, [autoCycle, isHovering]);

  const setHoverMode = useCallback((weatherMode: WeatherMode) => {
    setIsHovering(true);
    setMode(weatherMode);
  }, []);

  const clearHoverMode = useCallback(() => {
    setIsHovering(false);
  }, []);

  return { mode, setMode, setHoverMode, clearHoverMode };
}
