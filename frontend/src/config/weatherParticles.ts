import type { WeatherMode } from "../types/weather";
import type { ISourceOptions } from "@tsparticles/engine";

const rain: ISourceOptions = {
  fullScreen: false,
  background: { color: "transparent" },
  particles: {
    number: { value: 120, density: { enable: true } },
    color: { value: "#3b82f6" },
    shape: { type: "circle" },
    opacity: { value: { min: 0.3, max: 0.6 } },
    size: { value: { min: 1, max: 3 } },
    move: {
      enable: true,
      direction: "bottom",
      speed: { min: 8, max: 16 },
      straight: true,
    },
    life: {
      duration: { value: 0 },
    },
  },
  detectRetina: true,
};

const snow: ISourceOptions = {
  fullScreen: false,
  background: { color: "transparent" },
  particles: {
    number: { value: 80, density: { enable: true } },
    color: { value: "#94a3b8" },
    shape: { type: "circle" },
    opacity: { value: { min: 0.3, max: 0.7 } },
    size: { value: { min: 2, max: 6 } },
    move: {
      enable: true,
      direction: "bottom",
      speed: { min: 1, max: 3 },
      straight: false,
      outModes: { default: "out" },
    },
    wobble: {
      enable: true,
      distance: 10,
      speed: 5,
    },
  },
  detectRetina: true,
};

const sun: ISourceOptions = {
  fullScreen: false,
  background: { color: "transparent" },
  particles: {
    number: { value: 30, density: { enable: true } },
    color: { value: ["#f59e0b", "#d97706", "#eab308"] },
    shape: { type: "circle" },
    opacity: {
      value: { min: 0.1, max: 0.3 },
      animation: { enable: true, speed: 0.5, startValue: "random", sync: false },
    },
    size: {
      value: { min: 10, max: 50 },
      animation: { enable: true, speed: 3, startValue: "random", sync: false },
    },
    move: {
      enable: true,
      direction: "none",
      speed: { min: 0.3, max: 1 },
      outModes: { default: "bounce" },
    },
  },
  detectRetina: true,
};

const lightning: ISourceOptions = {
  fullScreen: false,
  background: { color: "transparent" },
  particles: {
    number: { value: 50, density: { enable: true } },
    color: { value: ["#7c3aed", "#8b5cf6", "#6366f1"] },
    shape: { type: "circle" },
    opacity: {
      value: { min: 0.2, max: 0.7 },
      animation: { enable: true, speed: 3, startValue: "random", sync: false },
    },
    size: { value: { min: 1, max: 4 } },
    move: {
      enable: true,
      direction: "bottom",
      speed: { min: 4, max: 12 },
      straight: false,
    },
    links: {
      enable: true,
      color: "#7c3aed",
      distance: 80,
      opacity: 0.25,
      width: 1,
    },
  },
  detectRetina: true,
};

export const weatherConfigs: Record<WeatherMode, ISourceOptions> = {
  rain,
  snow,
  sun,
  lightning,
};

export const WEATHER_MODES: WeatherMode[] = ["rain", "sun", "snow", "lightning"];
