export type WeatherMode = "rain" | "snow" | "sun" | "lightning";

export interface Capability {
  id: string;
  title: string;
  icon: string;
  description: string;
  example: string;
  weatherMode: WeatherMode;
  gradient: string;
  bgTint: string;
}
