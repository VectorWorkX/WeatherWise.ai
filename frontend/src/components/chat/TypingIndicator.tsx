export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 px-1">
      <div className="flex items-center gap-1">
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue" />
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue" />
        <span className="typing-dot inline-block h-2 w-2 rounded-full bg-brand-blue" />
      </div>
      <span className="text-xs text-slate-500">WeatherWise is analyzing...</span>
    </div>
  );
}
