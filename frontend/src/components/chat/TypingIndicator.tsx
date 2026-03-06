export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1">
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-neutral-500" />
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-neutral-500" />
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-neutral-500" />
      </div>
    </div>
  );
}
