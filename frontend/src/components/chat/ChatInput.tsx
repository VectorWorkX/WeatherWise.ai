import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";

interface Props {
  onSend: (message: string) => void;
  disabled?: boolean;
  initialValue?: string;
}

export default function ChatInput({ onSend, disabled, initialValue }: Props) {
  const [value, setValue] = useState(initialValue || "");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (initialValue) {
      setValue(initialValue);
    }
  }, [initialValue]);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  }, [value]);

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-3 rounded-2xl border border-white/[0.12] bg-white/[0.06] px-5 py-3 transition-colors focus-within:border-white/[0.2] focus-within:bg-white/[0.08]">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the weather..."
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none bg-transparent py-1.5 text-[15px] text-neutral-200 placeholder-neutral-500 outline-none disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-neutral-900 transition hover:bg-neutral-200 disabled:opacity-20 disabled:hover:bg-white"
          >
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-neutral-700">
          WeatherWise can make mistakes. Verify important weather information.
        </p>
      </div>
    </div>
  );
}
