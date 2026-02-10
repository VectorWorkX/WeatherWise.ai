import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { Send } from "lucide-react";

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

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
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
    <div className="border-t border-slate-200/60 glass">
      <div className="mx-auto flex max-w-3xl items-end gap-3 px-4 py-3">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about the weather..."
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none rounded-xl border border-slate-200 bg-white/80 px-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 outline-none transition focus:border-emerald-300 focus:ring-1 focus:ring-emerald-300/30 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-400 text-white shadow-lg transition hover:bg-emerald-500 disabled:opacity-40 disabled:shadow-none"
        >
          <Send size={18} />
        </button>
      </div>
      <p className="pb-2 text-center text-[10px] text-slate-400">
        Press Enter to send &middot; Shift+Enter for new line
      </p>
    </div>
  );
}
