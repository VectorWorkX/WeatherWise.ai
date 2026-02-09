import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

export default function GlassCard({
  children,
  className = "",
  onClick,
  onMouseEnter,
  onMouseLeave,
}: Props) {
  return (
    <div
      className={`glass rounded-2xl p-5 transition-all duration-300 hover:border-slate-300/60 hover:shadow-lg hover:shadow-blue-500/10 ${
        onClick ? "cursor-pointer" : ""
      } ${className}`}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {children}
    </div>
  );
}
