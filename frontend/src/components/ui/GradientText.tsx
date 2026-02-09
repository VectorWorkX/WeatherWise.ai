import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
}

export default function GradientText({ children, className = "" }: Props) {
  return (
    <span className={`gradient-text ${className}`}>{children}</span>
  );
}
