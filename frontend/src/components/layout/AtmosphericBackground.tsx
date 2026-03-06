export default function AtmosphericBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-[#0a0a0a]" />
      {/* Subtle blue glow at top center */}
      <div
        className="absolute"
        style={{
          top: "-25%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "900px",
          height: "500px",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(56,140,220,0.04) 0%, transparent 65%)",
        }}
      />
    </div>
  );
}
