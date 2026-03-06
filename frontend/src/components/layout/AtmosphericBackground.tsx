export default function AtmosphericBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-[#0a0a0a]" />

      {/* Warm blob — bottom-left (heat zone: red/orange) */}
      <div
        className="absolute"
        style={{
          bottom: "-5%",
          left: "-3%",
          width: "800px",
          height: "700px",
          borderRadius: "50%",
          background:
            "radial-gradient(ellipse, rgba(175,50,20,0.18) 0%, rgba(190,70,15,0.08) 40%, transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      {/* Cool blob — top-right (cold front: blue) */}
      <div
        className="absolute"
        style={{
          top: "-8%",
          right: "-3%",
          width: "900px",
          height: "750px",
          borderRadius: "50%",
          background:
            "radial-gradient(ellipse, rgba(30,80,180,0.20) 0%, rgba(25,120,200,0.08) 40%, transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      {/* Warm accent — top-left (amber) */}
      <div
        className="absolute"
        style={{
          top: "5%",
          left: "8%",
          width: "550px",
          height: "450px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(200,120,30,0.13) 0%, rgba(180,90,15,0.04) 50%, transparent 70%)",
          filter: "blur(50px)",
        }}
      />

      {/* Cool accent — bottom-right (teal) */}
      <div
        className="absolute"
        style={{
          bottom: "0%",
          right: "5%",
          width: "550px",
          height: "500px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(20,140,160,0.15) 0%, rgba(15,110,135,0.05) 50%, transparent 70%)",
          filter: "blur(50px)",
        }}
      />

      {/* Center transition — purple where warm meets cool */}
      <div
        className="absolute"
        style={{
          top: "40%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "700px",
          height: "600px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(85,45,125,0.10) 0%, rgba(70,35,110,0.04) 45%, transparent 65%)",
          filter: "blur(60px)",
        }}
      />
    </div>
  );
}
