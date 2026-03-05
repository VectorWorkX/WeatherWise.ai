interface Props {
  opacity?: number;
}

export default function WeatherElements({ opacity = 1 }: Props) {
  return (
    <div
      className="absolute inset-0 overflow-hidden pointer-events-none transition-opacity duration-1000"
      style={{ opacity }}
    >
      {/* ── Shared SVG defs (gradients + filters) ── */}
      <svg className="absolute" width="0" height="0" aria-hidden="true">
        <defs>
          {/* Sun */}
          <radialGradient id="we-sun-ambient">
            <stop offset="0%" stopColor="#FDE68A" stopOpacity="0.25" />
            <stop offset="60%" stopColor="#FDE68A" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#FDE68A" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="we-sun-glow">
            <stop offset="0%" stopColor="#FBBF24" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="we-sun-core">
            <stop offset="0%" stopColor="#FEF3C7" stopOpacity="0.85" />
            <stop offset="50%" stopColor="#F59E0B" stopOpacity="0.65" />
            <stop offset="100%" stopColor="#D97706" stopOpacity="0.35" />
          </radialGradient>

          {/* Clouds – dark (blue-tinted) */}
          <linearGradient id="we-cld-dk-body" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#8BA8C8" stopOpacity="0.48" />
            <stop offset="100%" stopColor="#5A7A96" stopOpacity="0.55" />
          </linearGradient>
          <linearGradient id="we-cld-dk-hi" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#E0EAF4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#9AB4CC" stopOpacity="0.32" />
          </linearGradient>

          {/* Clouds – light (soft blue-white) */}
          <linearGradient id="we-cld-lt-body" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#EFF4FA" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#C5D5E8" stopOpacity="0.48" />
          </linearGradient>

          {/* Cloud shadow (cooler blue-grey) */}
          <linearGradient id="we-cld-shadow" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#4A6580" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#4A6580" stopOpacity="0.03" />
          </linearGradient>

          {/* Lightning */}
          <filter id="we-bolt-glow" x="-60%" y="-30%" width="220%" height="160%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id="we-bolt-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#EDE9FE" stopOpacity="0.8" />
            <stop offset="40%" stopColor="#A78BFA" stopOpacity="0.65" />
            <stop offset="100%" stopColor="#7C3AED" stopOpacity="0.45" />
          </linearGradient>

          {/* Rain drops */}
          <linearGradient id="we-drop" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#BFDBFE" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.65" />
          </linearGradient>
          <radialGradient id="we-drop-hi" cx="0.35" cy="0.25" r="0.3">
            <stop offset="0%" stopColor="#fff" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#fff" stopOpacity="0" />
          </radialGradient>

          {/* Wind */}
          <linearGradient id="we-wind" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#10B981" stopOpacity="0.08" />
            <stop offset="50%" stopColor="#10B981" stopOpacity="0.42" />
            <stop offset="100%" stopColor="#10B981" stopOpacity="0.08" />
          </linearGradient>

          {/* Sparkle */}
          <radialGradient id="we-spark">
            <stop offset="0%" stopColor="#FBBF24" stopOpacity="0.65" />
            <stop offset="100%" stopColor="#FBBF24" stopOpacity="0" />
          </radialGradient>
        </defs>
      </svg>

      {/* ━━ Sun ━━ upper-left */}
      <div className="absolute -top-12 -left-12 weather-pulse">
        <svg width="290" height="290" viewBox="0 0 290 290" fill="none">
          <circle cx="145" cy="145" r="138" fill="url(#we-sun-ambient)" />
          <circle cx="145" cy="145" r="80" fill="url(#we-sun-glow)" />
          <circle cx="145" cy="145" r="40" fill="url(#we-sun-core)" />
          {/* 12 rays — alternating long / short */}
          {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map(
            (deg, i) => {
              const r = (deg * Math.PI) / 180;
              const inner = 52;
              const outer = i % 2 === 0 ? 95 : 72;
              return (
                <line
                  key={deg}
                  x1={145 + inner * Math.cos(r)}
                  y1={145 + inner * Math.sin(r)}
                  x2={145 + outer * Math.cos(r)}
                  y2={145 + outer * Math.sin(r)}
                  stroke="#FBBF24"
                  strokeWidth={i % 2 === 0 ? "3.5" : "2"}
                  strokeLinecap="round"
                  opacity={i % 2 === 0 ? 0.5 : 0.3}
                />
              );
            },
          )}
        </svg>
      </div>

      {/* ━━ Light cloud ━━ left-centre */}
      <div
        className="absolute top-[28%] left-0 weather-drift"
        style={{ animationDuration: "12s" }}
      >
        <svg width="240" height="130" viewBox="0 0 240 130" fill="none">
          <ellipse cx="120" cy="102" rx="100" ry="16" opacity="0.07" fill="#94A3B8" />
          <ellipse cx="90" cy="80" rx="55" ry="35" fill="url(#we-cld-lt-body)" />
          <ellipse cx="155" cy="76" rx="60" ry="38" fill="url(#we-cld-lt-body)" />
          <ellipse cx="55" cy="85" rx="42" ry="28" fill="url(#we-cld-lt-body)" />
          <ellipse cx="115" cy="58" rx="44" ry="26" fill="url(#we-cld-lt-body)" opacity="0.8" />
          <ellipse cx="165" cy="55" rx="38" ry="24" fill="url(#we-cld-lt-body)" opacity="0.7" />
        </svg>
      </div>

      {/* ━━ Small light cloud ━━ bottom-right */}
      <div
        className="absolute bottom-[16%] right-6 weather-float hidden sm:block"
        style={{ animationDuration: "9s", animationDelay: "3s" }}
      >
        <svg width="200" height="110" viewBox="0 0 200 110" fill="none">
          <ellipse cx="100" cy="86" rx="80" ry="14" opacity="0.06" fill="#94A3B8" />
          <ellipse cx="78" cy="66" rx="48" ry="30" fill="url(#we-cld-lt-body)" />
          <ellipse cx="130" cy="62" rx="52" ry="34" fill="url(#we-cld-lt-body)" />
          <ellipse cx="96" cy="48" rx="38" ry="24" fill="url(#we-cld-lt-body)" opacity="0.8" />
        </svg>
      </div>

      {/* ━━ Lightning bolt ━━ below large cloud, right side */}
      <div
        className="absolute top-36 right-20 weather-twinkle"
        style={{ animationDuration: "3s" }}
      >
        <svg width="80" height="155" viewBox="0 0 55 100" fill="none">
          {/* Glow layer */}
          <path
            d="M30,0 L12,40 L22,40 L15,100 L42,34 L28,34 Z"
            fill="#8B5CF6"
            opacity="0.35"
            filter="url(#we-bolt-glow)"
          />
          {/* Main bolt */}
          <path
            d="M30,0 L12,40 L22,40 L15,100 L42,34 L28,34 Z"
            fill="url(#we-bolt-fill)"
          />
          {/* Bright inner core */}
          <path
            d="M28,6 L15,38 L22,38 L17,88 L38,35 L28,35 Z"
            fill="#EDE9FE"
            opacity="0.2"
          />
        </svg>
      </div>

      {/* ━━ Rain drops ━━ below lightning */}
      <div
        className="absolute top-[42%] right-14 weather-bob"
        style={{ animationDuration: "2.5s" }}
      >
        <svg width="85" height="100" viewBox="0 0 60 70" fill="none">
          {/* Drop 1 */}
          <path
            d="M14,36 Q14,26 9.5,14 Q5,26 5,36 Q5,42 9.5,42 Q14,42 14,36Z"
            fill="url(#we-drop)"
          />
          <ellipse cx="8.5" cy="28" rx="2.5" ry="3.5" fill="url(#we-drop-hi)" />
          {/* Drop 2 */}
          <path
            d="M34,48 Q34,38 29,26 Q24,38 24,48 Q24,55 29,55 Q34,55 34,48Z"
            fill="url(#we-drop)"
          />
          <ellipse cx="28" cy="38" rx="2.5" ry="3.5" fill="url(#we-drop-hi)" />
          {/* Drop 3 */}
          <path
            d="M52,30 Q52,22 49,14 Q46,22 46,30 Q46,35 49,35 Q52,35 52,30Z"
            fill="url(#we-drop)"
          />
          <ellipse cx="48" cy="23" rx="2" ry="3" fill="url(#we-drop-hi)" />
        </svg>
      </div>

      {/* ━━ Extra rain drops ━━ staggered */}
      <div
        className="absolute top-[50%] right-32 weather-bob hidden sm:block"
        style={{ animationDuration: "3s", animationDelay: "0.8s" }}
      >
        <svg width="60" height="70" viewBox="0 0 40 46" fill="none">
          <path
            d="M12,26 Q12,19 9,10 Q6,19 6,26 Q6,32 9,32 Q12,32 12,26Z"
            fill="url(#we-drop)"
          />
          <ellipse cx="8.5" cy="19" rx="2" ry="2.5" fill="url(#we-drop-hi)" />
          <path
            d="M32,34 Q32,27 28.5,18 Q25,27 25,34 Q25,40 28.5,40 Q32,40 32,34Z"
            fill="url(#we-drop)"
          />
          <ellipse cx="28" cy="27" rx="2" ry="2.5" fill="url(#we-drop-hi)" />
        </svg>
      </div>

      {/* ━━ Wind curves ━━ bottom-left */}
      <div
        className="absolute bottom-16 left-2 weather-sway"
        style={{ animationDuration: "5s" }}
      >
        <svg width="180" height="85" viewBox="0 0 110 55" fill="none">
          <path
            d="M2,16 C20,4 42,4 58,16 C74,28 92,28 108,16"
            stroke="url(#we-wind)"
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
          <path
            d="M6,30 C24,20 44,20 60,30 C76,40 92,40 104,30"
            stroke="url(#we-wind)"
            strokeWidth="2.5"
            strokeLinecap="round"
            fill="none"
          />
          <path
            d="M14,43 C28,35 46,35 58,43 C70,51 84,51 96,43"
            stroke="url(#we-wind)"
            strokeWidth="2"
            strokeLinecap="round"
            fill="none"
          />
        </svg>
      </div>

      {/* ━━ Sparkle ━━ near sun */}
      <div
        className="absolute top-20 left-56 weather-twinkle"
        style={{ animationDuration: "2.5s" }}
      >
        <svg width="36" height="36" viewBox="0 0 22 22" fill="none">
          <circle cx="11" cy="11" r="7" fill="url(#we-spark)" />
          <path
            d="M11,1 L12.6,9 L20,11 L12.6,13 L11,21 L9.4,13 L2,11 L9.4,9 Z"
            fill="#FBBF24"
            opacity="0.6"
          />
        </svg>
      </div>

      {/* ━━ Sparkle ━━ upper area */}
      <div
        className="absolute top-4 right-[36%] weather-twinkle hidden sm:block"
        style={{ animationDuration: "3.5s", animationDelay: "1.2s" }}
      >
        <svg width="30" height="30" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="6" fill="url(#we-spark)" opacity="0.45" />
          <path
            d="M9,1 L10.4,7.4 L16,9 L10.4,10.6 L9,17 L7.6,10.6 L2,9 L7.6,7.4 Z"
            fill="#C4B5FD"
            opacity="0.5"
          />
        </svg>
      </div>

      {/* ━━ Snowflake ━━ top-centre */}
      <div className="absolute top-16 right-24 weather-spin-slow hidden sm:block">
        <svg width="48" height="48" viewBox="0 0 32 32" fill="none">
          {/* Main axes */}
          <line x1="16" y1="2" x2="16" y2="30" stroke="#93C5FD" strokeWidth="2" strokeLinecap="round" opacity="0.4" />
          <line x1="2" y1="16" x2="30" y2="16" stroke="#93C5FD" strokeWidth="2" strokeLinecap="round" opacity="0.4" />
          <line x1="5.5" y1="5.5" x2="26.5" y2="26.5" stroke="#93C5FD" strokeWidth="1.6" strokeLinecap="round" opacity="0.32" />
          <line x1="26.5" y1="5.5" x2="5.5" y2="26.5" stroke="#93C5FD" strokeWidth="1.6" strokeLinecap="round" opacity="0.32" />
          {/* Branch tips */}
          <line x1="16" y1="2" x2="13" y2="6" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="16" y1="2" x2="19" y2="6" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="16" y1="30" x2="13" y2="26" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="16" y1="30" x2="19" y2="26" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="2" y1="16" x2="6" y2="13" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="2" y1="16" x2="6" y2="19" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="30" y1="16" x2="26" y2="13" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          <line x1="30" y1="16" x2="26" y2="19" stroke="#93C5FD" strokeWidth="1.3" strokeLinecap="round" opacity="0.3" />
          {/* Centre dot */}
          <circle cx="16" cy="16" r="2.2" fill="#93C5FD" opacity="0.38" />
        </svg>
      </div>
    </div>
  );
}
