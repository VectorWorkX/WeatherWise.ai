import React, { lazy, Suspense } from "react";
import { motion } from "framer-motion";
import type { VisualReportData } from "../../types/chat";

const ReactECharts = lazy(() => import("echarts-for-react"));

// Error boundary prevents malformed ECharts options from crashing the entire UI
class ChartErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-[300px] flex items-center justify-center text-neutral-500 text-sm">
          Chart could not be rendered
        </div>
      );
    }
    return this.props.children;
  }
}

interface Props {
  report: VisualReportData;
}

export default function WeatherReport({ report }: Props) {
  if (report.error && (!report.sections || report.sections.length === 0)) {
    return (
      <div className="mt-4 rounded-xl border border-white/[0.08] bg-white/[0.03] px-5 py-4">
        <p className="text-sm text-neutral-400">Visualization could not be generated.</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="mt-4 rounded-2xl border border-white/[0.08] bg-white/[0.03] overflow-hidden"
    >
      {/* Header */}
      {report.title && (
        <div className="px-5 pt-5 pb-3">
          <h3 className="text-lg font-semibold text-neutral-100">{report.title}</h3>
          {report.subtitle && (
            <p className="text-sm text-neutral-400 mt-1">{report.subtitle}</p>
          )}
        </div>
      )}

      {/* Metric Cards */}
      {report.metrics && report.metrics.length > 0 && (
        <div className="px-5 pb-4">
          <div className="flex flex-wrap gap-3">
            {report.metrics.map((metric, i) => (
              <div
                key={i}
                className="flex-1 min-w-[140px] rounded-xl bg-white/[0.05] border border-white/[0.06] px-4 py-3"
              >
                <p className="text-xs text-neutral-500 uppercase tracking-wide">{metric.label}</p>
                <p className="text-xl font-bold text-neutral-100 mt-1">
                  {metric.value}
                  {metric.unit && <span className="text-sm font-normal text-neutral-400 ml-1">{metric.unit}</span>}
                </p>
                {metric.detail && (
                  <p className="text-xs text-neutral-500 mt-1">{metric.detail}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chart Sections — stacked vertically, each with its own narrative */}
      {report.sections && report.sections.length > 0 && (
        <div className="px-5 pb-4 space-y-4">
          {report.sections.map((section, i) => (
            <ChartSection key={i} section={section} />
          ))}
        </div>
      )}

      {/* Insights */}
      {report.insights && report.insights.length > 0 && (
        <div className="px-5 pb-5">
          <div className="rounded-xl bg-white/[0.04] border border-white/[0.06] px-4 py-3">
            <p className="text-xs text-neutral-500 uppercase tracking-wide mb-2">Key Insights</p>
            <ul className="space-y-2">
              {report.insights.map((insight, i) => (
                <li key={i} className="text-sm text-neutral-300 flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">&#8226;</span>
                  <span dangerouslySetInnerHTML={{
                    __html: insight
                      .replace(/\*\*(.+?)\*\*/g, '<strong class="text-neutral-100">$1</strong>')
                  }} />
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </motion.div>
  );
}

/** Sanitize an ECharts option so labels, titles, and legends never clip or overlap. */
function sanitizeOption(raw: Record<string, unknown>): Record<string, unknown> {
  const option = { ...raw };

  // --- grid: enforce containLabel and reserve space for title/legend ---
  // grid can be a single object OR an array (dual-axis charts use [{}, {}])
  const hasTitle = Boolean(option.title);
  const hasLegend = Boolean(option.legend);
  // Minimum top margin so grid doesn't overlap title + legend
  const minTop = hasTitle && hasLegend ? 70 : hasTitle || hasLegend ? 50 : 30;

  const fixGrid = (g: Record<string, unknown>): Record<string, unknown> => {
    const fixed: Record<string, unknown> = { ...g, containLabel: true };
    // Reserve space above for title + legend
    const curTop = typeof fixed.top === "number" ? fixed.top : undefined;
    if (curTop === undefined || curTop < minTop) {
      fixed.top = minTop;
    }
    // Reserve space on right for markLine labels / annotations that extend past the axis
    const curRight = typeof fixed.right === "number" ? fixed.right : undefined;
    if (curRight === undefined || curRight < 30) {
      fixed.right = 30;
    }
    return fixed;
  };

  if (Array.isArray(option.grid)) {
    option.grid = (option.grid as Record<string, unknown>[]).map(fixGrid);
  } else {
    const gridObj =
      typeof option.grid === "object" && option.grid !== null
        ? (option.grid as Record<string, unknown>)
        : {};
    option.grid = fixGrid(gridObj);
  }

  // --- pie/radar charts: shrink radius to leave room for labels ---
  if (Array.isArray(option.series)) {
    option.series = (option.series as Record<string, unknown>[]).map((s) => {
      if (s.type === "pie" && !s.radius) {
        return { ...s, radius: ["30%", "60%"], center: ["50%", "55%"] };
      }
      return s;
    });
  }

  return option;
}

function ChartSection({ section }: { section: NonNullable<VisualReportData["sections"]>[number] }) {
  const raw = section.echarts_option as Record<string, unknown>;
  const option = sanitizeOption(raw);

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-neutral-300">{section.heading}</h4>
      </div>
      {section.narrative && (
        <p className="text-xs text-neutral-500 mb-3">{section.narrative}</p>
      )}
      <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-4 overflow-hidden">
        <ChartErrorBoundary>
          <Suspense fallback={<div className="h-[400px] flex items-center justify-center text-neutral-500 text-sm">Loading chart...</div>}>
            <ReactECharts
              option={option}
              theme="dark"
              style={{ height: "400px", width: "100%" }}
              notMerge={true}
            />
          </Suspense>
        </ChartErrorBoundary>
      </div>
    </div>
  );
}
