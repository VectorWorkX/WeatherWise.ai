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

      {/* Chart Sections */}
      {report.sections && report.sections.length > 0 && (
        <div className="px-5 pb-4">
          {/* Group by priority: priority 1 = full width, priority 2 = side by side */}
          {(() => {
            const fullWidth = report.sections!.filter((s) => s.priority === 1);
            const halfWidth = report.sections!.filter((s) => s.priority !== 1);

            return (
              <>
                {/* Full-width charts */}
                {fullWidth.map((section, i) => (
                  <ChartSection key={`full-${i}`} section={section} />
                ))}

                {/* Side-by-side charts */}
                {halfWidth.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {halfWidth.map((section, i) => (
                      <ChartSection key={`half-${i}`} section={section} />
                    ))}
                  </div>
                )}
              </>
            );
          })()}
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

function ChartSection({ section }: { section: NonNullable<VisualReportData["sections"]>[number] }) {
  const option = section.echarts_option as Record<string, unknown>;

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-neutral-300">{section.heading}</h4>
      </div>
      {section.narrative && (
        <p className="text-xs text-neutral-500 mb-3">{section.narrative}</p>
      )}
      <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3">
        <ChartErrorBoundary>
          <Suspense fallback={<div className="h-[300px] flex items-center justify-center text-neutral-500 text-sm">Loading chart...</div>}>
            <ReactECharts
              option={option}
              theme="dark"
            style={{ height: "300px", width: "100%" }}
            notMerge={true}
            />
          </Suspense>
        </ChartErrorBoundary>
      </div>
    </div>
  );
}
