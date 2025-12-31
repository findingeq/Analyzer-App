/**
 * IntervalMetrics Component
 * Displays clickable interval metrics below the chart
 */

import { useCallback } from "react";
import { useRunStore } from "@/store/use-run-store";
import { IntervalStatus } from "@/lib/api-types";

// =============================================================================
// Color Theme (matches MainChart)
// =============================================================================

const STATUS_COLORS = {
  [IntervalStatus.BELOW_THRESHOLD]: {
    bg: "bg-emerald-500/20",
    border: "border-emerald-500",
    text: "text-emerald-400",
  },
  [IntervalStatus.BORDERLINE]: {
    bg: "bg-amber-500/20",
    border: "border-amber-500",
    text: "text-amber-400",
  },
  [IntervalStatus.ABOVE_THRESHOLD]: {
    bg: "bg-red-500/20",
    border: "border-red-500",
    text: "text-red-400",
  },
};

// =============================================================================
// Component
// =============================================================================

export function IntervalMetrics() {
  const {
    analysisResult,
    selectedIntervalId,
    setSelectedInterval,
    setZoomRange,
  } = useRunStore();

  // Click handler - select interval and zoom to it
  const handleIntervalClick = useCallback(
    (intervalNum: number) => {
      if (!analysisResult) return;

      // Select the interval
      setSelectedInterval(intervalNum);

      // Find interval bounds for zooming
      const interval = analysisResult.intervals.find(
        (i) => i.interval_num === intervalNum
      );
      const breathData = analysisResult.breath_data;

      if (interval && breathData.times.length > 0) {
        const totalDuration =
          breathData.times[breathData.times.length - 1] - breathData.times[0];
        const padding = totalDuration * 0.02; // 2% padding

        // Calculate zoom range as percentages
        const startPct =
          ((interval.start_time - breathData.times[0] - padding) /
            totalDuration) *
          100;
        const endPct =
          ((interval.end_time - breathData.times[0] + padding) / totalDuration) *
          100;

        setZoomRange(Math.max(0, startPct), Math.min(100, endPct));
      }
    },
    [analysisResult, setSelectedInterval, setZoomRange]
  );

  // Empty state
  if (!analysisResult || !analysisResult.results.length) {
    return null;
  }

  return (
    <div className="flex gap-2 p-3 overflow-x-auto">
      {analysisResult.results.map((result) => {
        const colors = STATUS_COLORS[result.status] || STATUS_COLORS[IntervalStatus.BELOW_THRESHOLD];
        const isSelected = selectedIntervalId === result.interval_num;

        return (
          <button
            key={result.interval_num}
            onClick={() => handleIntervalClick(result.interval_num)}
            className={`
              flex-shrink-0 rounded-lg border-2 p-3 transition-all cursor-pointer
              ${colors.bg} ${colors.border}
              ${isSelected ? "ring-2 ring-white/30 scale-105" : "hover:scale-102 hover:brightness-110"}
            `}
          >
            <div className="text-xs font-medium text-zinc-300 mb-1">
              Int {result.interval_num}
            </div>
            <div className="space-y-0.5 text-left">
              <div className={`text-sm font-semibold ${colors.text}`}>
                {result.avg_ve.toFixed(1)} L/min
              </div>
              <div className="text-xs text-zinc-400">
                {result.ve_drift_pct >= 0 ? "+" : ""}
                {result.ve_drift_pct.toFixed(2)}%/min
              </div>
              {result.split_slope_ratio !== null &&
                result.split_slope_ratio !== undefined && (
                  <div className="text-xs text-zinc-500">
                    ×{result.split_slope_ratio.toFixed(1)} split
                  </div>
                )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

export default IntervalMetrics;
