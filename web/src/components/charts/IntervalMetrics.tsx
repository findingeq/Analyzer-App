/**
 * IntervalMetrics Component
 * Displays clickable interval metrics below the chart, aligned with intervals
 */

import { useCallback, useMemo } from "react";
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

// Chart grid margins (must match MainChart)
const CHART_LEFT_MARGIN = 60;
const CHART_RIGHT_MARGIN = 60;

// =============================================================================
// Component
// =============================================================================

export function IntervalMetrics() {
  const {
    analysisResult,
    selectedIntervalId,
    showCusum,
    setSelectedInterval,
    setZoomRange,
  } = useRunStore();

  // Calculate interval positions as percentages
  const intervalPositions = useMemo(() => {
    if (!analysisResult || !analysisResult.intervals.length) return [];

    const breathData = analysisResult.breath_data;
    if (!breathData.times.length) return [];

    const startTime = breathData.times[0];
    const endTime = breathData.times[breathData.times.length - 1];
    const totalDuration = endTime - startTime;

    if (totalDuration <= 0) return [];

    return analysisResult.intervals.map((interval) => {
      // Calculate center position as percentage of total duration
      const intervalCenter = (interval.start_time + interval.end_time) / 2;
      const centerPct = ((intervalCenter - startTime) / totalDuration) * 100;

      // Calculate width as percentage
      const widthPct = ((interval.end_time - interval.start_time) / totalDuration) * 100;

      return {
        intervalNum: interval.interval_num,
        centerPct,
        widthPct,
      };
    });
  }, [analysisResult]);

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

  const rightMargin = showCusum ? CHART_RIGHT_MARGIN : 20;

  return (
    <div
      className="relative h-24 overflow-hidden"
      style={{
        marginLeft: CHART_LEFT_MARGIN,
        marginRight: rightMargin,
      }}
    >
      {analysisResult.results.map((result) => {
        const colors = STATUS_COLORS[result.status] || STATUS_COLORS[IntervalStatus.BELOW_THRESHOLD];
        const isSelected = selectedIntervalId === result.interval_num;
        const position = intervalPositions.find((p) => p.intervalNum === result.interval_num);

        if (!position) return null;

        return (
          <button
            key={result.interval_num}
            onClick={() => handleIntervalClick(result.interval_num)}
            className={`
              absolute top-2 transform -translate-x-1/2
              rounded-lg border-2 px-2 py-1.5 transition-all cursor-pointer
              ${colors.bg} ${colors.border}
              ${isSelected ? "ring-2 ring-white/30 scale-105 z-10" : "hover:scale-105 hover:brightness-110 hover:z-10"}
            `}
            style={{
              left: `${position.centerPct}%`,
              maxWidth: `${Math.max(position.widthPct * 0.9, 10)}%`,
              minWidth: "70px",
            }}
          >
            <div className="text-xs font-medium text-zinc-300">
              Int {result.interval_num}
            </div>
            <div className={`text-sm font-semibold ${colors.text}`}>
              {result.avg_ve.toFixed(1)} L/min
            </div>
            <div className="text-xs text-zinc-400">
              {result.ve_drift_pct >= 0 ? "+" : ""}
              {result.ve_drift_pct.toFixed(2)}%/min
            </div>
          </button>
        );
      })}
    </div>
  );
}

export default IntervalMetrics;
