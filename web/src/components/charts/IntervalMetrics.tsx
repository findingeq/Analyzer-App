/**
 * IntervalMetrics Component
 * Displays clickable interval metrics below the chart, aligned with intervals
 */

import { useCallback, useMemo } from "react";
import { useRunStore } from "@/store/use-run-store";
import { IntervalStatus, RunType } from "@/lib/api-types";

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
    zoomStart,
    zoomEnd,
    resetZoom,
    runType,
  } = useRunStore();

  // Expected drift thresholds based on run type
  const expectedDriftThreshold = runType === RunType.VT1_STEADY ? 0.3 : 1.0;
  const splitSlopeThreshold = 1.2;

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

  // Check if currently zoomed
  const isZoomed = zoomStart > 0.1 || zoomEnd < 99.9;

  // Click handler - select interval and zoom to it, or zoom out if already selected and zoomed
  const handleIntervalClick = useCallback(
    (intervalNum: number) => {
      if (!analysisResult) return;

      // If clicking the already selected interval and zoomed, zoom out
      if (selectedIntervalId === intervalNum && isZoomed) {
        resetZoom();
        return;
      }

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
    [analysisResult, setSelectedInterval, setZoomRange, selectedIntervalId, isZoomed, resetZoom]
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
        const isInactive = isZoomed && !isSelected;

        if (!position) return null;

        // Color logic for metrics based on classification thresholds
        // Avg VE: Green if CUSUM not triggered OR recovered; Red if triggered and not recovered
        const hasUnrecoveredAlarm = result.alarm_time !== null && result.alarm_time !== undefined && !result.cusum_recovered;
        const avgVeColor = hasUnrecoveredAlarm ? "text-red-400" : "text-emerald-400";

        // Slope: Green if < expected drift; Red if >= expected drift
        const slopeColor = Math.abs(result.ve_drift_pct) >= expectedDriftThreshold ? "text-red-400" : "text-emerald-400";

        // Split slope ratio: Green if < 1.2x; Red if >= 1.2x
        const splitRatio = result.split_slope_ratio;
        const splitRatioColor = splitRatio !== null && splitRatio !== undefined && splitRatio >= splitSlopeThreshold
          ? "text-red-400"
          : "text-emerald-400";

        return (
          <button
            key={result.interval_num}
            onClick={() => handleIntervalClick(result.interval_num)}
            className={`
              absolute top-2 transform -translate-x-1/2
              rounded-lg border-2 px-2 py-1.5 transition-all cursor-pointer
              ${colors.bg} ${colors.border}
              ${isSelected ? "ring-2 ring-white/30 scale-105 z-10" : "hover:scale-105 hover:brightness-110 hover:z-10"}
              ${isInactive ? "opacity-50 brightness-75" : ""}
            `}
            style={{
              left: `${position.centerPct}%`,
              maxWidth: `${Math.max(position.widthPct * 0.9, 10)}%`,
              minWidth: "80px",
            }}
          >
            <div className="text-xs font-medium text-zinc-300">
              Int {result.interval_num}
            </div>
            <div className={`text-sm font-semibold ${avgVeColor}`}>
              {result.avg_ve.toFixed(1)} L/min
            </div>
            <div className={`text-xs ${slopeColor}`}>
              {result.ve_drift_pct >= 0 ? "+" : ""}
              {result.ve_drift_pct.toFixed(2)}%/min
            </div>
            {splitRatio !== null && splitRatio !== undefined && (
              <div className={`text-xs ${splitRatioColor}`}>
                {splitRatio.toFixed(2)}x
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default IntervalMetrics;
