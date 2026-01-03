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
    advancedParams,
  } = useRunStore();

  // Get expected drift threshold based on run type (from calibrated params)
  const expectedDriftThreshold = runType === RunType.MODERATE
    ? advancedParams.expectedDriftVt1
    : advancedParams.expectedDriftVt2;

  // Split slope ratio threshold from calibrated params (only used for Heavy/Severe)
  const splitSlopeThreshold = advancedParams.splitRatioVt2;

  // Moderate runs don't show split slope metrics
  const showSplitSlope = runType !== RunType.MODERATE;

  // Calculate interval positions as percentages (based on interval range, not breath data)
  const intervalPositions = useMemo(() => {
    if (!analysisResult || !analysisResult.intervals.length) return [];

    const intervals = analysisResult.intervals;
    // Use interval range for positioning (matches chart x-axis)
    const startTime = intervals[0].start_time;
    const endTime = intervals[intervals.length - 1].end_time;
    const totalDuration = endTime - startTime;

    if (totalDuration <= 0) return [];

    return intervals.map((interval) => {
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
      const intervals = analysisResult.intervals;
      const interval = intervals.find((i) => i.interval_num === intervalNum);

      if (interval && intervals.length > 0) {
        // Use interval range for zoom calculation (matches chart x-axis)
        const xAxisStart = intervals[0].start_time;
        const xAxisEnd = intervals[intervals.length - 1].end_time;
        const totalDuration = xAxisEnd - xAxisStart;

        // Find the next interval to include recovery period
        const nextInterval = intervals.find((i) => i.interval_num === intervalNum + 1);
        const zoomEndTime = nextInterval ? nextInterval.start_time : interval.end_time;

        // Calculate zoom range - show only this interval and recovery after it
        const startPct = ((interval.start_time - xAxisStart) / totalDuration) * 100;
        const endPct = ((zoomEndTime - xAxisStart) / totalDuration) * 100;

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

        // Overall slope: binary color based on expected drift threshold (calibrated)
        // Green if below threshold, Red if at or above threshold
        const overallSlope = result.ve_drift_pct;
        const slopeAboveThreshold = overallSlope !== null && overallSlope !== undefined
          && overallSlope >= expectedDriftThreshold;
        const slopeColor = slopeAboveThreshold ? "text-red-400" : "text-emerald-400";

        // Split slope ratio: Green if <= 1.2x; Red if > 1.2x
        const splitRatio = result.split_slope_ratio;
        const splitRatioTriggered = splitRatio !== null && splitRatio !== undefined
          && splitRatio >= splitSlopeThreshold;
        const splitRatioColor = splitRatioTriggered ? "text-red-400" : "text-emerald-400";

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
            {overallSlope !== null && overallSlope !== undefined && !result.is_ceiling_based && (
              <div className={`text-xs ${slopeColor}`}>
                {overallSlope >= 0 ? "+" : ""}{overallSlope.toFixed(2)}%/min
              </div>
            )}
            {showSplitSlope && splitRatio !== null && splitRatio !== undefined && (
              <div className={`text-xs ${splitRatioColor}`}>
                {splitRatioTriggered ? ">1.2x" : "≤1.2x"}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default IntervalMetrics;
