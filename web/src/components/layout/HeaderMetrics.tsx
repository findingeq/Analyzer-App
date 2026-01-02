/**
 * HeaderMetrics Component
 * Displays run type format and cumulative drift in the header
 * Shows additional interval-specific metrics when zoomed
 */

import { useRunStore } from "@/store/use-run-store";
import { RunType } from "@/lib/api-types";

export function HeaderMetrics() {
  const {
    analysisResult,
    runType,
    numIntervals,
    intervalDurationMin,
    recoveryDurationMin,
    selectedIntervalId,
    zoomStart,
    zoomEnd,
  } = useRunStore();

  // Check if we're zoomed in (not at default 0-100 range)
  const isZoomed = zoomStart !== 0 || zoomEnd !== 100;

  // Get selected interval result
  const selectedInterval = analysisResult?.results?.find(
    (r) => r.interval_num === selectedIntervalId
  );

  // Format run type display (e.g., "Heavy Intervals 4×10")
  const formatRunType = () => {
    if (!runType) return null;

    // Round duration to nearest integer for display
    const durationDisplay = Math.round(intervalDurationMin);

    if (runType === RunType.VT1_STEADY) {
      // Moderate can now have intervals too
      if (numIntervals > 1) {
        return `Moderate Intervals ${numIntervals}×${durationDisplay}`;
      }
      return `Moderate ${durationDisplay}min`;
    }

    if (runType === RunType.SEVERE) {
      return `Severe Intervals ${numIntervals}×${durationDisplay}`;
    }

    return `Heavy Intervals ${numIntervals}×${durationDisplay}`;
  };

  // Calculate average VE across all intervals
  const calculateAverageVE = () => {
    if (!analysisResult?.results?.length) return null;
    const totalVE = analysisResult.results.reduce((sum, r) => sum + r.avg_ve, 0);
    return totalVE / analysisResult.results.length;
  };

  // Calculate average pace (min/mile) from interval speeds
  const calculateAveragePace = () => {
    if (!analysisResult?.results?.length) return null;

    // Get work intervals with valid speeds
    const speeds = analysisResult.results
      .filter((r) => r.speed !== null && r.speed !== undefined && r.speed > 0)
      .map((r) => r.speed as number);

    if (speeds.length === 0) return null;

    // Average speed in mph
    const avgSpeedMph = speeds.reduce((sum, s) => sum + s, 0) / speeds.length;

    // Convert to min/mile
    if (avgSpeedMph <= 0) return null;
    return 60 / avgSpeedMph;
  };

  // Format pace as "mm:ss"
  const formatPace = (paceMinPerMile: number) => {
    const mins = Math.floor(paceMinPerMile);
    const secs = Math.round((paceMinPerMile - mins) * 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Calculate selected interval's pace
  const calculateSelectedIntervalPace = () => {
    if (!selectedInterval?.speed || selectedInterval.speed <= 0) return null;
    return 60 / selectedInterval.speed;
  };

  // Calculate selected interval's duration in minutes
  const calculateSelectedIntervalDuration = () => {
    if (!selectedInterval) return null;
    return (selectedInterval.end_time - selectedInterval.start_time) / 60;
  };

  const averageVE = calculateAverageVE();
  const averagePace = calculateAveragePace();
  const selectedIntervalPace = calculateSelectedIntervalPace();
  const selectedIntervalDuration = calculateSelectedIntervalDuration();

  // No analysis yet
  if (!analysisResult) {
    return (
      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <span>Upload a CSV file to begin</span>
      </div>
    );
  }

  const cumulativeDrift = analysisResult.cumulative_drift;

  return (
    <div className="flex items-center gap-6">
      {/* Run Type Format */}
      <div className="text-sm font-medium text-foreground">{formatRunType()}</div>

      {/* Average Pace (if speed data available) */}
      {averagePace !== null && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Avg Pace:</span>
          <span className="font-medium text-zinc-300">
            {formatPace(averagePace)} /mi
          </span>
        </div>
      )}

      {/* Average VE across all intervals */}
      {averageVE !== null && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Avg VE:</span>
          <span className="font-medium text-zinc-300">
            {averageVE.toFixed(1)} L/min
          </span>
        </div>
      )}

      {/* Cumulative Drift (VT2 only) - hide when zoomed */}
      {cumulativeDrift && !isZoomed && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Cumulative Drift:</span>
          <span
            className={`font-medium ${
              cumulativeDrift.slope_pct > 0
                ? "text-red-400"
                : cumulativeDrift.slope_pct < -0.5
                  ? "text-emerald-400"
                  : "text-zinc-300"
            }`}
          >
            {cumulativeDrift.slope_pct >= 0 ? "+" : ""}
            {cumulativeDrift.slope_pct.toFixed(2)}%/min
          </span>
        </div>
      )}

      {/* Zoomed interval metrics */}
      {isZoomed && selectedInterval && (
        <>
          {/* Interval indicator */}
          <div className="flex items-center gap-2 text-sm border-l border-zinc-700 pl-4">
            <span className="text-primary font-medium">
              Interval {selectedIntervalId}
            </span>
          </div>

          {/* Interval Duration */}
          {selectedIntervalDuration !== null && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Duration:</span>
              <span className="font-medium text-zinc-300">
                {selectedIntervalDuration.toFixed(1)} min
              </span>
            </div>
          )}

          {/* Recovery Duration (VT2/SEVERE only) */}
          {(runType === RunType.VT2_INTERVAL || runType === RunType.SEVERE) && recoveryDurationMin > 0 && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Recovery:</span>
              <span className="font-medium text-zinc-300">
                {recoveryDurationMin} min
              </span>
            </div>
          )}

          {/* Interval Pace */}
          {selectedIntervalPace !== null && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Pace:</span>
              <span className="font-medium text-zinc-300">
                {formatPace(selectedIntervalPace)} /mi
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default HeaderMetrics;
