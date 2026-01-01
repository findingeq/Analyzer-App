/**
 * HeaderMetrics Component
 * Displays run type format and cumulative drift in the header
 */

import { useRunStore } from "@/store/use-run-store";
import { RunType } from "@/lib/api-types";

export function HeaderMetrics() {
  const { analysisResult, runType, numIntervals, intervalDurationMin } =
    useRunStore();

  // Format run type display (e.g., "VT2 Intervals 4×10")
  const formatRunType = () => {
    if (!runType) return null;

    if (runType === RunType.VT1_STEADY) {
      return `VT1 Steady ${intervalDurationMin}min`;
    }

    return `VT2 Intervals ${numIntervals}×${intervalDurationMin}`;
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

  const averageVE = calculateAverageVE();
  const averagePace = calculateAveragePace();

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

      {/* Cumulative Drift (VT2 only) */}
      {cumulativeDrift && (
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
          <span className="text-xs text-muted-foreground">
            (p={cumulativeDrift.pvalue.toFixed(3)})
          </span>
        </div>
      )}
    </div>
  );
}

export default HeaderMetrics;
