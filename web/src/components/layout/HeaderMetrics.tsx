/**
 * HeaderMetrics Component
 * Displays run type format and cumulative drift in the header
 */

import { useRunStore } from "@/store/use-run-store";
import { RunType } from "@/lib/api-types";
import { Button } from "@/components/ui/button";

export function HeaderMetrics() {
  const { analysisResult, runType, numIntervals, intervalDurationMin, resetZoom, zoomStart, zoomEnd } =
    useRunStore();

  // Format run type display (e.g., "VT2 Intervals 4×10")
  const formatRunType = () => {
    if (!runType) return null;

    if (runType === RunType.VT1_STEADY) {
      return `VT1 Steady ${intervalDurationMin}min`;
    }

    return `VT2 Intervals ${numIntervals}×${intervalDurationMin}`;
  };

  // Check if zoomed
  const isZoomed = zoomStart > 0.1 || zoomEnd < 99.9;

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
    <div className="flex items-center gap-4">
      {/* Run Type Format */}
      <div className="text-sm font-medium text-foreground">{formatRunType()}</div>

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

      {/* Spacer */}
      <div className="flex-1" />

      {/* Reset Zoom Button */}
      {isZoomed && (
        <Button variant="outline" size="sm" onClick={resetZoom}>
          Reset Zoom
        </Button>
      )}
    </div>
  );
}

export default HeaderMetrics;
