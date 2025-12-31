/**
 * Sidebar Component
 * File upload, run configuration, and interval results
 */

import { useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useRunStore } from "@/store/use-run-store";
import { RunType, IntervalStatus } from "@/lib/api-types";
import { parseCSV, detectIntervals, runAnalysis, readFileAsText } from "@/lib/client";
import { formatTime } from "@/lib/utils";

export function Sidebar() {
  const {
    csvContent,
    csvMetadata,
    fileName,
    runType,
    numIntervals,
    intervalDurationMin,
    recoveryDurationMin,
    analysisResult,
    isAnalyzing,
    selectedIntervalId,
    showSlopeLines,
    showCusum,
    setCSVContent,
    setCSVMetadata,
    setRunType,
    setNumIntervals,
    setIntervalDuration,
    setRecoveryDuration,
    setAnalysisResult,
    setIsAnalyzing,
    setSelectedInterval,
    toggleSlopeLines,
    toggleCusum,
    reset,
  } = useRunStore();

  // Parse CSV mutation
  const parseMutation = useMutation({
    mutationFn: async (file: File) => {
      const content = await readFileAsText(file);
      setCSVContent(content, file.name);
      const metadata = await parseCSV({ csv_content: content });
      const intervals = await detectIntervals({ csv_content: content });
      return { metadata, intervals };
    },
    onSuccess: ({ metadata, intervals }) => {
      setCSVMetadata(metadata);
      setRunType(intervals.run_type);
      setNumIntervals(intervals.num_intervals);
      setIntervalDuration(intervals.interval_duration_min);
      setRecoveryDuration(intervals.recovery_duration_min);
    },
  });

  // Run analysis mutation
  const analysisMutation = useMutation({
    mutationFn: async () => {
      if (!csvContent || !runType) throw new Error("Missing data");
      setIsAnalyzing(true);
      const result = await runAnalysis({
        csv_content: csvContent,
        run_type: runType,
        num_intervals: numIntervals,
        interval_duration_min: intervalDurationMin,
        recovery_duration_min: recoveryDurationMin,
      });
      return result;
    },
    onSuccess: (result) => {
      setAnalysisResult(result);
      setIsAnalyzing(false);
    },
    onError: () => {
      setIsAnalyzing(false);
    },
  });

  // File upload handler
  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        parseMutation.mutate(file);
      }
    },
    [parseMutation]
  );

  // Get status badge variant
  const getStatusVariant = (status: IntervalStatus) => {
    switch (status) {
      case IntervalStatus.BELOW_THRESHOLD:
        return "success";
      case IntervalStatus.BORDERLINE:
        return "warning";
      case IntervalStatus.ABOVE_THRESHOLD:
        return "danger";
      default:
        return "default";
    }
  };

  return (
    <div className="space-y-4">
      {/* File Upload */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Upload CSV</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="csv-upload"
            />
            <label htmlFor="csv-upload">
              <Button
                variant="outline"
                className="w-full cursor-pointer"
                asChild
              >
                <span>
                  {fileName ? fileName : "Choose file..."}
                </span>
              </Button>
            </label>
            {parseMutation.isPending && (
              <p className="text-xs text-muted-foreground">Parsing...</p>
            )}
            {csvMetadata && (
              <div className="text-xs text-muted-foreground">
                <p>Format: {csvMetadata.format}</p>
                <p>Breaths: {csvMetadata.total_breaths}</p>
                <p>Duration: {formatTime(csvMetadata.duration_seconds)}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Run Configuration */}
      {csvContent && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Run Type */}
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">Run Type</label>
              <Select
                value={runType ?? undefined}
                onValueChange={(v) => setRunType(v as RunType)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select run type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={RunType.VT1_STEADY}>VT1 (Steady State)</SelectItem>
                  <SelectItem value={RunType.VT2_INTERVAL}>VT2 (Intervals)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Intervals */}
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">
                Intervals
              </label>
              <Select
                value={numIntervals.toString()}
                onValueChange={(v) => setNumIntervals(parseInt(v))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 2, 3, 4, 5, 6].map((n) => (
                    <SelectItem key={n} value={n.toString()}>
                      {n} interval{n > 1 ? "s" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Interval Duration */}
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">
                Interval Duration (min)
              </label>
              <Select
                value={intervalDurationMin.toString()}
                onValueChange={(v) => setIntervalDuration(parseFloat(v))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[4, 5, 6, 7, 8, 10, 12, 15, 20, 30].map((n) => (
                    <SelectItem key={n} value={n.toString()}>
                      {n} min
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Recovery Duration (for VT2) */}
            {runType === RunType.VT2_INTERVAL && numIntervals > 1 && (
              <div className="space-y-2">
                <label className="text-xs text-muted-foreground">
                  Recovery Duration (min)
                </label>
                <Select
                  value={recoveryDurationMin.toString()}
                  onValueChange={(v) => setRecoveryDuration(parseFloat(v))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[0, 1, 2, 3, 4, 5].map((n) => (
                      <SelectItem key={n} value={n.toString()}>
                        {n} min
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Run Analysis Button */}
            <Button
              onClick={() => analysisMutation.mutate()}
              disabled={isAnalyzing || !runType}
              className="w-full"
            >
              {isAnalyzing ? "Analyzing..." : "Run Analysis"}
            </Button>

            {analysisMutation.isError && (
              <p className="text-xs text-destructive">
                Error: {analysisMutation.error.message}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Chart Options */}
      {analysisResult && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Chart Options</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs text-muted-foreground">
                Show Slope Lines
              </label>
              <Switch checked={showSlopeLines} onCheckedChange={toggleSlopeLines} />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs text-muted-foreground">
                Show CUSUM
              </label>
              <Switch checked={showCusum} onCheckedChange={toggleCusum} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {analysisResult && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {analysisResult.results.map((result) => (
              <div
                key={result.interval_num}
                className={`cursor-pointer rounded-md border p-3 transition-colors ${
                  selectedIntervalId === result.interval_num
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-muted-foreground"
                }`}
                onClick={() => setSelectedInterval(result.interval_num)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    Interval {result.interval_num}
                  </span>
                  <Badge variant={getStatusVariant(result.status)}>
                    {result.status === IntervalStatus.BELOW_THRESHOLD
                      ? "Below"
                      : result.status === IntervalStatus.BORDERLINE
                        ? "Borderline"
                        : "Above"}
                  </Badge>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-x-4 text-xs text-muted-foreground">
                  <div>VE Drift: {result.ve_drift_pct.toFixed(2)}%/min</div>
                  <div>Peak CUSUM: {result.peak_cusum.toFixed(1)}</div>
                  {result.speed && <div>Speed: {result.speed.toFixed(1)} mph</div>}
                </div>
              </div>
            ))}

            {/* Cumulative Drift (VT2) */}
            {analysisResult.cumulative_drift && (
              <>
                <Separator className="my-3" />
                <div className="text-xs">
                  <p className="font-medium">Cumulative Drift</p>
                  <p className="text-muted-foreground">
                    {analysisResult.cumulative_drift.slope_pct.toFixed(2)}%/min
                    (p={analysisResult.cumulative_drift.pvalue.toFixed(3)})
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Reset Button */}
      {csvContent && (
        <Button variant="ghost" onClick={reset} className="w-full">
          Reset
        </Button>
      )}
    </div>
  );
}

export default Sidebar;
