/**
 * Sidebar Component
 * File upload, cloud session selection, run configuration
 */

import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useRunStore } from "@/store/use-run-store";
import { RunType } from "@/lib/api-types";
import {
  parseCSV,
  detectIntervals,
  runAnalysis,
  readFileAsText,
  listSessions,
  getSession,
  type SessionInfo,
} from "@/lib/client";
import { formatTime } from "@/lib/utils";
import {
  Upload,
  Cloud,
  Settings,
  BarChart3,
  Eye,
  Sliders,
  ChevronDown,
  RotateCcw,
  Play,
} from "lucide-react";

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
    showSlopeLines,
    showCusum,
    dataSource,
    sidebarCollapsed,
    vt1Ceiling,
    vt2Ceiling,
    useThresholdsForAll,
    advancedParams,
    setCSVContent,
    setCSVMetadata,
    setRunType,
    setNumIntervals,
    setIntervalDuration,
    setRecoveryDuration,
    setAnalysisResult,
    setIsAnalyzing,
    toggleSlopeLines,
    toggleCusum,
    setDataSource,
    setVt1Ceiling,
    setVt2Ceiling,
    setUseThresholdsForAll,
    setAdvancedParams,
    reset,
  } = useRunStore();

  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Fetch cloud sessions
  const sessionsQuery = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
    enabled: dataSource === "cloud",
    retry: false,
  });

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

  // Load cloud session mutation
  const loadSessionMutation = useMutation({
    mutationFn: async (sessionId: string) => {
      const session = await getSession(sessionId);
      setCSVContent(session.csv_content, sessionId);
      const metadata = await parseCSV({ csv_content: session.csv_content });
      const intervals = await detectIntervals({ csv_content: session.csv_content });
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
        params: {
          vt1_ve_ceiling: vt1Ceiling,
          vt2_ve_ceiling: vt2Ceiling,
          use_thresholds_for_all: useThresholdsForAll,
          phase3_onset_override: advancedParams.phase3OnsetOverride,
          max_drift_pct_vt1: advancedParams.maxDriftVt1,
          max_drift_pct_vt2: advancedParams.maxDriftVt2,
          h_multiplier_vt1: advancedParams.hMultiplierVt1,
          h_multiplier_vt2: advancedParams.hMultiplierVt2,
          sigma_pct_vt1: advancedParams.sigmaPctVt1,
          sigma_pct_vt2: advancedParams.sigmaPctVt2,
          expected_drift_pct_vt1: advancedParams.expectedDriftVt1,
          expected_drift_pct_vt2: advancedParams.expectedDriftVt2,
        },
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

  // Handle session selection
  const handleSessionSelect = useCallback(
    (sessionId: string) => {
      setSelectedSessionId(sessionId);
      loadSessionMutation.mutate(sessionId);
    },
    [loadSessionMutation]
  );

  // Reset session selection when switching data source
  useEffect(() => {
    setSelectedSessionId(null);
  }, [dataSource]);

  // Format session date for display
  const formatSessionDate = (isoDate: string) => {
    try {
      const date = new Date(isoDate);
      return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return isoDate;
    }
  };

  // Collapsed mode: show icon buttons
  if (sidebarCollapsed) {
    return (
      <div className="flex flex-col items-center gap-3 pt-2">
        {/* Data Source Icons */}
        <Button
          variant={dataSource === "local" ? "default" : "ghost"}
          size="icon"
          onClick={() => setDataSource("local")}
          title="Local File"
        >
          <Upload className="h-4 w-4" />
        </Button>
        <Button
          variant={dataSource === "cloud" ? "default" : "ghost"}
          size="icon"
          onClick={() => setDataSource("cloud")}
          title="Cloud Sessions"
        >
          <Cloud className="h-4 w-4" />
        </Button>

        <div className="w-6 border-t border-border" />

        {/* Config Icon */}
        {csvContent && (
          <Button
            variant="ghost"
            size="icon"
            title="Configuration"
            disabled
          >
            <Settings className="h-4 w-4" />
          </Button>
        )}

        {/* Run Analysis Icon */}
        {csvContent && runType && (
          <Button
            variant="default"
            size="icon"
            onClick={() => analysisMutation.mutate()}
            disabled={isAnalyzing}
            title="Run Analysis"
          >
            <Play className="h-4 w-4" />
          </Button>
        )}

        <div className="w-6 border-t border-border" />

        {/* Chart Options Icons */}
        {analysisResult && (
          <>
            <Button
              variant={showSlopeLines ? "default" : "ghost"}
              size="icon"
              onClick={toggleSlopeLines}
              title="Show Slope Lines"
            >
              <BarChart3 className="h-4 w-4" />
            </Button>
            <Button
              variant={showCusum ? "default" : "ghost"}
              size="icon"
              onClick={toggleCusum}
              title="Show CUSUM"
            >
              <Eye className="h-4 w-4" />
            </Button>
          </>
        )}

        <div className="w-6 border-t border-border" />

        {/* VT Thresholds Icon */}
        <Button
          variant="ghost"
          size="icon"
          title={`VT1: ${vt1Ceiling} / VT2: ${vt2Ceiling}`}
          disabled
        >
          <Sliders className="h-4 w-4" />
        </Button>

        {/* Reset Icon */}
        {csvContent && (
          <Button
            variant="ghost"
            size="icon"
            onClick={reset}
            title="Reset"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        )}
      </div>
    );
  }

  // Expanded mode: full sidebar
  return (
    <div className="space-y-4">
      {/* Data Source Toggle */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Data Source</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              variant={dataSource === "local" ? "default" : "outline"}
              size="sm"
              className="flex-1"
              onClick={() => setDataSource("local")}
            >
              Local File
            </Button>
            <Button
              variant={dataSource === "cloud" ? "default" : "outline"}
              size="sm"
              className="flex-1"
              onClick={() => setDataSource("cloud")}
            >
              Cloud
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Local File Upload */}
      {dataSource === "local" && (
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
                  <span>{fileName ? fileName : "Choose file..."}</span>
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
      )}

      {/* Cloud Sessions */}
      {dataSource === "cloud" && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Cloud Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {sessionsQuery.isLoading && (
                <p className="text-xs text-muted-foreground">Loading sessions...</p>
              )}
              {sessionsQuery.isError && (
                <p className="text-xs text-destructive">
                  Cloud storage not available. Firebase may not be configured.
                </p>
              )}
              {sessionsQuery.isSuccess && sessionsQuery.data.length === 0 && (
                <p className="text-xs text-muted-foreground">No sessions found.</p>
              )}
              {sessionsQuery.isSuccess && sessionsQuery.data.length > 0 && (
                <Select
                  value={selectedSessionId ?? undefined}
                  onValueChange={handleSessionSelect}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a session..." />
                  </SelectTrigger>
                  <SelectContent>
                    {sessionsQuery.data.map((session: SessionInfo) => (
                      <SelectItem key={session.session_id} value={session.session_id}>
                        <div className="flex flex-col">
                          <span className="font-medium">{session.filename}</span>
                          <span className="text-xs text-muted-foreground">
                            {formatSessionDate(session.uploaded_at)}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {loadSessionMutation.isPending && (
                <p className="text-xs text-muted-foreground">Loading session...</p>
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
      )}

      {/* VT Thresholds */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">VT Thresholds</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">VT1 Ceiling</label>
              <Input
                type="number"
                step="0.1"
                value={vt1Ceiling}
                onChange={(e) => setVt1Ceiling(parseFloat(e.target.value) || 0)}
                className="h-8"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">VT2 Ceiling</label>
              <Input
                type="number"
                step="0.1"
                value={vt2Ceiling}
                onChange={(e) => setVt2Ceiling(parseFloat(e.target.value) || 0)}
                className="h-8"
              />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <label className="text-xs text-muted-foreground">Use Thresholds for All</label>
            <Switch
              checked={useThresholdsForAll}
              onCheckedChange={setUseThresholdsForAll}
            />
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
              <label className="text-xs text-muted-foreground">Intervals</label>
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
              <label className="text-xs text-muted-foreground">Show CUSUM</label>
              <Switch checked={showCusum} onCheckedChange={toggleCusum} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Advanced CUSUM Parameters */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="pb-3 cursor-pointer hover:bg-muted/50 rounded-t-lg">
              <CardTitle className="text-sm flex items-center justify-between">
                Advanced
                <ChevronDown
                  className={`h-4 w-4 transition-transform ${
                    advancedOpen ? "rotate-180" : ""
                  }`}
                />
              </CardTitle>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-4 pt-0">
              {/* Ramp-up Override */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">
                  Ramp-up Period Override (sec)
                </label>
                <Input
                  type="number"
                  placeholder="Auto-detect"
                  value={advancedParams.phase3OnsetOverride ?? ""}
                  onChange={(e) =>
                    setAdvancedParams({
                      phase3OnsetOverride: e.target.value
                        ? parseFloat(e.target.value)
                        : null,
                    })
                  }
                  className="h-8"
                />
              </div>

              {/* H Multiplier */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">H (VT1)</label>
                  <Input
                    type="number"
                    step="0.5"
                    value={advancedParams.hMultiplierVt1}
                    onChange={(e) =>
                      setAdvancedParams({
                        hMultiplierVt1: parseFloat(e.target.value) || 5.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">H (VT2)</label>
                  <Input
                    type="number"
                    step="0.5"
                    value={advancedParams.hMultiplierVt2}
                    onChange={(e) =>
                      setAdvancedParams({
                        hMultiplierVt2: parseFloat(e.target.value) || 5.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
              </div>

              {/* Sigma % */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Sigma % (VT1)</label>
                  <Input
                    type="number"
                    step="0.5"
                    value={advancedParams.sigmaPctVt1}
                    onChange={(e) =>
                      setAdvancedParams({
                        sigmaPctVt1: parseFloat(e.target.value) || 7.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Sigma % (VT2)</label>
                  <Input
                    type="number"
                    step="0.5"
                    value={advancedParams.sigmaPctVt2}
                    onChange={(e) =>
                      setAdvancedParams({
                        sigmaPctVt2: parseFloat(e.target.value) || 4.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
              </div>

              {/* Expected Drift */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Exp. Drift % (VT1)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={advancedParams.expectedDriftVt1}
                    onChange={(e) =>
                      setAdvancedParams({
                        expectedDriftVt1: parseFloat(e.target.value) || 0.3,
                      })
                    }
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Exp. Drift % (VT2)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={advancedParams.expectedDriftVt2}
                    onChange={(e) =>
                      setAdvancedParams({
                        expectedDriftVt2: parseFloat(e.target.value) || 1.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
              </div>

              {/* Max Drift Thresholds */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Max Drift % (VT1)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={advancedParams.maxDriftVt1}
                    onChange={(e) =>
                      setAdvancedParams({
                        maxDriftVt1: parseFloat(e.target.value) || 1.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Max Drift % (VT2)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={advancedParams.maxDriftVt2}
                    onChange={(e) =>
                      setAdvancedParams({
                        maxDriftVt2: parseFloat(e.target.value) || 3.0,
                      })
                    }
                    className="h-8"
                  />
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

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
