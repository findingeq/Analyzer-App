/**
 * StartupScreen Component
 * Landing screen with intensity chart and run list
 */

import { useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { IntensityChart } from "./IntensityChart";
import { RunListTable } from "./RunListTable";
import { useRunStore } from "@/store/use-run-store";
import {
  listSessions,
  getSession,
  parseCSV,
  detectIntervals,
  deleteSession,
  getCalibrationParams,
  toggleCalibration,
  runAnalysis,
  updateCalibration,
  updateSessionAnalysis,
  updateSessionCalibration,
  getOrCreateUserId,
} from "@/lib/client";
import type { RunType } from "@/lib/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

export function StartupScreen() {
  const {
    setCSVContent,
    setCSVMetadata,
    setRunType,
    setNumIntervals,
    setIntervalDuration,
    setRecoveryDuration,
    setDataSource,
    setAnalysisResult,
    setIsAnalyzing,
    setPendingVEPrompt,
    vt1Ceiling,
    vt2Ceiling,
    useThresholdsForAll,
    advancedParams,
  } = useRunStore();

  const queryClient = useQueryClient();

  // Fetch calibration params
  const calibrationQuery = useQuery({
    queryKey: ["calibration-params"],
    queryFn: () => getCalibrationParams(),
    retry: false,
  });

  // Toggle calibration mutation
  const toggleCalibrationMutation = useMutation({
    mutationFn: (enabled: boolean) => toggleCalibration(enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["calibration-params"] });
    },
    onError: (error) => {
      console.error("Failed to toggle calibration:", error);
    },
  });

  // Fetch cloud sessions
  const sessionsQuery = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
    retry: false,
  });

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId, getOrCreateUserId()),
    onSuccess: () => {
      // Refetch sessions after delete
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      // Refresh calibration params in case deleted session contributed
      queryClient.invalidateQueries({ queryKey: ["calibration-params"] });
    },
    onError: (error) => {
      console.error("Failed to delete session:", error);
    },
  });

  // Load session mutation - loads CSV and immediately runs analysis
  const loadSessionMutation = useMutation({
    mutationFn: async (sessionId: string) => {
      const session = await getSession(sessionId);
      setCSVContent(session.csv_content, sessionId);
      const metadata = await parseCSV({ csv_content: session.csv_content });
      const intervals = await detectIntervals({ csv_content: session.csv_content });

      // Set metadata immediately
      setCSVMetadata(metadata);
      setRunType(intervals.run_type);
      setNumIntervals(intervals.num_intervals);
      setIntervalDuration(intervals.interval_duration_min);
      setRecoveryDuration(intervals.recovery_duration_min);
      setDataSource("cloud");

      // Automatically run analysis
      setIsAnalyzing(true);
      const analysisResult = await runAnalysis({
        csv_content: session.csv_content,
        run_type: intervals.run_type as RunType,
        num_intervals: intervals.num_intervals,
        interval_duration_min: intervals.interval_duration_min,
        recovery_duration_min: intervals.recovery_duration_min,
        params: {
          vt1_ve_ceiling: vt1Ceiling,
          vt2_ve_ceiling: vt2Ceiling,
          use_thresholds_for_all: useThresholdsForAll,
          phase3_onset_override: advancedParams.phase3OnsetOverride,
          max_drift_pct_vt2: advancedParams.maxDriftVt2,
          h_multiplier_vt1: advancedParams.hMultiplierVt1,
          h_multiplier_vt2: advancedParams.hMultiplierVt2,
          sigma_pct_vt1: advancedParams.sigmaPctVt1,
          sigma_pct_vt2: advancedParams.sigmaPctVt2,
          expected_drift_pct_vt1: advancedParams.expectedDriftVt1,
          expected_drift_pct_vt2: advancedParams.expectedDriftVt2,
        },
      });

      return { metadata, intervals, analysisResult, runType: intervals.run_type as RunType, sessionId };
    },
    onSuccess: async ({ analysisResult, runType, sessionId }) => {
      setAnalysisResult(analysisResult);
      setIsAnalyzing(false);

      // Check if session is excluded from calibration
      const sessionInfo = sessionsQuery.data?.find((s) => s.session_id === sessionId);
      const isExcluded = sessionInfo?.summary?.exclude_from_calibration ?? false;

      // Update calibration with results (only if not excluded)
      let calibrationContribution: { contributed: boolean; run_type?: string | null; sigma_pct?: number | null } | null = null;
      if (runType && analysisResult.results?.length > 0 && !isExcluded) {
        try {
          const calibrationResult = await updateCalibration(runType, analysisResult.results);
          if (calibrationResult.ve_prompt) {
            setPendingVEPrompt(calibrationResult.ve_prompt);
          }
          // Store contribution info to save to session metadata
          calibrationContribution = calibrationResult.contribution ?? null;
        } catch (error) {
          console.warn("Calibration update failed:", error);
        }
      }

      // Calculate average sigma and drift across intervals for session summary
      if (analysisResult.results?.length > 0) {
        const validSigmas = analysisResult.results
          .map((r) => r.observed_sigma_pct)
          .filter((v): v is number => v != null && !isNaN(v));
        const validDrifts = analysisResult.results
          .map((r) => r.ve_drift_pct)
          .filter((v): v is number => v != null && !isNaN(v));

        const avgSigma = validSigmas.length > 0
          ? validSigmas.reduce((a, b) => a + b, 0) / validSigmas.length
          : null;
        const avgDrift = validDrifts.length > 0
          ? validDrifts.reduce((a, b) => a + b, 0) / validDrifts.length
          : null;

        // Calculate analysis outcome (below, above, or mixed) using majority rule
        let analysisOutcome: string | null = null;
        const results = analysisResult.results;
        if (results.length === 1) {
          // Single interval: use direct classification
          const status = results[0].status;
          if (status === "BELOW_THRESHOLD") {
            analysisOutcome = "below";
          } else if (status === "ABOVE_THRESHOLD") {
            analysisOutcome = "above";
          } else {
            analysisOutcome = "mixed"; // BORDERLINE counts as mixed
          }
        } else {
          // Multiple intervals: use majority rule
          const totalIntervals = results.length;
          const belowCount = results.filter((r) => r.status === "BELOW_THRESHOLD").length;
          const aboveCount = results.filter((r) => r.status === "ABOVE_THRESHOLD").length;

          if (belowCount > totalIntervals / 2) {
            analysisOutcome = "below";
          } else if (aboveCount > totalIntervals / 2) {
            analysisOutcome = "above";
          } else {
            analysisOutcome = "mixed";
          }
        }

        // Save analysis results and calibration contribution to session metadata
        try {
          await updateSessionAnalysis(sessionId, avgSigma, avgDrift, calibrationContribution, analysisOutcome);
          // Refresh sessions list to show updated sigma/drift
          queryClient.invalidateQueries({ queryKey: ["sessions"] });
        } catch (error) {
          console.warn("Failed to save analysis results to session:", error);
        }
      }
    },
    onError: (error) => {
      console.error("Failed to load session:", error);
      setIsAnalyzing(false);
    },
  });

  const handleSelectSession = useCallback(
    (sessionId: string) => {
      loadSessionMutation.mutate(sessionId);
    },
    [loadSessionMutation]
  );

  const handleDeleteSession = useCallback(
    (sessionId: string) => {
      if (window.confirm("Delete this session? This cannot be undone.")) {
        deleteSessionMutation.mutate(sessionId);
      }
    },
    [deleteSessionMutation]
  );

  const handleCalibrationToggle = useCallback(
    (checked: boolean) => {
      toggleCalibrationMutation.mutate(checked);
    },
    [toggleCalibrationMutation]
  );

  // Toggle session calibration exclusion mutation
  const toggleSessionCalibrationMutation = useMutation({
    mutationFn: ({ sessionId, exclude }: { sessionId: string; exclude: boolean }) =>
      updateSessionCalibration(sessionId, exclude),
    onSuccess: () => {
      // Refresh sessions to show updated calibration status
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
    onError: (error) => {
      console.error("Failed to toggle session calibration:", error);
    },
  });

  const handleToggleSessionCalibration = useCallback(
    (sessionId: string, exclude: boolean) => {
      toggleSessionCalibrationMutation.mutate({ sessionId, exclude });
    },
    [toggleSessionCalibrationMutation]
  );

  // Safely get sessions array with validation
  const sessions = Array.isArray(sessionsQuery.data) ? sessionsQuery.data : [];
  const calibrationEnabled = calibrationQuery.data?.enabled ?? true;

  return (
    <div className="h-full flex flex-col p-6 gap-6 overflow-auto">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-lg font-semibold text-foreground mb-1">
          Welcome to VT Check
        </h2>
        <p className="text-sm text-muted-foreground">
          Select a session to analyze or upload a new CSV file
        </p>
      </div>

      {/* Calibration Toggle */}
      <div className="flex items-center justify-center gap-3">
        <Switch
          id="calibration-toggle"
          checked={calibrationEnabled}
          onCheckedChange={handleCalibrationToggle}
          disabled={toggleCalibrationMutation.isPending || calibrationQuery.isLoading}
        />
        <Label
          htmlFor="calibration-toggle"
          className="text-sm text-muted-foreground cursor-pointer"
        >
          ML Calibration {calibrationEnabled ? "On" : "Off"}
        </Label>
      </div>

      {/* Loading State */}
      {sessionsQuery.isLoading && (
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-sm">Loading sessions...</p>
        </div>
      )}

      {/* Error State */}
      {sessionsQuery.isError && (
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-sm">Cloud storage not available.</p>
          <p className="text-xs mt-1">Upload a local CSV file to get started.</p>
        </div>
      )}

      {/* Content */}
      {!sessionsQuery.isLoading && !sessionsQuery.isError && (
        <>
          {/* Intensity Chart */}
          <IntensityChart sessions={sessions} />

          {/* Run List Table */}
          <RunListTable
            sessions={sessions}
            onSelectSession={handleSelectSession}
            onDeleteSession={handleDeleteSession}
            onToggleCalibration={handleToggleSessionCalibration}
            isLoading={loadSessionMutation.isPending || deleteSessionMutation.isPending}
          />

          {/* Error loading session */}
          {loadSessionMutation.isError && (
            <div className="text-center py-2 text-destructive text-sm">
              Failed to load session. Please try again.
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default StartupScreen;
