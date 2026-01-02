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
} from "@/lib/client";
import { useQueryClient } from "@tanstack/react-query";

export function StartupScreen() {
  const {
    setCSVContent,
    setCSVMetadata,
    setRunType,
    setNumIntervals,
    setIntervalDuration,
    setRecoveryDuration,
    setDataSource,
  } = useRunStore();

  const queryClient = useQueryClient();

  // Fetch cloud sessions
  const sessionsQuery = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
    retry: false,
  });

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      // Refetch sessions after delete
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
    onError: (error) => {
      console.error("Failed to delete session:", error);
    },
  });

  // Load session mutation
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
      // Switch to cloud data source
      setDataSource("cloud");
    },
    onError: (error) => {
      console.error("Failed to load session:", error);
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

  // Safely get sessions array with validation
  const sessions = Array.isArray(sessionsQuery.data) ? sessionsQuery.data : [];

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
