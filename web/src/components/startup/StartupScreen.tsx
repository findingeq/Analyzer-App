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
} from "@/lib/client";

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

  // Fetch cloud sessions
  const sessionsQuery = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
    retry: false,
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
  });

  const handleSelectSession = useCallback(
    (sessionId: string) => {
      loadSessionMutation.mutate(sessionId);
    },
    [loadSessionMutation]
  );

  const sessions = sessionsQuery.data || [];

  return (
    <div className="h-full flex flex-col p-6 gap-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-lg font-semibold text-foreground mb-1">
          Welcome to VT Check
        </h2>
        <p className="text-sm text-muted-foreground">
          Select a session to analyze or upload a new CSV file
        </p>
      </div>

      {/* Error State */}
      {sessionsQuery.isError && (
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-sm">Cloud storage not available.</p>
          <p className="text-xs mt-1">Upload a local CSV file to get started.</p>
        </div>
      )}

      {/* Content */}
      {!sessionsQuery.isError && (
        <>
          {/* Intensity Chart */}
          <IntensityChart sessions={sessions} />

          {/* Run List Table */}
          <RunListTable
            sessions={sessions}
            onSelectSession={handleSelectSession}
            isLoading={sessionsQuery.isLoading || loadSessionMutation.isPending}
          />
        </>
      )}
    </div>
  );
}

export default StartupScreen;
