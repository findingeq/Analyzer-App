/**
 * RunListTable Component
 * Filterable, sortable table of cloud sessions with click-to-analyze
 */

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Filter,
  X,
} from "lucide-react";
import type { SessionInfo } from "@/lib/client";

interface RunListTableProps {
  sessions: SessionInfo[];
  onSelectSession: (sessionId: string) => void;
  isLoading?: boolean;
}

type SortField = "date" | "intensity" | "type" | "pace" | "duration";
type SortDirection = "asc" | "desc";

interface Filters {
  intensity: string;
  type: string;
  minPace: string;
  maxPace: string;
  minDuration: string;
  maxDuration: string;
}

export function RunListTable({
  sessions,
  onSelectSession,
  isLoading,
}: RunListTableProps) {
  const [sortField, setSortField] = useState<SortField>("date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<Filters>({
    intensity: "all",
    type: "all",
    minPace: "",
    maxPace: "",
    minDuration: "",
    maxDuration: "",
  });

  // Format date for display
  const formatDate = (dateStr: string | null | undefined) => {
    if (!dateStr) return "Unknown";
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return dateStr;
    }
  };

  // Format pace (min/mile)
  const formatPace = (paceMinPerMile: number | null | undefined) => {
    if (!paceMinPerMile) return "-";
    const mins = Math.floor(paceMinPerMile);
    const secs = Math.round((paceMinPerMile - mins) * 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Format duration
  const formatDuration = (seconds: number | null | undefined) => {
    if (!seconds) return "-";
    const mins = Math.round(seconds / 60);
    return `${mins} min`;
  };

  // Format type (intervals × duration)
  const formatType = (session: SessionInfo) => {
    const summary = session.summary;
    if (!summary) return "-";

    if (summary.run_type === "VT1") {
      return "Steady State";
    }

    const intervals = summary.num_intervals || 1;
    const duration = summary.interval_duration_min || 0;
    if (intervals === 1) {
      return `${Math.round(duration)} min`;
    }
    return `${intervals}×${Math.round(duration)}`;
  };

  // Get intensity badge color
  const getIntensityColor = (intensity: string | null | undefined) => {
    switch (intensity?.toLowerCase()) {
      case "moderate":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "heavy":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "severe":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  // Filter and sort sessions
  const filteredSessions = useMemo(() => {
    let result = sessions.filter((s) => s.summary);

    // Apply filters
    if (filters.intensity !== "all") {
      result = result.filter(
        (s) => s.summary?.intensity?.toLowerCase() === filters.intensity.toLowerCase()
      );
    }
    if (filters.type !== "all") {
      result = result.filter((s) => {
        if (filters.type === "steady") return s.summary?.run_type === "VT1";
        if (filters.type === "interval") return s.summary?.run_type === "VT2";
        return true;
      });
    }
    if (filters.minPace) {
      const minPace = parseFloat(filters.minPace);
      result = result.filter((s) => {
        const pace = s.summary?.avg_pace_min_per_mile;
        return pace && pace >= minPace;
      });
    }
    if (filters.maxPace) {
      const maxPace = parseFloat(filters.maxPace);
      result = result.filter((s) => {
        const pace = s.summary?.avg_pace_min_per_mile;
        return pace && pace <= maxPace;
      });
    }
    if (filters.minDuration) {
      const minDur = parseFloat(filters.minDuration) * 60; // Convert to seconds
      result = result.filter((s) => {
        const dur = s.summary?.duration_seconds;
        return dur && dur >= minDur;
      });
    }
    if (filters.maxDuration) {
      const maxDur = parseFloat(filters.maxDuration) * 60;
      result = result.filter((s) => {
        const dur = s.summary?.duration_seconds;
        return dur && dur <= maxDur;
      });
    }

    // Sort
    result.sort((a, b) => {
      let aVal: string | number = 0;
      let bVal: string | number = 0;

      switch (sortField) {
        case "date":
          aVal = a.summary?.date || "";
          bVal = b.summary?.date || "";
          break;
        case "intensity":
          const order = { severe: 3, heavy: 2, moderate: 1 };
          aVal = order[a.summary?.intensity?.toLowerCase() as keyof typeof order] || 0;
          bVal = order[b.summary?.intensity?.toLowerCase() as keyof typeof order] || 0;
          break;
        case "type":
          aVal = a.summary?.run_type || "";
          bVal = b.summary?.run_type || "";
          break;
        case "pace":
          aVal = a.summary?.avg_pace_min_per_mile || 999;
          bVal = b.summary?.avg_pace_min_per_mile || 999;
          break;
        case "duration":
          aVal = a.summary?.duration_seconds || 0;
          bVal = b.summary?.duration_seconds || 0;
          break;
      }

      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDirection === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      return sortDirection === "asc"
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number);
    });

    return result;
  }, [sessions, filters, sortField, sortDirection]);

  // Handle sort click
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  // Clear filters
  const clearFilters = () => {
    setFilters({
      intensity: "all",
      type: "all",
      minPace: "",
      maxPace: "",
      minDuration: "",
      maxDuration: "",
    });
  };

  // Render sort icon
  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="h-3 w-3 ml-1 opacity-50" />;
    }
    return sortDirection === "asc" ? (
      <ArrowUp className="h-3 w-3 ml-1" />
    ) : (
      <ArrowDown className="h-3 w-3 ml-1" />
    );
  };

  const hasActiveFilters =
    filters.intensity !== "all" ||
    filters.type !== "all" ||
    filters.minPace ||
    filters.maxPace ||
    filters.minDuration ||
    filters.maxDuration;

  return (
    <Card className="bg-card/50 flex-1">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>Run History</span>
          <div className="flex items-center gap-2">
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="h-6 px-2 text-xs"
              >
                <X className="h-3 w-3 mr-1" />
                Clear
              </Button>
            )}
            <Button
              variant={showFilters ? "default" : "ghost"}
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="h-6 px-2"
            >
              <Filter className="h-3 w-3" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {/* Filters */}
        {showFilters && (
          <div className="px-4 py-2 border-b border-border bg-muted/30 grid grid-cols-3 gap-2">
            <Select
              value={filters.intensity}
              onValueChange={(v) => setFilters({ ...filters, intensity: v })}
            >
              <SelectTrigger className="h-7 text-xs">
                <SelectValue placeholder="Intensity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Intensities</SelectItem>
                <SelectItem value="moderate">Moderate</SelectItem>
                <SelectItem value="heavy">Heavy</SelectItem>
                <SelectItem value="severe">Severe</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={filters.type}
              onValueChange={(v) => setFilters({ ...filters, type: v })}
            >
              <SelectTrigger className="h-7 text-xs">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="steady">Steady State</SelectItem>
                <SelectItem value="interval">Intervals</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex gap-1">
              <Input
                type="number"
                placeholder="Min pace"
                value={filters.minPace}
                onChange={(e) => setFilters({ ...filters, minPace: e.target.value })}
                className="h-7 text-xs"
              />
              <Input
                type="number"
                placeholder="Max pace"
                value={filters.maxPace}
                onChange={(e) => setFilters({ ...filters, maxPace: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
          </div>
        )}

        {/* Table */}
        <div className="overflow-auto max-h-[400px]">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-card z-10">
              <tr className="border-b border-border">
                <th
                  className="py-2 px-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("date")}
                >
                  <div className="flex items-center">
                    Date
                    <SortIcon field="date" />
                  </div>
                </th>
                <th
                  className="py-2 px-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("intensity")}
                >
                  <div className="flex items-center">
                    Intensity
                    <SortIcon field="intensity" />
                  </div>
                </th>
                <th
                  className="py-2 px-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("type")}
                >
                  <div className="flex items-center">
                    Type
                    <SortIcon field="type" />
                  </div>
                </th>
                <th
                  className="py-2 px-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("pace")}
                >
                  <div className="flex items-center">
                    Avg Pace
                    <SortIcon field="pace" />
                  </div>
                </th>
                <th
                  className="py-2 px-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("duration")}
                >
                  <div className="flex items-center">
                    Duration
                    <SortIcon field="duration" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-muted-foreground">
                    Loading sessions...
                  </td>
                </tr>
              )}
              {!isLoading && filteredSessions.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-muted-foreground">
                    {hasActiveFilters
                      ? "No sessions match your filters"
                      : "No sessions found"}
                  </td>
                </tr>
              )}
              {!isLoading &&
                filteredSessions.map((session) => (
                  <tr
                    key={session.session_id}
                    onClick={() => onSelectSession(session.session_id)}
                    className="border-b border-border/50 cursor-pointer hover:bg-muted/50 transition-colors"
                  >
                    <td className="py-2 px-3 text-primary font-medium">
                      {formatDate(session.summary?.date)}
                    </td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] border ${getIntensityColor(
                          session.summary?.intensity
                        )}`}
                      >
                        {session.summary?.intensity || "Unknown"}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">
                      {formatType(session)}
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">
                      {formatPace(session.summary?.avg_pace_min_per_mile)}
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">
                      {formatDuration(session.summary?.duration_seconds)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

export default RunListTable;
