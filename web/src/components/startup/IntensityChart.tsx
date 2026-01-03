/**
 * IntensityChart Component
 * Horizontal bar chart showing time in intensity domains (7-day rolling)
 */

import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { SessionInfo } from "@/lib/client";

interface IntensityChartProps {
  sessions: SessionInfo[];
}

export function IntensityChart({ sessions }: IntensityChartProps) {

  // Calculate time in each zone from sessions in the last 7 days
  const zoneData = useMemo(() => {
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

    // Filter to last 7 days and has summary data
    const recentSessions = sessions.filter((s) => {
      if (!s.summary?.date) return false;
      const sessionDate = new Date(s.summary.date);
      return sessionDate >= sevenDaysAgo && sessionDate <= now;
    });

    // Accumulate time in each zone (in minutes)
    let moderateMin = 0;
    let heavyMin = 0;
    let severeMin = 0;

    recentSessions.forEach((session) => {
      const summary = session.summary;
      if (!summary) return;

      // Use intensity from summary if available
      const durationMin = (summary.duration_seconds || 0) / 60;
      const intensity = summary.intensity?.toLowerCase();

      if (intensity === "moderate") {
        moderateMin += durationMin;
      } else if (intensity === "heavy") {
        heavyMin += durationMin;
      } else if (intensity === "severe") {
        severeMin += durationMin;
      } else {
        // Default to moderate if not classified
        moderateMin += durationMin;
      }
    });

    return {
      moderate: Math.round(moderateMin),
      heavy: Math.round(heavyMin),
      severe: Math.round(severeMin),
      total: Math.round(moderateMin + heavyMin + severeMin),
    };
  }, [sessions]);

  const option = useMemo(
    () => ({
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params: Array<{ marker: string; name: string; value: number }>) => {
          const param = params[0];
          if (!param) return "";
          return `${param.marker} ${param.name}: ${param.value} min`;
        },
      },
      grid: {
        left: "3%",
        right: "10%",
        top: "15%",
        bottom: "3%",
        containLabel: true,
      },
      xAxis: {
        type: "value",
        name: "Minutes",
        nameLocation: "end",
        axisLabel: { color: "#888" },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: "#333" } },
      },
      yAxis: {
        type: "category",
        data: ["Severe", "Heavy", "Moderate"],
        axisLabel: { color: "#ccc" },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: [
        {
          type: "bar",
          data: [
            { value: zoneData.severe, itemStyle: { color: "#ef4444" } },
            { value: zoneData.heavy, itemStyle: { color: "#f59e0b" } },
            { value: zoneData.moderate, itemStyle: { color: "#22c55e" } },
          ],
          barWidth: "50%",
          label: {
            show: true,
            position: "right",
            color: "#ccc",
            formatter: "{c} min",
          },
        },
      ],
    }),
    [zoneData]
  );

  return (
    <Card className="bg-card/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          7-Day Intensity Distribution
        </CardTitle>
      </CardHeader>
      <CardContent>
        {zoneData.total > 0 ? (
          <ReactECharts
            option={option}
            style={{ height: "150px", width: "100%" }}
            opts={{ renderer: "svg" }}
          />
        ) : (
          <div className="h-[150px] flex items-center justify-center text-sm text-muted-foreground">
            No sessions in the last 7 days
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default IntensityChart;
