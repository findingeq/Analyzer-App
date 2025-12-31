/**
 * MainChart Component
 * Renders the primary VE and CUSUM visualization using ECharts
 */

import { useEffect, useMemo, useCallback } from "react";
import * as echarts from "echarts";
import { useChartResize } from "@/hooks/use-chart-resize";
import { useRunStore } from "@/store/use-run-store";
import { IntervalStatus } from "@/lib/api-types";
import type { EChartsOption } from "echarts";

// =============================================================================
// Chart Color Theme
// =============================================================================

const COLORS = {
  ve: "#60A5FA", // Blue 400 - chart-ve
  cusum: "#F87171", // Red 400
  slope: "#C084FC", // Purple 400 - chart-slope
  expected: "#6B7280", // Gray 500
  scatter: "rgba(96, 165, 250, 0.5)", // VE scatter dots
  gridLine: "#27272A", // Zinc 800
  text: "#A1A1AA", // Zinc 400
  background: "#09090B", // background
  intervalGood: "rgba(52, 211, 153, 0.15)", // status-good with alpha
  intervalWarn: "rgba(251, 191, 36, 0.15)", // status-warn with alpha
  intervalBad: "rgba(248, 113, 113, 0.15)", // status-bad with alpha
  intervalHover: "rgba(99, 102, 241, 0.25)", // primary with alpha
};

// =============================================================================
// Component
// =============================================================================

export function MainChart() {
  const { containerRef, chartRef } = useChartResize();

  const {
    analysisResult,
    selectedIntervalId,
    hoveredIntervalId,
    showSlopeLines,
    showCusum,
    zoomStart,
    zoomEnd,
    setZoomRange,
    setSelectedInterval,
  } = useRunStore();

  // Get status color for interval shading
  const getIntervalColor = useCallback((status: IntervalStatus) => {
    switch (status) {
      case IntervalStatus.BELOW_THRESHOLD:
        return COLORS.intervalGood;
      case IntervalStatus.BORDERLINE:
        return COLORS.intervalWarn;
      case IntervalStatus.ABOVE_THRESHOLD:
        return COLORS.intervalBad;
      default:
        return COLORS.intervalGood;
    }
  }, []);

  // Build chart options
  const chartOptions = useMemo((): EChartsOption | null => {
    if (!analysisResult) return null;

    const { breath_data, results, intervals } = analysisResult;

    // Build mark areas for intervals
    const markAreas: Array<Array<{ xAxis: number; itemStyle?: { color: string } }>> =
      results.map((result) => {
        const interval = intervals.find(
          (i) => i.interval_num === result.interval_num
        );
        if (!interval) return [];

        const isHighlighted =
          result.interval_num === selectedIntervalId ||
          result.interval_num === hoveredIntervalId;

        return [
          {
            xAxis: interval.start_time,
            itemStyle: {
              color: isHighlighted
                ? COLORS.intervalHover
                : getIntervalColor(result.status),
            },
          },
          {
            xAxis: interval.end_time,
          },
        ];
      });

    // Prepare series data - use explicit array for push compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series: any[] = [
      // VE Scatter (breath data points)
      {
        name: "VE (Breaths)",
        type: "scatter",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: breath_data.times.map((t, i) => [t, breath_data.ve_median[i]]),
        symbol: "circle",
        symbolSize: 3,
        itemStyle: {
          color: COLORS.scatter,
        },
        emphasis: {
          scale: false,
        },
        z: 1,
      },
      // VE Binned Line
      {
        name: "VE (Binned)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: breath_data.bin_times.map((t, i) => [t, breath_data.ve_binned[i]]),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          color: COLORS.ve,
          width: 2,
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: "rgba(96, 165, 250, 0.3)" },
            { offset: 1, color: "rgba(96, 165, 250, 0)" },
          ]),
        },
        markArea: {
          silent: true,
          data: markAreas,
        },
        z: 2,
      },
    ];

    // Add slope lines for selected interval
    if (showSlopeLines && selectedIntervalId !== null) {
      const selectedResult = results.find(
        (r) => r.interval_num === selectedIntervalId
      );
      if (selectedResult?.chart_data.slope_line_times.length) {
        series.push({
          name: "Slope Line",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: selectedResult.chart_data.slope_line_times.map((t, i) => [
            t,
            selectedResult.chart_data.slope_line_ve[i],
          ]),
          showSymbol: false,
          lineStyle: {
            color: COLORS.slope,
            width: 2,
            type: "dashed",
          },
          z: 3,
        });
      }
    }

    // Add CUSUM line for selected interval
    if (showCusum && selectedIntervalId !== null) {
      const selectedResult = results.find(
        (r) => r.interval_num === selectedIntervalId
      );
      if (selectedResult?.chart_data.cusum_values.length) {
        series.push({
          name: "CUSUM",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 1,
          data: selectedResult.chart_data.time_values.map((t, i) => [
            t,
            selectedResult.chart_data.cusum_values[i],
          ]),
          showSymbol: false,
          lineStyle: {
            color: COLORS.cusum,
            width: 2,
          },
          z: 4,
        });

        // Add threshold line
        series.push({
          name: "Threshold",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 1,
          data: [
            [
              selectedResult.chart_data.time_values[0],
              selectedResult.cusum_threshold,
            ],
            [
              selectedResult.chart_data.time_values[
                selectedResult.chart_data.time_values.length - 1
              ],
              selectedResult.cusum_threshold,
            ],
          ],
          showSymbol: false,
          lineStyle: {
            color: COLORS.cusum,
            width: 1,
            type: "dotted",
          },
          z: 3,
        });
      }
    }

    // Calculate max values for Y axes
    const maxVE = Math.max(...breath_data.ve_median, 100);
    const maxCusum =
      selectedIntervalId !== null
        ? Math.max(
            ...(results.find((r) => r.interval_num === selectedIntervalId)
              ?.chart_data.cusum_values ?? [0]),
            results.find((r) => r.interval_num === selectedIntervalId)
              ?.cusum_threshold ?? 100
          )
        : 100;

    return {
      backgroundColor: COLORS.background,
      animation: true,
      animationDuration: 300,
      tooltip: {
        trigger: "axis",
        backgroundColor: "#18181B",
        borderColor: "#27272A",
        textStyle: {
          color: "#FAFAFA",
        },
        formatter: (params: unknown) => {
          const p = params as Array<{ seriesName: string; value: number[] }>;
          if (!p.length) return "";
          const time = p[0].value[0];
          const mins = Math.floor(time / 60);
          const secs = Math.round(time % 60);
          let html = `<div class="font-medium">${mins}:${secs.toString().padStart(2, "0")}</div>`;
          p.forEach((item) => {
            if (item.seriesName !== "Threshold") {
              html += `<div>${item.seriesName}: ${item.value[1].toFixed(1)}</div>`;
            }
          });
          return html;
        },
      },
      legend: {
        show: true,
        top: 10,
        right: 10,
        textStyle: {
          color: COLORS.text,
        },
        data: ["VE (Binned)", "CUSUM", "Slope Line"],
      },
      grid: {
        left: 60,
        right: showCusum ? 60 : 20,
        top: 50,
        bottom: 80,
      },
      xAxis: {
        type: "value",
        name: "Time (seconds)",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: {
          color: COLORS.text,
        },
        axisLine: {
          lineStyle: {
            color: COLORS.gridLine,
          },
        },
        axisLabel: {
          color: COLORS.text,
          formatter: (value: number) => {
            const mins = Math.floor(value / 60);
            const secs = Math.round(value % 60);
            return `${mins}:${secs.toString().padStart(2, "0")}`;
          },
        },
        splitLine: {
          lineStyle: {
            color: COLORS.gridLine,
          },
        },
      },
      yAxis: [
        {
          type: "value",
          name: "VE (L/min)",
          nameTextStyle: {
            color: COLORS.ve,
          },
          position: "left",
          min: 0,
          max: Math.ceil(maxVE * 1.1),
          axisLine: {
            lineStyle: {
              color: COLORS.ve,
            },
          },
          axisLabel: {
            color: COLORS.ve,
          },
          splitLine: {
            lineStyle: {
              color: COLORS.gridLine,
            },
          },
        },
        {
          type: "value",
          name: "CUSUM",
          nameTextStyle: {
            color: COLORS.cusum,
          },
          position: "right",
          min: 0,
          max: Math.ceil(maxCusum * 1.2),
          show: showCusum,
          axisLine: {
            lineStyle: {
              color: COLORS.cusum,
            },
          },
          axisLabel: {
            color: COLORS.cusum,
          },
          splitLine: {
            show: false,
          },
        },
      ],
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: 0,
          start: zoomStart,
          end: zoomEnd,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: "slider",
          xAxisIndex: 0,
          start: zoomStart,
          end: zoomEnd,
          height: 30,
          bottom: 10,
          borderColor: COLORS.gridLine,
          fillerColor: "rgba(99, 102, 241, 0.2)",
          handleStyle: {
            color: COLORS.ve,
          },
          textStyle: {
            color: COLORS.text,
          },
          dataBackground: {
            lineStyle: {
              color: COLORS.ve,
            },
            areaStyle: {
              color: "rgba(96, 165, 250, 0.2)",
            },
          },
        },
      ],
      series,
    };
  }, [
    analysisResult,
    selectedIntervalId,
    hoveredIntervalId,
    showSlopeLines,
    showCusum,
    zoomStart,
    zoomEnd,
    getIntervalColor,
  ]);

  // Initialize chart
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Create chart instance
    chartRef.current = echarts.init(container, undefined, {
      renderer: "canvas",
    });

    // Handle zoom events
    chartRef.current.on("datazoom", (params: unknown) => {
      const p = params as { start?: number; end?: number; batch?: Array<{ start: number; end: number }> };
      if (p.batch) {
        setZoomRange(p.batch[0].start, p.batch[0].end);
      } else if (p.start !== undefined && p.end !== undefined) {
        setZoomRange(p.start, p.end);
      }
    });

    // Handle click on chart to select interval
    chartRef.current.on("click", (params: unknown) => {
      const p = params as { componentType: string; data: number[] };
      if (p.componentType === "series" && p.data) {
        const clickTime = p.data[0];
        // Find which interval was clicked
        if (analysisResult) {
          const clickedInterval = analysisResult.intervals.find(
            (i) => clickTime >= i.start_time && clickTime <= i.end_time
          );
          if (clickedInterval) {
            setSelectedInterval(clickedInterval.interval_num);
          }
        }
      }
    });

    return () => {
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [containerRef, chartRef, setZoomRange, setSelectedInterval, analysisResult]);

  // Update chart when options change
  useEffect(() => {
    if (chartRef.current && chartOptions) {
      chartRef.current.setOption(chartOptions, {
        notMerge: false,
        lazyUpdate: true,
      });
    }
  }, [chartOptions, chartRef]);

  // Empty state
  if (!analysisResult) {
    return (
      <div
        ref={containerRef}
        className="flex h-full w-full items-center justify-center bg-background text-muted-foreground"
      >
        Upload a CSV file to view analysis
      </div>
    );
  }

  return <div ref={containerRef} className="h-full w-full" />;
}

export default MainChart;
