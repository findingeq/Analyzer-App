/**
 * MainChart Component
 * Renders the primary VE and CUSUM visualization using ECharts
 */

import { useEffect, useMemo, useCallback, useRef } from "react";
import * as echarts from "echarts";
import { useChartResize } from "@/hooks/use-chart-resize";
import { useRunStore } from "@/store/use-run-store";
import { IntervalStatus } from "@/lib/api-types";
import { StartupScreen } from "@/components/startup/StartupScreen";
import type { EChartsOption } from "echarts";
import type { AnalysisResponse } from "@/lib/api-types";

// =============================================================================
// Chart Color Theme
// =============================================================================

const COLORS = {
  ve: "#60A5FA", // Blue 400 - chart-ve
  hr: "#F87171", // Red 400 - heart rate
  cusumOk: "#34D399", // Emerald 400 - below threshold
  cusumAlarm: "#FB923C", // Orange 400 - above threshold (changed from red)
  slope: "#C084FC", // Purple 400 - chart-slope
  expected: "#6B7280", // Gray 500
  scatter: "rgba(96, 165, 250, 0.5)", // VE scatter dots
  gridLine: "#27272A", // Zinc 800
  text: "#A1A1AA", // Zinc 400
  background: "#09090B", // background
  intervalGood: "rgba(52, 211, 153, 0.15)", // status-good with alpha
  intervalWarn: "rgba(251, 191, 36, 0.15)", // status-warn with alpha
  intervalBad: "rgba(248, 113, 113, 0.15)", // status-bad with alpha
};

// =============================================================================
// Component
// =============================================================================

export function MainChart() {
  const { containerRef, chartRef } = useChartResize();

  const {
    analysisResult,
    selectedIntervalId,
    showSlopeLines,
    showCusum,
    zoomStart,
    zoomEnd,
    setZoomRange,
    setSelectedInterval,
    vt1Ceiling,
    vt2Ceiling,
  } = useRunStore();

  // Ref to hold analysisResult for click handler (avoids stale closure)
  const analysisResultRef = useRef<AnalysisResponse | null>(null);
  analysisResultRef.current = analysisResult;

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

    // Build mark areas for intervals (fixed coloring based on status only)
    const markAreas: Array<Array<{ xAxis: number; itemStyle?: { color: string } }>> =
      results.map((result) => {
        const interval = intervals.find(
          (i) => i.interval_num === result.interval_num
        );
        if (!interval) return [];

        return [
          {
            xAxis: interval.start_time,
            itemStyle: {
              color: getIntervalColor(result.status),
            },
          },
          {
            xAxis: interval.end_time,
          },
        ];
      });

    // Prepare series data
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
          disabled: true,
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
        emphasis: {
          disabled: true,
        },
        markArea: {
          silent: true,
          data: markAreas,
        },
        z: 2,
      },
    ];

    // Add HR line if HR data is available
    // Uses top grid (gridIndex 1, xAxisIndex 1, yAxisIndex 2)
    if (breath_data.hr && breath_data.hr.length > 0) {
      series.push({
        name: "HR",
        type: "line",
        xAxisIndex: 1, // Top grid x-axis
        yAxisIndex: 2, // HR y-axis (top grid)
        data: breath_data.times.map((t, i) => [t, breath_data.hr![i]]),
        smooth: false,
        showSymbol: false,
        lineStyle: {
          color: COLORS.hr,
          width: 1.5,
        },
        emphasis: {
          disabled: true,
        },
        z: 2,
      });
    }

    // Add 3-hinge segment lines for all intervals when slope lines enabled
    if (showSlopeLines) {
      let slopeLegendShown = false;
      results.forEach((result) => {
        // Build continuous line from all 3 segments
        const segmentData: Array<[number, number]> = [];

        // Segment 1 (ramp-up to Phase III onset)
        if (result.chart_data.segment1_times?.length && result.chart_data.segment1_ve?.length) {
          result.chart_data.segment1_times.forEach((t, i) => {
            segmentData.push([t, result.chart_data.segment1_ve![i]]);
          });
        }

        // Segment 2 (Phase III onset to 2nd hinge) - skip first point if overlaps with segment1
        if (result.chart_data.segment2_times?.length && result.chart_data.segment2_ve?.length) {
          const startIdx = segmentData.length > 0 ? 1 : 0;
          result.chart_data.segment2_times.slice(startIdx).forEach((t, i) => {
            segmentData.push([t, result.chart_data.segment2_ve![startIdx + i]]);
          });
        }

        // Segment 3 (2nd hinge to end) - skip first point if overlaps with segment2
        if (result.chart_data.segment3_times?.length && result.chart_data.segment3_ve?.length) {
          const startIdx = segmentData.length > 0 ? 1 : 0;
          result.chart_data.segment3_times.slice(startIdx).forEach((t, i) => {
            segmentData.push([t, result.chart_data.segment3_ve![startIdx + i]]);
          });
        }

        // Fallback to combined slope line if segment data is missing
        if (segmentData.length === 0 && result.chart_data.slope_line_times?.length && result.chart_data.slope_line_ve?.length) {
          result.chart_data.slope_line_times.forEach((t, i) => {
            segmentData.push([t, result.chart_data.slope_line_ve[i]]);
          });
        }

        // Add slope line if we have data (need at least 2 points to draw a line)
        if (segmentData.length >= 2) {
          series.push({
            name: slopeLegendShown ? "" : "Slope",
            type: "line",
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: segmentData,
            showSymbol: false,
            lineStyle: {
              color: COLORS.slope,
              width: 2,
            },
            emphasis: {
              disabled: true,
            },
            z: 3,
          });
          slopeLegendShown = true;
        }
      });
    }

    // Add slope annotations for selected interval
    if (showSlopeLines && selectedIntervalId !== null) {
      const selectedResult = results.find(
        (r) => r.interval_num === selectedIntervalId
      );

      if (selectedResult) {
        // Single slope annotation (segment2 midpoint)
        if (
          selectedResult.ve_drift_pct !== null &&
          selectedResult.ve_drift_pct !== undefined &&
          selectedResult.chart_data.segment2_times?.length &&
          selectedResult.chart_data.segment2_times.length >= 2
        ) {
          const seg2Times = selectedResult.chart_data.segment2_times;
          const seg2Ve = selectedResult.chart_data.segment2_ve!;
          const midX = (seg2Times[0] + seg2Times[seg2Times.length - 1]) / 2;
          const midY = (seg2Ve[0] + seg2Ve[seg2Ve.length - 1]) / 2;

          series.push({
            type: "scatter",
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: [[midX, midY]],
            symbol: "none",
            label: {
              show: true,
              formatter: `${selectedResult.ve_drift_pct >= 0 ? "+" : ""}${selectedResult.ve_drift_pct.toFixed(1)}%/min`,
              color: COLORS.slope,
              fontSize: 10,
              backgroundColor: "rgba(9, 9, 11, 0.8)",
              padding: [2, 4],
              borderRadius: 2,
            },
            z: 10,
          });
        }
      }
    }

    // Add CUSUM lines for all intervals when CUSUM is enabled
    // Uses bottom grid (gridIndex 2, xAxisIndex 2, yAxisIndex 1)
    if (showCusum) {
      let cusumLegendShown = false;
      results.forEach((result) => {
        if (!result.chart_data.cusum_values.length) return;

        const timeValues = result.chart_data.time_values;
        const cusumValues = result.chart_data.cusum_values;
        const transitions = result.cusum_transitions || [];

        // Build segments based on transitions (green = normal, orange = alarm)
        if (transitions.length === 0) {
          // No transitions - all green
          series.push({
            name: cusumLegendShown ? "" : "CUSUM",
            type: "line",
            xAxisIndex: 2, // Bottom grid x-axis
            yAxisIndex: 1, // CUSUM y-axis (bottom grid)
            data: timeValues.map((t, i) => [t, cusumValues[i]]),
            showSymbol: false,
            lineStyle: {
              color: COLORS.cusumOk,
              width: 2.5, // Thickened for visibility when flat
            },
            emphasis: {
              disabled: true,
            },
            z: 4,
          });
          cusumLegendShown = true;
        } else {
          // Build segments from transitions
          // Start with green, switch at each transition
          let currentIdx = 0;
          let isAlarmState = false;

          for (let t = 0; t < transitions.length; t++) {
            const transition = transitions[t];
            const transitionIdx = timeValues.findIndex((time) => time >= transition.time);

            if (transitionIdx <= currentIdx) continue;

            // Draw segment from currentIdx to transitionIdx
            const segmentData = [];
            for (let i = currentIdx; i <= transitionIdx; i++) {
              segmentData.push([timeValues[i], cusumValues[i]]);
            }

            series.push({
              name: !cusumLegendShown ? "CUSUM" : "",
              type: "line",
              xAxisIndex: 2, // Bottom grid x-axis
              yAxisIndex: 1, // CUSUM y-axis (bottom grid)
              data: segmentData,
              showSymbol: false,
              lineStyle: {
                color: isAlarmState ? COLORS.cusumAlarm : COLORS.cusumOk,
                width: 2.5, // Thickened for visibility when flat
              },
              emphasis: {
                disabled: true,
              },
              z: 4,
            });
            cusumLegendShown = true;

            currentIdx = transitionIdx;
            isAlarmState = transition.is_alarm; // After this transition, switch state
          }

          // Draw final segment from last transition to end
          if (currentIdx < timeValues.length - 1) {
            const segmentData = [];
            for (let i = currentIdx; i < timeValues.length; i++) {
              segmentData.push([timeValues[i], cusumValues[i]]);
            }

            series.push({
              name: !cusumLegendShown ? "CUSUM" : "",
              type: "line",
              xAxisIndex: 2, // Bottom grid x-axis
              yAxisIndex: 1, // CUSUM y-axis (bottom grid)
              data: segmentData,
              showSymbol: false,
              lineStyle: {
                color: isAlarmState ? COLORS.cusumAlarm : COLORS.cusumOk,
                width: 2.5, // Thickened for visibility when flat
              },
              emphasis: {
                disabled: true,
              },
              z: 4,
            });
            cusumLegendShown = true;
          }
        }

      });
    }

    // Add VE ceiling line when ceiling-based analysis is used
    const hasCeilingBasedAnalysis = results.some((r) => r.is_ceiling_based);
    if (hasCeilingBasedAnalysis) {
      const { run_type } = analysisResult;
      const ceilingValue = run_type === "MODERATE" ? vt1Ceiling : vt2Ceiling;
      const firstTime = results[0]?.chart_data.time_values[0] ?? 0;
      const lastTime = results[results.length - 1]?.chart_data.time_values.slice(-1)[0] ?? 0;

      series.push({
        name: "",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0, // VE axis (primary)
        data: [
          [firstTime, ceilingValue],
          [lastTime, ceilingValue],
        ],
        showSymbol: false,
        lineStyle: {
          color: COLORS.cusumAlarm,
          width: 1,
          type: "dotted",
        },
        z: 3,
      });
    }

    // Calculate x-axis range based on intervals (to span full width)
    const firstIntervalStart = intervals.length > 0 ? intervals[0].start_time : breath_data.times[0];
    const lastIntervalEnd = intervals.length > 0 ? intervals[intervals.length - 1].end_time : breath_data.times[breath_data.times.length - 1];
    const xAxisMin = firstIntervalStart;
    const xAxisMax = lastIntervalEnd;

    // Check if HR data exists
    const hasHR = !!(breath_data.hr && breath_data.hr.length > 0);

    // =============================================================================
    // VE Axis Scaling: VE LINE stays within 20-80% of chart height
    // Use ve_binned (line data) for scaling, not ve_median (breath dots)
    // Breath dots may extend beyond but that's OK since they're faint
    // =============================================================================
    const veBinnedValues = breath_data.ve_binned.filter((v) => v != null && !isNaN(v));
    const minVE = veBinnedValues.length > 0 ? Math.min(...veBinnedValues) : 0;
    const maxVE = veBinnedValues.length > 0 ? Math.max(...veBinnedValues) : 100;
    const veRange = maxVE - minVE || 1; // Avoid division by zero
    // To place minVE at 20% and maxVE at 80%, axis range = data range / 0.6
    // Buffer below = 20% of axis = dataRange * (20/60) = dataRange * 0.333
    // Buffer above = 20% of axis = dataRange * (20/60) = dataRange * 0.333
    const veAxisMin = Math.max(0, minVE - veRange * 0.333);
    const veAxisMax = maxVE + veRange * 0.333;

    // =============================================================================
    // CUSUM Axis Scaling: Data stays within bottom 25% of chart (0-25%)
    // Axis ticks span bottom 50% of chart
    // Within the bottom 50% axis, data occupies 0-50% (which is 0-25% of full chart)
    // =============================================================================
    const allCusumValues = results.flatMap((r) => r.chart_data.cusum_values);
    const allThresholds = results.map((r) => r.cusum_threshold);
    const maxCusumData = allCusumValues.length > 0
      ? Math.max(...allCusumValues, ...allThresholds, 1) // Min of 1 to avoid zero range
      : 100;
    // To place data in bottom 50% of axis (0-25% of chart), extend max by 100%
    const cusumAxisMin = 0;
    const cusumAxisMax = maxCusumData * 2;

    // =============================================================================
    // HR Axis Scaling: Data stays within top 25% of chart (75-100%)
    // Axis ticks span top 50% of chart
    // Within the top 50% axis, data occupies 50-100% (which is 75-100% of full chart)
    // =============================================================================
    const hrValues = hasHR ? breath_data.hr!.filter((v) => v != null && !isNaN(v)) : [];
    const minHR = hrValues.length > 0 ? Math.min(...hrValues) : 80;
    const maxHR = hrValues.length > 0 ? Math.max(...hrValues) : 180;
    const hrRange = maxHR - minHR || 1; // Avoid division by zero
    const hrPadding = Math.max(5, hrRange * 0.05);
    // To place data in top 50% of axis (75-100% of chart), extend min downward by data range
    const hrAxisMin = minHR - hrPadding - hrRange;
    const hrAxisMax = maxHR + hrPadding;

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

          // Determine if time is in an interval or recovery period
          let timeLabel = "";
          let elapsedTime = time;

          // Check if within an interval
          const currentInterval = intervals.find(
            (i) => time >= i.start_time && time <= i.end_time
          );
          if (currentInterval) {
            // Time within interval - show elapsed time since interval start
            elapsedTime = time - currentInterval.start_time;
            timeLabel = `Interval ${currentInterval.interval_num}`;
          } else {
            // Check if within a recovery period (between intervals)
            for (let idx = 0; idx < intervals.length - 1; idx++) {
              const currentEnd = intervals[idx].end_time;
              const nextStart = intervals[idx + 1].start_time;
              if (time > currentEnd && time < nextStart) {
                elapsedTime = time - currentEnd;
                timeLabel = `Recovery ${idx + 1}`;
                break;
              }
            }
            // If still no label, show absolute time
            if (!timeLabel) {
              timeLabel = "Time";
            }
          }

          const mins = Math.floor(elapsedTime / 60);
          const secs = Math.round(elapsedTime % 60);

          // Build tooltip with time, VE, and HR (always show all three)
          let html = `<div class="font-medium">${timeLabel}: ${mins}:${secs.toString().padStart(2, "0")}</div>`;

          // Find VE value (try VE Binned first, then VE Breaths)
          const veItem = p.find((item) => item.seriesName === "VE (Binned)") ||
                         p.find((item) => item.seriesName === "VE (Breaths)");
          if (veItem && veItem.value[1] != null) {
            html += `<div style="color: ${COLORS.ve}">VE: ${veItem.value[1].toFixed(1)} L/min</div>`;
          }

          // Find HR value
          const hrItem = p.find((item) => item.seriesName === "HR");
          if (hrItem && hrItem.value[1] != null) {
            html += `<div style="color: ${COLORS.hr}">HR: ${Math.round(hrItem.value[1])} bpm</div>`;
          }

          return html;
        },
      },
      legend: {
        show: true,
        top: 10,
        right: 120,
        textStyle: {
          color: COLORS.text,
        },
        itemWidth: 20,
        itemHeight: 2,
        data: [
          {
            name: "VE (Binned)",
            icon: "rect",
            itemStyle: { color: COLORS.ve },
          },
          ...(hasHR ? [{
            name: "HR",
            icon: "rect",
            itemStyle: { color: COLORS.hr },
          }] : []),
          {
            name: "CUSUM",
            icon: "rect",
            itemStyle: {
              // Green if no unrecovered alarms, orange if any unrecovered alarm
              color: results.some((r) => r.alarm_time !== null && r.alarm_time !== undefined && !r.cusum_recovered)
                ? COLORS.cusumAlarm
                : COLORS.cusumOk,
            },
          },
          {
            name: "Slope",
            icon: "rect",
            itemStyle: { color: COLORS.slope },
          },
        ],
      },
      grid: [
        // Grid 0: Main grid for VE (full height)
        {
          left: 60,
          right: 70,
          top: 50,
          bottom: 80,
        },
        // Grid 1: Top grid for HR axis (top 50% - from top to middle)
        {
          left: 60,
          right: 70,
          top: 50,
          bottom: "52%", // Ends at 52% from bottom (top half)
        },
        // Grid 2: Bottom grid for CUSUM axis (bottom 50% - from middle to bottom)
        {
          left: 60,
          right: 70,
          top: "52%", // Starts at 52% from top (bottom half)
          bottom: 80,
        },
      ],
      xAxis: [
        // xAxis 0: Main x-axis for VE (visible)
        {
          type: "value",
          gridIndex: 0,
          name: "Time (seconds)",
          nameLocation: "middle",
          nameGap: 30,
          nameTextStyle: {
            color: COLORS.text,
          },
          min: xAxisMin,
          max: xAxisMax,
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
        // xAxis 1: Top grid x-axis for HR (hidden, synced)
        {
          type: "value",
          gridIndex: 1,
          show: false,
          min: xAxisMin,
          max: xAxisMax,
        },
        // xAxis 2: Bottom grid x-axis for CUSUM (hidden, synced)
        {
          type: "value",
          gridIndex: 2,
          show: false,
          min: xAxisMin,
          max: xAxisMax,
        },
      ],
      yAxis: [
        // yAxis 0: VE axis (main grid, left side, full height)
        // Data scaled to appear in 20-80% of chart
        {
          type: "value",
          gridIndex: 0,
          name: "VE (L/min)",
          nameTextStyle: {
            color: COLORS.ve,
          },
          position: "left",
          min: Math.floor(veAxisMin),
          max: Math.ceil(veAxisMax),
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
        // yAxis 1: CUSUM axis (bottom grid, right side)
        // Data scaled to appear in bottom 25% of chart
        {
          type: "value",
          gridIndex: 2,
          position: "right",
          min: cusumAxisMin,
          max: Math.ceil(cusumAxisMax),
          show: showCusum,
          axisLine: {
            lineStyle: {
              color: COLORS.cusumOk,
            },
          },
          axisLabel: {
            color: COLORS.cusumOk,
          },
          splitLine: {
            show: false,
          },
        },
        // yAxis 2: HR axis (top grid, right side)
        // Data scaled to appear in top 25% of chart
        {
          type: "value",
          gridIndex: 1,
          position: "right",
          min: Math.floor(hrAxisMin),
          max: Math.ceil(hrAxisMax),
          show: hasHR,
          axisLine: {
            lineStyle: {
              color: COLORS.hr,
            },
          },
          axisLabel: {
            color: COLORS.hr,
          },
          splitLine: {
            show: false,
          },
        },
      ],
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1, 2], // Sync all three x-axes
          start: zoomStart,
          end: zoomEnd,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: "slider",
          xAxisIndex: [0, 1, 2], // Sync all three x-axes
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
    showSlopeLines,
    showCusum,
    zoomStart,
    zoomEnd,
    getIntervalColor,
  ]);

  // Initialize chart (only once when container mounts)
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
    // Uses ref to avoid stale closure when analysisResult changes
    chartRef.current.on("click", (params: unknown) => {
      const p = params as { componentType: string; data: number[] };
      if (p.componentType === "series" && p.data) {
        const clickTime = p.data[0];
        // Find which interval was clicked (use ref for current value)
        const currentResult = analysisResultRef.current;
        if (currentResult) {
          const clickedInterval = currentResult.intervals.find(
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
  }, [containerRef, chartRef, setZoomRange, setSelectedInterval]);

  // Update chart when options change
  useEffect(() => {
    if (chartRef.current && chartOptions) {
      chartRef.current.setOption(chartOptions, {
        notMerge: true,  // Replace all options to clear old series when switching files
        lazyUpdate: true,
      });
    }
  }, [chartOptions, chartRef]);

  // Always render chart container so it's available for initialization
  // Show StartupScreen overlay when no analysis result
  return (
    <div className="h-full w-full relative">
      {/* Chart container - always rendered so ref is stable */}
      <div
        ref={containerRef}
        className="h-full w-full"
        style={{ visibility: analysisResult ? 'visible' : 'hidden' }}
      />
      {/* Startup screen overlay when no data */}
      {!analysisResult && (
        <div className="absolute inset-0 bg-background overflow-auto">
          <StartupScreen />
        </div>
      )}
    </div>
  );
}

export default MainChart;
