/**
 * Zustand Store for VT Threshold Analyzer
 * Manages global UI state and analysis data
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  RunType,
  AnalysisResponse,
  ParseCSVResponse,
  IntervalResult,
  CalibrationUpdateResponse,
} from "@/lib/api-types";

// =============================================================================
// Store State Interface
// =============================================================================

type DataSource = "local" | "cloud";

interface RunState {
  // File state
  csvContent: string | null;
  csvMetadata: ParseCSVResponse | null;
  fileName: string | null;

  // Run configuration
  runType: RunType | null;
  numIntervals: number;
  intervalDurationMin: number;
  recoveryDurationMin: number;

  // Analysis results
  analysisResult: AnalysisResponse | null;
  isAnalyzing: boolean;

  // UI state
  selectedIntervalId: number | null;
  hoveredIntervalId: number | null;
  showSlopeLines: boolean;
  showCusum: boolean;
  sidebarCollapsed: boolean;

  // Zoom state for chart
  zoomStart: number;
  zoomEnd: number;

  // Cloud state
  dataSource: DataSource;

  // VT Thresholds (persisted)
  vt1Ceiling: number;
  vt2Ceiling: number;
  useThresholdsForAll: boolean;

  // Calibration state
  pendingVEPrompt: CalibrationUpdateResponse["ve_prompt"] | null;

  // Advanced CUSUM parameters (persisted)
  // Note: VT1/Moderate doesn't use maxDrift or splitRatio in classification
  advancedParams: {
    phase3OnsetOverride: number | null;
    maxDriftVt2: number;
    hMultiplierVt1: number;
    hMultiplierVt2: number;
    sigmaPctVt1: number;
    sigmaPctVt2: number;
    expectedDriftVt1: number;
    expectedDriftVt2: number;
    splitRatioVt2: number;
  };
}

// =============================================================================
// Store Actions Interface
// =============================================================================

interface RunActions {
  // File actions
  setCSVContent: (content: string | null, fileName?: string) => void;
  setCSVMetadata: (metadata: ParseCSVResponse | null) => void;

  // Configuration actions
  setRunType: (runType: RunType | null) => void;
  setNumIntervals: (num: number) => void;
  setIntervalDuration: (duration: number) => void;
  setRecoveryDuration: (duration: number) => void;

  // Analysis actions
  setAnalysisResult: (result: AnalysisResponse | null) => void;
  setIsAnalyzing: (isAnalyzing: boolean) => void;

  // UI actions
  setSelectedInterval: (intervalNum: number | null) => void;
  setHoveredInterval: (intervalNum: number | null) => void;
  toggleSlopeLines: () => void;
  toggleCusum: () => void;
  toggleSidebar: () => void;

  // Zoom actions
  setZoomRange: (start: number, end: number) => void;
  resetZoom: () => void;

  // Cloud actions
  setDataSource: (source: DataSource) => void;

  // VT Threshold actions
  setVt1Ceiling: (value: number) => void;
  setVt2Ceiling: (value: number) => void;
  setUseThresholdsForAll: (value: boolean) => void;

  // Advanced params actions
  setAdvancedParams: (params: Partial<RunState["advancedParams"]>) => void;

  // Calibration actions
  setPendingVEPrompt: (prompt: CalibrationUpdateResponse["ve_prompt"] | null) => void;
  clearVEPrompt: () => void;

  // Reset
  reset: () => void;
}

// =============================================================================
// Initial State
// =============================================================================

const defaultAdvancedParams = {
  phase3OnsetOverride: null as number | null,
  maxDriftVt2: 3.0,
  hMultiplierVt1: 5.0,
  hMultiplierVt2: 5.0,
  sigmaPctVt1: 7.0,
  sigmaPctVt2: 4.0,
  expectedDriftVt1: 0.3,
  expectedDriftVt2: 1.0,
  splitRatioVt2: 1.2,
};

const initialState: RunState = {
  csvContent: null,
  csvMetadata: null,
  fileName: null,
  runType: null,
  numIntervals: 1,
  intervalDurationMin: 8,
  recoveryDurationMin: 0,
  analysisResult: null,
  isAnalyzing: false,
  selectedIntervalId: null,
  hoveredIntervalId: null,
  showSlopeLines: true,
  showCusum: true,
  sidebarCollapsed: false,
  zoomStart: 0,
  zoomEnd: 100,
  dataSource: "local",
  vt1Ceiling: 100.0,
  vt2Ceiling: 120.0,
  useThresholdsForAll: false,
  advancedParams: defaultAdvancedParams,
  pendingVEPrompt: null,
};

// =============================================================================
// Store Creation
// =============================================================================

export const useRunStore = create<RunState & RunActions>()(
  persist(
    (set) => ({
      ...initialState,

      // File actions
      setCSVContent: (content, fileName) =>
        set({
          csvContent: content,
          fileName: fileName ?? null,
          // Reset analysis and zoom when new file is loaded
          analysisResult: null,
          selectedIntervalId: null,
          hoveredIntervalId: null,
          zoomStart: 0,
          zoomEnd: 100,
        }),

      setCSVMetadata: (metadata) => set({ csvMetadata: metadata }),

      // Configuration actions
      setRunType: (runType) => set({ runType }),
      setNumIntervals: (numIntervals) => set({ numIntervals }),
      setIntervalDuration: (intervalDurationMin) => set({ intervalDurationMin }),
      setRecoveryDuration: (recoveryDurationMin) => set({ recoveryDurationMin }),

      // Analysis actions
      setAnalysisResult: (analysisResult) =>
        set({
          analysisResult,
          // Auto-select first interval and reset zoom when results arrive
          selectedIntervalId: analysisResult?.results?.[0]?.interval_num ?? null,
          zoomStart: 0,
          zoomEnd: 100,
        }),
      setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),

      // UI actions
      setSelectedInterval: (selectedIntervalId) => set({ selectedIntervalId }),
      setHoveredInterval: (hoveredIntervalId) => set({ hoveredIntervalId }),
      toggleSlopeLines: () => set((state) => ({ showSlopeLines: !state.showSlopeLines })),
      toggleCusum: () => set((state) => ({ showCusum: !state.showCusum })),
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      // Zoom actions
      setZoomRange: (zoomStart, zoomEnd) => set({ zoomStart, zoomEnd }),
      resetZoom: () => set({ zoomStart: 0, zoomEnd: 100 }),

      // Cloud actions
      setDataSource: (dataSource) =>
        set({
          dataSource,
          // Reset file state when switching data source
          csvContent: null,
          csvMetadata: null,
          fileName: null,
          analysisResult: null,
          zoomStart: 0,
          zoomEnd: 100,
        }),

      // VT Threshold actions
      setVt1Ceiling: (vt1Ceiling) => set({ vt1Ceiling }),
      setVt2Ceiling: (vt2Ceiling) => set({ vt2Ceiling }),
      setUseThresholdsForAll: (useThresholdsForAll) => set({ useThresholdsForAll }),

      // Advanced params actions
      setAdvancedParams: (params) =>
        set((state) => ({
          advancedParams: { ...state.advancedParams, ...params },
        })),

      // Calibration actions
      setPendingVEPrompt: (pendingVEPrompt) => set({ pendingVEPrompt }),
      clearVEPrompt: () => set({ pendingVEPrompt: null }),

      // Reset
      reset: () => set(initialState),
    }),
    {
      name: "vt-check-settings",
      partialize: (state) => ({
        vt1Ceiling: state.vt1Ceiling,
        vt2Ceiling: state.vt2Ceiling,
        useThresholdsForAll: state.useThresholdsForAll,
        advancedParams: state.advancedParams,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);

// =============================================================================
// Selectors
// =============================================================================

/**
 * Get the currently selected interval result
 */
export const useSelectedIntervalResult = (): IntervalResult | null => {
  const { analysisResult, selectedIntervalId } = useRunStore();
  if (!analysisResult || selectedIntervalId === null) return null;
  return (
    analysisResult.results.find((r) => r.interval_num === selectedIntervalId) ??
    null
  );
};

/**
 * Get the currently hovered interval result
 */
export const useHoveredIntervalResult = (): IntervalResult | null => {
  const { analysisResult, hoveredIntervalId } = useRunStore();
  if (!analysisResult || hoveredIntervalId === null) return null;
  return (
    analysisResult.results.find((r) => r.interval_num === hoveredIntervalId) ??
    null
  );
};

/**
 * Check if an interval is selected or hovered
 */
export const useIsIntervalHighlighted = (intervalNum: number): boolean => {
  const { selectedIntervalId, hoveredIntervalId } = useRunStore();
  return selectedIntervalId === intervalNum || hoveredIntervalId === intervalNum;
};
