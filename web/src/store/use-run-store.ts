/**
 * Zustand Store for VT Threshold Analyzer
 * Manages global UI state and analysis data
 */

import { create } from "zustand";
import type {
  RunType,
  AnalysisResponse,
  ParseCSVResponse,
  IntervalResult,
} from "@/lib/api-types";

// =============================================================================
// Store State Interface
// =============================================================================

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

  // Zoom state for chart
  zoomStart: number;
  zoomEnd: number;
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

  // Zoom actions
  setZoomRange: (start: number, end: number) => void;
  resetZoom: () => void;

  // Reset
  reset: () => void;
}

// =============================================================================
// Initial State
// =============================================================================

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
  zoomStart: 0,
  zoomEnd: 100,
};

// =============================================================================
// Store Creation
// =============================================================================

export const useRunStore = create<RunState & RunActions>((set) => ({
  ...initialState,

  // File actions
  setCSVContent: (content, fileName) =>
    set({
      csvContent: content,
      fileName: fileName ?? null,
      // Reset analysis when new file is loaded
      analysisResult: null,
      selectedIntervalId: null,
      hoveredIntervalId: null,
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
      // Auto-select first interval when results arrive
      selectedIntervalId: analysisResult?.results?.[0]?.interval_num ?? null,
    }),
  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),

  // UI actions
  setSelectedInterval: (selectedIntervalId) => set({ selectedIntervalId }),
  setHoveredInterval: (hoveredIntervalId) => set({ hoveredIntervalId }),
  toggleSlopeLines: () => set((state) => ({ showSlopeLines: !state.showSlopeLines })),
  toggleCusum: () => set((state) => ({ showCusum: !state.showCusum })),

  // Zoom actions
  setZoomRange: (zoomStart, zoomEnd) => set({ zoomStart, zoomEnd }),
  resetZoom: () => set({ zoomStart: 0, zoomEnd: 100 }),

  // Reset
  reset: () => set(initialState),
}));

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
