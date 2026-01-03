/**
 * API Client for VT Threshold Analyzer
 * Type-safe fetch wrapper using generated API types
 */

import type {
  ParseCSVRequest,
  ParseCSVResponse,
  DetectIntervalsRequest,
  DetectIntervalsResponse,
  AnalysisRequest,
  AnalysisResponse,
  APIError,
  CalibrationParamsResponse,
  CalibrationState,
  CalibrationUpdateRequest,
  CalibrationUpdateResponse,
  VEApprovalRequest,
  RunType,
  IntervalResult,
  IntervalStatus,
} from "./api-types";

const API_BASE = "/api";

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchApi<TRequest, TResponse>(
  endpoint: string,
  method: "GET" | "POST" | "PUT" | "DELETE",
  body?: TRequest
): Promise<TResponse> {
  const options: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
    },
  };

  if (body !== undefined) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, options);

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorData: APIError = await response.json();
      detail = errorData.detail || detail;
    } catch {
      // Use default detail if JSON parsing fails
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<TResponse>;
}

// =============================================================================
// Files API
// =============================================================================

/**
 * Parse a CSV file and return metadata
 */
export async function parseCSV(
  request: ParseCSVRequest
): Promise<ParseCSVResponse> {
  return fetchApi<ParseCSVRequest, ParseCSVResponse>(
    "/files/parse",
    "POST",
    request
  );
}

/**
 * Detect intervals from power data in a CSV file
 */
export async function detectIntervals(
  request: DetectIntervalsRequest
): Promise<DetectIntervalsResponse> {
  return fetchApi<DetectIntervalsRequest, DetectIntervalsResponse>(
    "/files/detect-intervals",
    "POST",
    request
  );
}

// =============================================================================
// Analysis API
// =============================================================================

/**
 * Run full CUSUM analysis on respiratory data
 */
export async function runAnalysis(
  request: AnalysisRequest
): Promise<AnalysisResponse> {
  return fetchApi<AnalysisRequest, AnalysisResponse>(
    "/analysis/run",
    "POST",
    request
  );
}

// =============================================================================
// File Upload Helper
// =============================================================================

/**
 * Read a file and return its content as a string
 */
export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

// =============================================================================
// Cloud Sessions API
// =============================================================================

export interface SessionSummary {
  date?: string | null;
  run_type?: string | null;
  vt1_threshold?: number | null;
  vt2_threshold?: number | null;
  speed?: number | null;
  duration_seconds?: number | null;
  num_intervals?: number | null;
  interval_duration_min?: number | null;
  recovery_duration_min?: number | null;
  intensity?: string | null;
  avg_pace_min_per_mile?: number | null;
  // Analysis results (populated after analysis is run)
  observed_sigma_pct?: number | null;
  observed_drift_pct?: number | null;
  exclude_from_calibration?: boolean;
}

export interface SessionInfo {
  session_id: string;
  filename: string;
  uploaded_at: string;
  size_bytes: number;
  summary?: SessionSummary | null;
}

export interface SessionContent {
  session_id: string;
  csv_content: string;
}

/**
 * List all cloud sessions from Firebase Storage
 */
export async function listSessions(): Promise<SessionInfo[]> {
  return fetchApi<undefined, SessionInfo[]>("/sessions", "GET");
}

/**
 * Get CSV content for a specific cloud session
 */
export async function getSession(sessionId: string): Promise<SessionContent> {
  return fetchApi<undefined, SessionContent>(`/sessions/${encodeURIComponent(sessionId)}`, "GET");
}

/**
 * Update a session with analysis results (sigma %, drift %)
 * Called after analysis is run to store observed values for display in run list
 */
export async function updateSessionAnalysis(
  sessionId: string,
  observedSigmaPct: number | null,
  observedDriftPct: number | null
): Promise<{ success: boolean; session_id: string; message: string }> {
  return fetchApi<
    { session_id: string; observed_sigma_pct: number | null; observed_drift_pct: number | null },
    { success: boolean; session_id: string; message: string }
  >(
    `/sessions/${encodeURIComponent(sessionId)}/analysis`,
    "POST",
    {
      session_id: sessionId,
      observed_sigma_pct: observedSigmaPct,
      observed_drift_pct: observedDriftPct,
    }
  );
}

/**
 * Update a session's calibration exclusion status
 * When excluded, the session won't contribute to ML calibration updates
 */
export async function updateSessionCalibration(
  sessionId: string,
  excludeFromCalibration: boolean
): Promise<{ success: boolean; session_id: string; exclude_from_calibration: boolean; message: string }> {
  return fetchApi<
    { exclude_from_calibration: boolean },
    { success: boolean; session_id: string; exclude_from_calibration: boolean; message: string }
  >(
    `/sessions/${encodeURIComponent(sessionId)}/calibration`,
    "POST",
    {
      exclude_from_calibration: excludeFromCalibration,
    }
  );
}

// =============================================================================
// Calibration API
// =============================================================================

/**
 * Get or generate a unique user ID for calibration
 * Uses localStorage to persist across sessions
 */
export function getOrCreateUserId(): string {
  const key = "vt_calibration_user_id";
  let userId = localStorage.getItem(key);
  if (!userId) {
    // Generate a UUID v4-like identifier
    userId = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
    localStorage.setItem(key, userId);
  }
  return userId;
}

/**
 * Get calibrated parameters for iOS app sync
 */
export async function getCalibrationParams(
  userId?: string
): Promise<CalibrationParamsResponse> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<undefined, CalibrationParamsResponse>(
    `/calibration/params?user_id=${encodeURIComponent(uid)}`,
    "GET"
  );
}

/**
 * Get full calibration state for a user
 */
export async function getCalibrationState(
  userId?: string
): Promise<CalibrationState> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<undefined, CalibrationState>(
    `/calibration/state?user_id=${encodeURIComponent(uid)}`,
    "GET"
  );
}

/**
 * Update calibration from analysis results
 */
export async function updateCalibration(
  runType: RunType,
  intervalResults: IntervalResult[],
  userId?: string
): Promise<CalibrationUpdateResponse> {
  const uid = userId || getOrCreateUserId();
  // Domain-specific sigma defaults: VT1/Moderate=7%, VT2/Heavy/Severe=4%
  const defaultSigmaPct = runType === "MODERATE" ? 7.0 : 4.0;
  const mappedResults: CalibrationUpdateRequest["interval_results"] = intervalResults.map((r) => ({
    start_time: r.start_time,
    end_time: r.end_time,
    status: r.status as IntervalStatus,
    ve_drift_pct: r.ve_drift_pct,
    avg_ve: r.avg_ve,
    split_slope_ratio: r.split_slope_ratio,
    sigma_pct: r.observed_sigma_pct ?? defaultSigmaPct, // MADSD-calculated, domain-specific fallback
  }));
  return fetchApi<CalibrationUpdateRequest, CalibrationUpdateResponse>(
    "/calibration/update",
    "POST",
    {
      user_id: uid,
      run_type: runType,
      interval_results: mappedResults,
    }
  );
}

/**
 * Approve or reject a VE threshold change
 */
export async function approveVEThreshold(
  threshold: "vt1" | "vt2",
  approved: boolean,
  proposedValue: number,
  userId?: string
): Promise<{ success: boolean; approved: boolean; new_value: number }> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<VEApprovalRequest, { success: boolean; approved: boolean; new_value: number }>(
    "/calibration/approve-ve",
    "POST",
    {
      user_id: uid,
      threshold,
      approved,
      proposed_value: proposedValue,
    }
  );
}

/**
 * Reset calibration to defaults
 */
export async function resetCalibration(
  userId?: string
): Promise<{ success: boolean; message: string }> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<undefined, { success: boolean; message: string }>(
    `/calibration/reset?user_id=${encodeURIComponent(uid)}`,
    "POST"
  );
}

/**
 * Manually set a VE threshold value
 * Called when user changes threshold in the UI - syncs to calibration baseline
 */
export async function setVEThresholdManual(
  threshold: "vt1" | "vt2",
  value: number,
  userId?: string
): Promise<{ success: boolean; threshold: string; value: number }> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<undefined, { success: boolean; threshold: string; value: number }>(
    `/calibration/set-ve-threshold?user_id=${encodeURIComponent(uid)}&threshold=${threshold}&value=${value}`,
    "POST"
  );
}

/**
 * Delete a cloud session
 */
export async function deleteSession(
  sessionId: string
): Promise<{ success: boolean; message: string }> {
  return fetchApi<undefined, { success: boolean; message: string }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    "DELETE"
  );
}

/**
 * Toggle calibration on or off
 * When disabled, learned data is preserved but system defaults are used
 */
export async function toggleCalibration(
  enabled: boolean,
  userId?: string
): Promise<{ success: boolean; enabled: boolean; message: string }> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<undefined, { success: boolean; enabled: boolean; message: string }>(
    `/calibration/toggle?user_id=${encodeURIComponent(uid)}&enabled=${enabled}`,
    "POST"
  );
}

/**
 * Manually set advanced calibration parameters
 * Called when user syncs advanced params to cloud
 */
export interface AdvancedParamsRequest {
  sigmaPctVt1: number;
  expectedDriftVt1: number;
  hMultiplierVt1: number;
  sigmaPctVt2: number;
  expectedDriftVt2: number;
  maxDriftVt2: number;
  splitRatioVt2: number;
  hMultiplierVt2: number;
}

export async function setAdvancedParams(
  params: AdvancedParamsRequest,
  userId?: string
): Promise<{ success: boolean; message: string }> {
  const uid = userId || getOrCreateUserId();
  return fetchApi<
    {
      user_id: string;
      sigma_pct_vt1: number;
      expected_drift_vt1: number;
      h_multiplier_vt1: number;
      sigma_pct_vt2: number;
      expected_drift_vt2: number;
      max_drift_vt2: number;
      split_ratio_vt2: number;
      h_multiplier_vt2: number;
    },
    { success: boolean; message: string }
  >(`/calibration/set-advanced-params`, "POST", {
    user_id: uid,
    sigma_pct_vt1: params.sigmaPctVt1,
    expected_drift_vt1: params.expectedDriftVt1,
    h_multiplier_vt1: params.hMultiplierVt1,
    sigma_pct_vt2: params.sigmaPctVt2,
    expected_drift_vt2: params.expectedDriftVt2,
    max_drift_vt2: params.maxDriftVt2,
    split_ratio_vt2: params.splitRatioVt2,
    h_multiplier_vt2: params.hMultiplierVt2,
  });
}
