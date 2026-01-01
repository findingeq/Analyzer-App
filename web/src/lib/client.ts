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
