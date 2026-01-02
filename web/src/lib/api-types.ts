/**
 * API Types for VT Threshold Analyzer
 * Generated from backend Pydantic schemas
 */

// =============================================================================
// Enums (as const objects for erasableSyntaxOnly compatibility)
// =============================================================================

export const RunType = {
  MODERATE: "MODERATE",
  HEAVY: "HEAVY",
  SEVERE: "SEVERE",
} as const;

export type RunType = (typeof RunType)[keyof typeof RunType];

export const IntervalStatus = {
  BELOW_THRESHOLD: "BELOW_THRESHOLD",
  BORDERLINE: "BORDERLINE",
  ABOVE_THRESHOLD: "ABOVE_THRESHOLD",
} as const;

export type IntervalStatus = (typeof IntervalStatus)[keyof typeof IntervalStatus];

// =============================================================================
// Analysis Parameters
// =============================================================================

export interface AnalysisParams {
  // Ramp-up period (blanking) parameters
  phase3_onset_override?: number | null;
  phase3_min_time?: number;
  phase3_max_time?: number;
  phase3_default?: number;

  // VT1-specific parameters
  vt1_blanking_time?: number;
  vt1_calibration_short?: number;
  vt1_calibration_long?: number;

  // VT2-specific parameters
  vt2_calibration_duration?: number;

  // Legacy calibration
  calibration_duration?: number;

  // CUSUM thresholds
  h_multiplier_vt1?: number;
  h_multiplier_vt2?: number;
  slack_multiplier?: number;

  // Expected drift rates
  expected_drift_pct_vt1?: number;
  expected_drift_pct_vt2?: number;

  // Max drift thresholds
  max_drift_pct_vt1?: number;
  max_drift_pct_vt2?: number;

  // Sigma percentages
  sigma_pct_vt1?: number;
  sigma_pct_vt2?: number;

  // Signal filtering
  median_window?: number;
  bin_size?: number;
  hampel_window_sec?: number;
  hampel_n_sigma?: number;

  // Ceiling-based analysis
  vt1_ve_ceiling?: number;
  vt2_ve_ceiling?: number;
  use_thresholds_for_all?: boolean;
  ceiling_warmup_sec?: number;
}

// =============================================================================
// Data Models
// =============================================================================

export interface Interval {
  start_time: number;
  end_time: number;
  start_idx: number;
  end_idx: number;
  interval_num: number;
}

export interface ChartData {
  time_values: number[];
  ve_binned: number[];
  cusum_values: number[];
  expected_ve: number[];

  // Slope segment lines
  segment1_times?: number[] | null;
  segment1_ve?: number[] | null;
  segment2_times?: number[] | null;
  segment2_ve?: number[] | null;
  segment3_times?: number[] | null;
  segment3_ve?: number[] | null;

  // Combined slope line
  slope_line_times: number[];
  slope_line_ve: number[];
}

export interface IntervalResult {
  // Basic info
  interval_num: number;
  start_time: number;
  end_time: number;

  // Classification
  status: IntervalStatus;

  // VE metrics
  baseline_ve: number;
  avg_ve: number;
  peak_ve: number;
  initial_ve: number;
  terminal_ve: number;
  last_60s_avg_ve: number;
  last_30s_avg_ve: number;

  // Drift metrics
  ve_drift_rate: number;
  ve_drift_pct: number;

  // CUSUM metrics
  peak_cusum: number;
  final_cusum: number;
  cusum_threshold: number;
  alarm_time?: number | null;
  cusum_recovered: boolean;

  // Analysis type flags
  is_ceiling_based: boolean;
  is_segmented: boolean;

  // Phase III detection
  phase3_onset_rel?: number | null;

  // Second hinge detection
  hinge2_time_rel?: number | null;
  slope1_pct?: number | null;
  slope2_pct?: number | null;
  split_slope_ratio?: number | null;
  hinge2_detected: boolean;

  // Speed
  speed?: number | null;

  // Observed noise for calibration
  observed_sigma_pct?: number | null;

  // Chart data
  chart_data: ChartData;

  // Raw breath data
  breath_times: number[];
  ve_median: number[];
}

export interface CumulativeDriftResult {
  slope_abs: number;
  slope_pct: number;
  baseline_ve: number;
  pvalue: number;

  // Visualization data
  interval_end_times: number[];
  interval_avg_ve: number[];
  line_times: number[];
  line_ve: number[];
}

export interface BreathData {
  times: number[];
  ve_median: number[];
  bin_times: number[];
  ve_binned: number[];
  hr?: number[] | null;
}

// =============================================================================
// Request/Response Schemas
// =============================================================================

export interface ParseCSVRequest {
  csv_content: string;
  format?: string | null;
}

export interface ParseCSVResponse {
  success: boolean;
  format: string;
  total_breaths: number;
  duration_seconds: number;
  has_power_data: boolean;
  detected_run_type?: string | null;
  detected_intervals?: number | null;
  detected_interval_duration?: number | null;
  detected_recovery_duration?: number | null;
  detected_vt1_threshold?: number | null;
  detected_vt2_threshold?: number | null;
  detected_speeds?: number[] | null;
}

export interface DetectIntervalsRequest {
  csv_content: string;
}

export interface DetectIntervalsResponse {
  run_type: RunType;
  num_intervals: number;
  interval_duration_min: number;
  recovery_duration_min: number;
  total_duration_min: number;
  detection_method: string;
}

export interface AnalysisRequest {
  csv_content: string;
  csv_format?: string | null;
  run_type: RunType;
  num_intervals: number;
  interval_duration_min: number;
  recovery_duration_min?: number;
  params?: AnalysisParams | null;
}

export interface AnalysisResponse {
  success: boolean;
  run_type: RunType;
  intervals: Interval[];
  results: IntervalResult[];
  cumulative_drift?: CumulativeDriftResult | null;
  breath_data: BreathData;
  detected_phase3_onset?: number | null;
  error?: string | null;
}

// =============================================================================
// API Error
// =============================================================================

export interface APIError {
  detail: string;
}

// =============================================================================
// Calibration Types
// =============================================================================

export interface NIGPosterior {
  mu: number;
  kappa: number;
  alpha: number;
  beta: number;
  n_obs: number;
}

export interface DomainPosterior {
  expected_drift: NIGPosterior;
  max_drift: NIGPosterior;
  sigma: NIGPosterior;
  split_ratio: NIGPosterior;
}

export interface VEThresholdState {
  current_value: number;
  posterior: NIGPosterior;
  anchor_kappa: number;
}

export interface CalibrationState {
  moderate: DomainPosterior;
  heavy: DomainPosterior;
  severe: DomainPosterior;
  vt1_ve: VEThresholdState;
  vt2_ve: VEThresholdState;
  enabled: boolean;
  last_updated?: string | null;
  run_counts: Record<string, number>;
}

export interface CalibrationParamsResponse {
  vt1_ve: number;
  vt2_ve: number;
  sigma_pct_moderate: number;
  sigma_pct_heavy: number;
  sigma_pct_severe: number;
  expected_drift_moderate: number;
  expected_drift_heavy: number;
  expected_drift_severe: number;
  max_drift_moderate: number;
  max_drift_heavy: number;
  max_drift_severe: number;
  enabled: boolean;
  last_updated?: string | null;
}

export interface CalibrationUpdateRequest {
  user_id: string;
  run_type: RunType;
  interval_results: Array<{
    start_time: number;
    end_time: number;
    status: IntervalStatus;
    ve_drift_pct: number;
    avg_ve: number;
    split_slope_ratio?: number | null;
    sigma_pct: number;
  }>;
}

export interface CalibrationUpdateResponse {
  success: boolean;
  run_count: number;
  ve_prompt?: {
    threshold: "vt1" | "vt2";
    current_value: number;
    proposed_value: number;
    divergence: number;
  } | null;
}

export interface VEApprovalRequest {
  user_id: string;
  threshold: "vt1" | "vt2";
  approved: boolean;
  proposed_value: number;
}
