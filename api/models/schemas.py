"""
Request/Response Schemas for VT Threshold Analyzer API
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from .enums import RunType, IntervalStatus
from .params import AnalysisParams


# =============================================================================
# Data Models
# =============================================================================

class Interval(BaseModel):
    """Represents a work interval in the run."""
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    start_idx: int = Field(description="Start index in breath data")
    end_idx: int = Field(description="End index in breath data")
    interval_num: int = Field(description="1-indexed interval number")


class ChartData(BaseModel):
    """Chart visualization data for an interval."""
    time_values: List[float] = Field(description="Bin timestamps (seconds)")
    ve_binned: List[float] = Field(description="Binned VE values after filtering")
    cusum_values: List[float] = Field(description="CUSUM values over time")
    expected_ve: List[float] = Field(description="Expected VE trajectory")

    # Slope segment lines (for visualization)
    segment1_times: Optional[List[float]] = Field(
        default=None,
        description="Phase II (ramp-up) segment times"
    )
    segment1_ve: Optional[List[float]] = Field(
        default=None,
        description="Phase II (ramp-up) segment VE values"
    )
    segment2_times: Optional[List[float]] = Field(
        default=None,
        description="Phase III first segment times"
    )
    segment2_ve: Optional[List[float]] = Field(
        default=None,
        description="Phase III first segment VE values"
    )
    segment3_times: Optional[List[float]] = Field(
        default=None,
        description="Phase III second segment times (after 2nd hinge)"
    )
    segment3_ve: Optional[List[float]] = Field(
        default=None,
        description="Phase III second segment VE values"
    )

    # Combined slope line for simpler rendering
    slope_line_times: List[float] = Field(
        default_factory=list,
        description="Combined slope line times"
    )
    slope_line_ve: List[float] = Field(
        default_factory=list,
        description="Combined slope line VE values"
    )


class IntervalResult(BaseModel):
    """Analysis result for a single interval."""

    # Basic info
    interval_num: int = Field(description="1-indexed interval number")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")

    # Classification
    status: IntervalStatus = Field(description="Classification status")

    # VE metrics
    baseline_ve: float = Field(description="Calibration baseline VE (L/min)")
    avg_ve: float = Field(description="Average VE in analysis window (L/min)")
    peak_ve: float = Field(description="Peak VE in interval (L/min)")
    initial_ve: float = Field(description="Initial VE after calibration (L/min)")
    terminal_ve: float = Field(description="Average VE in last 60s (L/min)")
    last_60s_avg_ve: float = Field(description="Last 60s average for cumulative drift")
    last_30s_avg_ve: float = Field(default=0.0, description="Last 30s average for ceiling CUSUM drift")

    # Drift metrics
    ve_drift_rate: float = Field(description="VE drift rate (L/min per minute)")
    ve_drift_pct: float = Field(description="VE drift as % of baseline per minute")

    # CUSUM metrics
    peak_cusum: float = Field(description="Peak CUSUM value")
    final_cusum: float = Field(description="Final CUSUM value")
    cusum_threshold: float = Field(description="CUSUM threshold (H)")
    alarm_time: Optional[float] = Field(
        default=None,
        description="Time when CUSUM alarm triggered (seconds)"
    )
    cusum_recovered: bool = Field(
        default=False,
        description="True if CUSUM alarm triggered but recovered"
    )

    # Analysis type flags
    is_ceiling_based: bool = Field(
        default=False,
        description="True if ceiling-based analysis was used"
    )
    is_segmented: bool = Field(
        default=False,
        description="True if segmented regression was used"
    )

    # Phase III detection
    phase3_onset_rel: Optional[float] = Field(
        default=None,
        description="Phase III onset relative to interval start (seconds)"
    )

    # Second hinge detection (slope change)
    hinge2_time_rel: Optional[float] = Field(
        default=None,
        description="2nd hinge time relative to interval start (seconds)"
    )
    slope1_pct: Optional[float] = Field(
        default=None,
        description="Slope before 2nd hinge (% of baseline per minute)"
    )
    slope2_pct: Optional[float] = Field(
        default=None,
        description="Slope after 2nd hinge (% of baseline per minute)"
    )
    split_slope_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of slope2/slope1"
    )
    hinge2_detected: bool = Field(
        default=False,
        description="True if 2nd hinge was successfully detected"
    )

    # Speed (from iOS CSV)
    speed: Optional[float] = Field(
        default=None,
        description="Speed for this interval (mph)"
    )

    # Observed noise for calibration
    observed_sigma_pct: Optional[float] = Field(
        default=None,
        description="Observed noise as % of baseline (MADSD method)"
    )

    # Chart data (for visualization)
    chart_data: ChartData = Field(description="Data for chart rendering")

    # Raw breath data (for scatter plot dots)
    breath_times: List[float] = Field(
        default_factory=list,
        description="Breath timestamps (for scatter dots)"
    )
    ve_median: List[float] = Field(
        default_factory=list,
        description="Median-filtered VE values (for scatter dots)"
    )


class CumulativeDriftResult(BaseModel):
    """Cumulative drift analysis result across all intervals."""
    slope_abs: float = Field(description="Slope in L/min per minute")
    slope_pct: float = Field(description="Slope as % of baseline per minute")
    baseline_ve: float = Field(description="Baseline VE (interval 1 terminal)")
    pvalue: float = Field(description="Statistical significance p-value")

    # Data for visualization
    interval_end_times: List[float] = Field(
        description="X-values: elapsed time at end of each interval"
    )
    interval_avg_ve: List[float] = Field(
        description="Y-values: last-60s average VE for each interval"
    )
    line_times: List[float] = Field(
        description="X-values for regression line"
    )
    line_ve: List[float] = Field(
        description="Y-values for regression line"
    )


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ParseCSVRequest(BaseModel):
    """Request to parse a CSV file."""
    csv_content: str = Field(description="Raw CSV content as string")
    format: Optional[str] = Field(
        default=None,
        description="CSV format ('ios' or 'vitalpro'). If None, auto-detect."
    )


class ParseCSVResponse(BaseModel):
    """Response from CSV parsing."""
    success: bool
    format: str = Field(description="Detected format ('ios' or 'vitalpro')")
    total_breaths: int = Field(description="Total number of breaths parsed")
    duration_seconds: float = Field(description="Total recording duration")
    has_power_data: bool = Field(description="Whether power data is available")

    # Run parameters (from iOS CSV metadata)
    detected_run_type: Optional[str] = Field(
        default=None,
        description="Run type from metadata"
    )
    detected_intervals: Optional[int] = Field(
        default=None,
        description="Number of intervals from metadata"
    )
    detected_interval_duration: Optional[float] = Field(
        default=None,
        description="Interval duration (min) from metadata"
    )
    detected_recovery_duration: Optional[float] = Field(
        default=None,
        description="Recovery duration (min) from metadata"
    )
    detected_vt1_threshold: Optional[float] = Field(
        default=None,
        description="VT1 VE threshold from metadata"
    )
    detected_vt2_threshold: Optional[float] = Field(
        default=None,
        description="VT2 VE threshold from metadata"
    )
    detected_speeds: Optional[List[float]] = Field(
        default=None,
        description="Speeds from metadata"
    )


class DetectIntervalsRequest(BaseModel):
    """Request to detect intervals from power data."""
    csv_content: str = Field(description="Raw CSV content")


class DetectIntervalsResponse(BaseModel):
    """Response from interval detection."""
    run_type: RunType
    num_intervals: int
    interval_duration_min: float
    recovery_duration_min: float
    total_duration_min: float
    detection_method: str = Field(
        description="Method used: 'power_kmeans', 'metadata', or 'manual'"
    )


class AnalysisRequest(BaseModel):
    """Request to run full CUSUM analysis."""
    csv_content: str = Field(description="Raw CSV content")
    csv_format: Optional[str] = Field(
        default=None,
        description="CSV format ('ios' or 'vitalpro'). If None, auto-detect."
    )
    run_type: RunType = Field(description="Run type (VT1 or VT2)")
    num_intervals: int = Field(description="Number of intervals")
    interval_duration_min: float = Field(description="Interval duration in minutes")
    recovery_duration_min: float = Field(
        default=0.0,
        description="Recovery duration in minutes"
    )
    params: Optional[AnalysisParams] = Field(
        default=None,
        description="Analysis parameters. If None, use defaults."
    )


class BreathData(BaseModel):
    """Breath-by-breath data for chart visualization."""
    times: List[float] = Field(description="All breath timestamps")
    ve_median: List[float] = Field(description="Median-filtered VE values")
    bin_times: List[float] = Field(description="Bin timestamps")
    ve_binned: List[float] = Field(description="Binned VE values")
    hr: Optional[List[float]] = Field(default=None, description="Heart rate values at each breath")


class AnalysisResponse(BaseModel):
    """Response from full analysis."""
    success: bool
    run_type: RunType
    intervals: List[Interval]
    results: List[IntervalResult]
    cumulative_drift: Optional[CumulativeDriftResult] = Field(
        default=None,
        description="Cumulative drift result (VT2 multi-interval only)"
    )

    # Full breath data for chart
    breath_data: BreathData

    # Detected parameters (for reference)
    detected_phase3_onset: Optional[float] = Field(
        default=None,
        description="Auto-detected Phase III onset (seconds)"
    )

    # Error info
    error: Optional[str] = Field(
        default=None,
        description="Error message if analysis failed"
    )


# =============================================================================
# Calibration Schemas
# =============================================================================

class NIGPosteriorSchema(BaseModel):
    """Normal-Inverse-Gamma posterior for a single parameter."""
    mu: float = Field(default=0.0, description="Posterior mean estimate")
    kappa: float = Field(default=1.0, description="Precision of mean")
    alpha: float = Field(default=2.0, description="Shape parameter")
    beta: float = Field(default=1.0, description="Scale parameter")
    n_obs: int = Field(default=0, description="Observation count")


class DomainPosteriorSchema(BaseModel):
    """All NIG posteriors for a single intensity domain."""
    expected_drift: NIGPosteriorSchema = Field(default_factory=NIGPosteriorSchema)
    max_drift: NIGPosteriorSchema = Field(default_factory=NIGPosteriorSchema)
    sigma: NIGPosteriorSchema = Field(default_factory=NIGPosteriorSchema)
    split_ratio: NIGPosteriorSchema = Field(default_factory=NIGPosteriorSchema)


class VEThresholdStateSchema(BaseModel):
    """State for a VE threshold (VT1 or VT2) using Anchor & Pull method."""
    current_value: float = Field(default=60.0, description="Current user-approved threshold (anchor)")
    posterior: NIGPosteriorSchema = Field(default_factory=NIGPosteriorSchema)
    anchor_kappa: float = Field(default=4.0, description="Virtual sample size for anchoring")


class CalibrationStateSchema(BaseModel):
    """Complete calibration state for a user."""
    moderate: DomainPosteriorSchema = Field(default_factory=DomainPosteriorSchema)
    heavy: DomainPosteriorSchema = Field(default_factory=DomainPosteriorSchema)
    severe: DomainPosteriorSchema = Field(default_factory=DomainPosteriorSchema)
    vt1_ve: VEThresholdStateSchema = Field(default_factory=VEThresholdStateSchema)
    vt2_ve: VEThresholdStateSchema = Field(default_factory=VEThresholdStateSchema)
    enabled: bool = Field(default=True, description="Whether calibration is active")
    last_updated: Optional[str] = Field(default=None, description="ISO timestamp")
    run_counts: dict = Field(default_factory=lambda: {'moderate': 0, 'heavy': 0, 'severe': 0})


class CalibrationParamsResponse(BaseModel):
    """Calibrated parameters for iOS app sync."""
    vt1_ve: float = Field(description="VT1 VE threshold (L/min)")
    vt2_ve: float = Field(description="VT2 VE threshold (L/min)")
    sigma_pct_moderate: float = Field(description="Sigma % for Moderate domain")
    sigma_pct_heavy: float = Field(description="Sigma % for Heavy domain")
    sigma_pct_severe: float = Field(description="Sigma % for Severe domain")
    expected_drift_moderate: float = Field(description="Expected drift %/min for Moderate domain")
    expected_drift_heavy: float = Field(description="Expected drift %/min for Heavy domain")
    expected_drift_severe: float = Field(description="Expected drift %/min for Severe domain")
    max_drift_moderate: float = Field(description="Max drift % threshold for Moderate domain")
    max_drift_heavy: float = Field(description="Max drift % threshold for Heavy domain")
    max_drift_severe: float = Field(description="Max drift % threshold for Severe domain")
    split_ratio_moderate: float = Field(description="Split slope ratio for Moderate domain")
    split_ratio_heavy: float = Field(description="Split slope ratio for Heavy domain")
    split_ratio_severe: float = Field(description="Split slope ratio for Severe domain")
    enabled: bool = Field(default=True, description="Whether calibration is active")
    last_updated: Optional[str] = Field(default=None, description="ISO timestamp")


class CalibrationUpdateRequest(BaseModel):
    """Request to update calibration from analysis results."""
    user_id: str = Field(description="User/device identifier")
    run_type: RunType = Field(description="Intensity domain")
    interval_results: List[dict] = Field(description="List of interval result dicts")


class CalibrationUpdateResponse(BaseModel):
    """Response from calibration update."""
    success: bool
    run_count: int = Field(description="New qualifying run count for domain")
    ve_prompt: Optional[dict] = Field(
        default=None,
        description="VE threshold prompt if change >= 1 L/min"
    )


class VEApprovalRequest(BaseModel):
    """Request to approve/reject VE threshold change."""
    user_id: str = Field(description="User/device identifier")
    threshold: str = Field(description="'vt1' or 'vt2'")
    approved: bool = Field(description="User's approval decision")
    proposed_value: float = Field(description="The proposed threshold value")


class AdvancedParamsRequest(BaseModel):
    """Request to manually set advanced calibration parameters."""
    user_id: str = Field(description="User/device identifier")
    # VT1 (Moderate) params
    sigma_pct_vt1: float = Field(description="Sigma % for VT1/Moderate")
    expected_drift_vt1: float = Field(description="Expected drift %/min for VT1/Moderate")
    h_multiplier_vt1: float = Field(description="H multiplier for VT1/Moderate")
    # VT2 (Heavy/Severe) params
    sigma_pct_vt2: float = Field(description="Sigma % for VT2/Heavy")
    expected_drift_vt2: float = Field(description="Expected drift %/min for VT2/Heavy")
    max_drift_vt2: float = Field(description="Max drift % for VT2/Heavy")
    split_ratio_vt2: float = Field(description="Split ratio for VT2/Heavy")
    h_multiplier_vt2: float = Field(description="H multiplier for VT2/Heavy")
