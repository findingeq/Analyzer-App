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
