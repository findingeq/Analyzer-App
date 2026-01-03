"""
Analysis Parameters for VT Threshold Analyzer
"""

from typing import Optional
from pydantic import BaseModel, Field


class AnalysisParams(BaseModel):
    """Configuration parameters for CUSUM analysis."""

    # Ramp-up period (blanking) parameters
    phase3_onset_override: Optional[float] = Field(
        default=None,
        description="User override for ramp-up period end (seconds)"
    )
    phase3_min_time: float = Field(
        default=90.0,
        description="Minimum time for Phase III detection for VT2 (seconds)"
    )
    phase3_max_time: float = Field(
        default=210.0,
        description="Maximum time for Phase III detection (seconds)"
    )
    phase3_default: float = Field(
        default=150.0,
        description="Default Phase III onset if detection fails (seconds)"
    )

    # Calibration duration (same for all domains)
    vt2_calibration_duration: float = Field(
        default=60.0,
        description="Calibration duration after Phase III onset (1 minute)"
    )

    # Legacy calibration (fallback)
    calibration_duration: float = Field(
        default=30.0,
        description="Legacy calibration duration (seconds)"
    )

    # CUSUM thresholds
    h_multiplier_vt1: float = Field(
        default=5.0,
        description="CUSUM threshold multiplier for VT1"
    )
    h_multiplier_vt2: float = Field(
        default=5.0,
        description="CUSUM threshold multiplier for VT2"
    )
    slack_multiplier: float = Field(
        default=0.5,
        description="CUSUM slack parameter multiplier"
    )

    # Expected drift rates (percentage of baseline VE per minute)
    expected_drift_pct_vt1: float = Field(
        default=0.3,
        description="Expected drift % per minute for VT1 (moderate domain)"
    )
    expected_drift_pct_vt2: float = Field(
        default=1.0,
        description="Expected drift % per minute for VT2 (heavy domain)"
    )

    # Max drift thresholds for classification
    max_drift_pct_vt1: float = Field(
        default=1.0,
        description="Max acceptable drift % per minute for VT1"
    )
    max_drift_pct_vt2: float = Field(
        default=3.0,
        description="Max acceptable drift % per minute for VT2"
    )

    # Sigma percentages (for CUSUM threshold calculation)
    sigma_pct_vt1: float = Field(
        default=7.0,
        description="Sigma as % of baseline VE for VT1"
    )
    sigma_pct_vt2: float = Field(
        default=4.0,
        description="Sigma as % of baseline VE for VT2"
    )

    # Signal filtering parameters
    median_window: int = Field(
        default=9,
        description="Rolling median filter window size (breaths)"
    )
    bin_size: float = Field(
        default=4.0,
        description="Time bin size (seconds)"
    )
    hampel_window_sec: float = Field(
        default=30.0,
        description="Hampel filter window size (seconds)"
    )
    hampel_n_sigma: float = Field(
        default=3.0,
        description="Hampel filter sigma threshold"
    )

    # Slope model mode for Heavy/Severe intervals (TESTING - remove after selection)
    # Options: "single_slope", "two_hinge", "two_hinge_constrained", "quadratic"
    slope_model_mode: str = Field(
        default="two_hinge",
        description="Slope model for Heavy/Severe: single_slope, two_hinge, two_hinge_constrained, quadratic"
    )

    # TESTING - Huber loss delta parameter for regression smoothness
    # Lower = more robust to outliers but may underfit; Higher = more sensitive but may overfit
    huber_delta: float = Field(
        default=5.0,
        description="Huber loss delta threshold (L/min) for robust regression"
    )

    # TESTING - LOESS smoothness parameter for visual trend line
    # Lower = more wiggly (follows data closely); Higher = smoother (more averaged)
    loess_frac: float = Field(
        default=0.4,
        description="LOESS smoothing fraction (0.1-0.8) for visual trend line"
    )

    # Ceiling-based analysis parameters
    vt1_ve_ceiling: float = Field(
        default=100.0,
        description="User-provided VT1 VE ceiling (L/min)"
    )
    vt2_ve_ceiling: float = Field(
        default=120.0,
        description="User-provided VT2 VE ceiling (L/min)"
    )
    use_thresholds_for_all: bool = Field(
        default=False,
        description="If True, use ceiling-based analysis for all intervals"
    )
    ceiling_warmup_sec: float = Field(
        default=20.0,
        description="Warm-up period for ceiling-based CUSUM (seconds)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "vt1_ve_ceiling": 100.0,
                "vt2_ve_ceiling": 120.0,
                "use_thresholds_for_all": False,
                "h_multiplier_vt1": 5.0,
                "h_multiplier_vt2": 5.0,
            }
        }
