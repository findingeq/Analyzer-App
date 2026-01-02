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
        default=180.0,
        description="Maximum time for Phase III detection for VT2 (seconds)"
    )
    phase3_default: float = Field(
        default=150.0,
        description="Default Phase III onset if detection fails (seconds)"
    )

    # VT1-specific parameters (moderate domain)
    # For runs < 20 min: search 90s - 6min for Phase II end
    vt1_phase3_min_time_short: float = Field(
        default=90.0,
        description="Minimum time for Phase III detection for VT1 runs < 20 min (seconds)"
    )
    vt1_phase3_max_time_short: float = Field(
        default=360.0,
        description="Maximum time for Phase III detection for VT1 runs < 20 min (seconds)"
    )
    # For runs >= 20 min: search 90s - 15min for Phase II end (thermal equilibration)
    vt1_phase3_min_time_long: float = Field(
        default=90.0,
        description="Minimum time for Phase III detection for VT1 runs >= 20 min (seconds)"
    )
    vt1_phase3_max_time_long: float = Field(
        default=900.0,
        description="Maximum time for Phase III detection for VT1 runs >= 20 min (15 min in seconds)"
    )
    vt1_phase3_default: float = Field(
        default=360.0,
        description="Default Phase III onset for VT1 if detection fails (6 min in seconds)"
    )
    vt1_calibration_duration: float = Field(
        default=60.0,
        description="Calibration window for VT1 runs (1 minute)"
    )

    # VT2-specific parameters
    vt2_calibration_duration: float = Field(
        default=60.0,
        description="Calibration duration after Phase III onset for VT2 (1 minute)"
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
