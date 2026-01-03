"""
Services for VT Threshold Analyzer API
"""

from .csv_parser import (
    detect_csv_format,
    parse_csv_auto,
    parse_vitalpro_csv,
    parse_ios_csv,
)
from .signal_filter import (
    apply_median_filter,
    apply_time_binning,
    apply_hampel_filter,
    apply_hybrid_filtering,
)
from .interval_detector import (
    detect_intervals,
    create_intervals_from_params,
)
from .regression import (
    fit_single_slope,
    fit_robust_hinge,
    fit_second_hinge,
    # TESTING - Remove after slope model selection
    fit_second_hinge_constrained,
    fit_quadratic_slope,
)
from .cusum_analyzer import (
    analyze_interval_segmented,
    analyze_interval_ceiling,
)
from .cumulative_drift import (
    compute_cumulative_drift,
)

__all__ = [
    # CSV Parser
    "detect_csv_format",
    "parse_csv_auto",
    "parse_vitalpro_csv",
    "parse_ios_csv",
    # Signal Filter
    "apply_median_filter",
    "apply_time_binning",
    "apply_hampel_filter",
    "apply_hybrid_filtering",
    # Interval Detector
    "detect_intervals",
    "create_intervals_from_params",
    # Regression
    "fit_single_slope",
    "fit_robust_hinge",
    "fit_second_hinge",
    # TESTING - Remove after slope model selection
    "fit_second_hinge_constrained",
    "fit_quadratic_slope",
    # CUSUM Analyzer
    "analyze_interval_segmented",
    "analyze_interval_ceiling",
    # Cumulative Drift
    "compute_cumulative_drift",
]
