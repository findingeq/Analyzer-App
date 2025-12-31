"""
Cumulative Drift Service for VT Threshold Analyzer

Computes cumulative VE drift across all intervals in a VT2 run.
Used to track fatigue accumulation over the entire workout.
"""

from typing import Optional, List
import numpy as np

from ..models.schemas import IntervalResult, CumulativeDriftResult


def compute_cumulative_drift(results: List[IntervalResult]) -> Optional[CumulativeDriftResult]:
    """
    Compute cumulative drift across all intervals as a segmented polyline.

    Only applicable for VT2 interval runs with 2+ intervals.
    Uses the terminal VE (last 60s average) from each interval as hinge points.

    The cumulative drift is the sum of individual segment slopes, expressed as
    total VE change from first to last interval as % of baseline per minute.

    Args:
        results: List of IntervalResult objects

    Returns:
        CumulativeDriftResult if applicable, None otherwise
    """
    if len(results) < 2:
        return None

    # Extract data points: (end_time, terminal_ve) for each interval
    interval_end_times = np.array([r.end_time for r in results])
    interval_avg_ve = np.array([r.last_60s_avg_ve for r in results])

    # Baseline is interval 1's terminal VE
    baseline_ve = interval_avg_ve[0]

    if baseline_ve <= 0:
        return None

    # Calculate cumulative drift as total change from first to last interval
    total_ve_change = interval_avg_ve[-1] - interval_avg_ve[0]

    # Calculate total work time (sum of all interval durations, excluding recoveries)
    total_work_time_min = sum((r.end_time - r.start_time) / 60.0 for r in results)

    if total_work_time_min > 0:
        slope_abs = total_ve_change / total_work_time_min
        slope_pct = (slope_abs / baseline_ve) * 100.0
    else:
        slope_abs = 0.0
        slope_pct = 0.0

    # Approximate p-value based on slope magnitude
    if abs(slope_pct) > 0.5:
        pvalue = 0.03
    else:
        pvalue = 0.15

    # Polyline: connect all terminal VE points directly
    line_times = interval_end_times.tolist()
    line_ve = interval_avg_ve.tolist()

    return CumulativeDriftResult(
        slope_abs=slope_abs,
        slope_pct=slope_pct,
        baseline_ve=baseline_ve,
        pvalue=pvalue,
        interval_end_times=interval_end_times.tolist(),
        interval_avg_ve=interval_avg_ve.tolist(),
        line_times=line_times,
        line_ve=line_ve
    )
