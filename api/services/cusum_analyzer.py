"""
CUSUM Analysis Service for VT Threshold Analyzer

Implements two CUSUM analysis approaches:
1. Segmented analysis - For intervals >= 6 min, uses drift detection
2. Ceiling-based analysis - For short intervals, uses absolute VE threshold
"""

from typing import Optional, List
import numpy as np
import pandas as pd

from ..models.enums import RunType, IntervalStatus
from ..models.params import AnalysisParams
from ..models.schemas import Interval, IntervalResult, ChartData

from .signal_filter import apply_hybrid_filtering
from .regression import fit_single_slope, fit_robust_hinge, fit_second_hinge


def _create_default_result(
    interval: Interval,
    is_ceiling_based: bool = False,
    is_segmented: bool = True,
    speed: Optional[float] = None
) -> IntervalResult:
    """Create a default result for invalid intervals."""
    return IntervalResult(
        interval_num=interval.interval_num,
        start_time=interval.start_time,
        end_time=interval.end_time,
        status=IntervalStatus.BELOW_THRESHOLD,
        baseline_ve=0,
        avg_ve=0,
        peak_ve=0,
        initial_ve=0,
        terminal_ve=0,
        last_60s_avg_ve=0,
        ve_drift_rate=0,
        ve_drift_pct=0,
        peak_cusum=0,
        final_cusum=0,
        cusum_threshold=0,
        alarm_time=None,
        cusum_recovered=False,
        is_ceiling_based=is_ceiling_based,
        is_segmented=is_segmented,
        phase3_onset_rel=None,
        hinge2_time_rel=None,
        slope1_pct=None,
        slope2_pct=None,
        split_slope_ratio=None,
        hinge2_detected=False,
        speed=speed,
        chart_data=ChartData(
            time_values=[interval.start_time],
            ve_binned=[0],
            cusum_values=[0],
            expected_ve=[0],
            slope_line_times=[],
            slope_line_ve=[]
        ),
        breath_times=[interval.start_time],
        ve_median=[0]
    )


def analyze_interval_segmented(
    breath_df: pd.DataFrame,
    interval: Interval,
    params: AnalysisParams,
    run_type: RunType,
    speed: Optional[float] = None
) -> IntervalResult:
    """
    Perform CUSUM analysis using robust hinge model to detect Phase III onset.

    Uses a piecewise linear "hinge" model with Huber loss to find the breakpoint
    where Phase II (on-kinetics) transitions to Phase III (steady state/drift).

    Calibration window is then set to [breakpoint, breakpoint + calibration_duration].

    Phase III detection is constrained to 90-240 seconds. If detection fails,
    falls back to 150 seconds default.

    Args:
        breath_df: DataFrame with breath-by-breath data
        interval: Interval to analyze
        params: Analysis parameters
        run_type: Type of run (VT1 or VT2)
        speed: Optional speed for this interval

    Returns:
        IntervalResult with analysis results
    """
    # Extract interval data
    idx_start = interval.start_idx
    idx_end = interval.end_idx

    if idx_end <= idx_start:
        return _create_default_result(interval, is_ceiling_based=False, is_segmented=True, speed=speed)

    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end]
    breath_times_raw = breath_df['breath_time'].values[idx_start:idx_end]

    # Relative time within interval (in seconds)
    rel_breath_times = breath_times_raw - breath_times_raw[0]

    # Apply hybrid filtering
    ve_median, bin_times_rel, ve_binned = apply_hybrid_filtering(ve_raw, rel_breath_times, params)

    # Convert bin times to minutes for drift calculations
    bin_times_min = bin_times_rel / 60.0

    # Calculate interval duration
    interval_duration_sec = interval.end_time - interval.start_time
    interval_duration_min = interval_duration_sec / 60.0

    # Phase III onset detection
    if run_type == RunType.VT1_STEADY:
        # VT1: Fixed blanking at 6 minutes
        phase3_onset_rel = params.vt1_blanking_time
        phase3_onset_time = phase3_onset_rel + breath_times_raw[0]

        # Fit simple linear model for segment 1
        ramp_mask = bin_times_rel <= phase3_onset_rel
        if np.sum(ramp_mask) >= 2:
            ramp_times = bin_times_rel[ramp_mask]
            ramp_ve = ve_binned[ramp_mask]
            b1 = (ramp_ve[-1] - ramp_ve[0]) / (ramp_times[-1] - ramp_times[0]) if (ramp_times[-1] - ramp_times[0]) > 0 else 0
            b0 = ramp_ve[0] - b1 * ramp_times[0]
        else:
            b0 = np.mean(ve_binned) if len(ve_binned) > 0 else 0
            b1 = 0
        b2 = 0
    else:
        # VT2: Use robust hinge model
        tau, b0, b1, b2, _, detection_succeeded = fit_robust_hinge(bin_times_rel, ve_binned, params)
        phase3_onset_rel = tau
        phase3_onset_time = tau + breath_times_raw[0]

    # Generate segment 1 line
    seg1_t_start = bin_times_rel[0]
    seg1_t_end = phase3_onset_rel
    segment1_times_rel = np.array([seg1_t_start, seg1_t_end])
    segment1_ve = b0 + b1 * segment1_times_rel

    # Set calibration window
    if run_type == RunType.VT1_STEADY:
        cal_start = phase3_onset_rel
        if interval_duration_min >= 15.0:
            cal_duration = params.vt1_calibration_long
        else:
            cal_duration = params.vt1_calibration_short
        cal_end = cal_start + cal_duration
    else:
        cal_start = phase3_onset_rel
        cal_end = phase3_onset_rel + params.vt2_calibration_duration

    # Determine domain-specific parameters
    if run_type == RunType.VT1_STEADY:
        h_mult = params.h_multiplier_vt1
        expected_drift_pct = params.expected_drift_pct_vt1
        sigma_pct = params.sigma_pct_vt1
        max_drift_threshold = params.max_drift_pct_vt1
    else:
        h_mult = params.h_multiplier_vt2
        expected_drift_pct = params.expected_drift_pct_vt2
        sigma_pct = params.sigma_pct_vt2
        max_drift_threshold = params.max_drift_pct_vt2

    # Find calibration window
    cal_mask = (bin_times_rel >= cal_start) & (bin_times_rel <= cal_end)
    n_cal_points = np.sum(cal_mask)

    if n_cal_points < 3:
        cal_mask = (bin_times_rel >= cal_start) & (bin_times_rel <= cal_start + 60)
        n_cal_points = np.sum(cal_mask)
        if n_cal_points < 2:
            cal_mask = np.abs(bin_times_rel - phase3_onset_rel) <= 30

    cal_ve = ve_binned[cal_mask]
    cal_times_min = bin_times_min[cal_mask]

    cal_midpoint_min = np.mean(cal_times_min) if len(cal_times_min) > 0 else phase3_onset_rel / 60.0
    cal_ve_mean = np.mean(cal_ve) if len(cal_ve) > 0 else np.mean(ve_binned)

    # Convert percentage drift to absolute drift
    expected_drift = (expected_drift_pct / 100.0) * cal_ve_mean
    alpha = cal_ve_mean - expected_drift * cal_midpoint_min
    ve_expected = alpha + expected_drift * bin_times_min

    # CUSUM parameters
    sigma_ref = (sigma_pct / 100.0) * cal_ve_mean
    k = params.slack_multiplier * sigma_ref
    h = h_mult * sigma_ref

    # CUSUM calculation
    cusum = np.zeros(len(bin_times_rel))
    s = 0.0
    alarm_time = None
    alarm_triggered = False

    for i in range(len(bin_times_rel)):
        if bin_times_rel[i] >= cal_end:
            residual = ve_binned[i] - ve_expected[i]
            s = max(0, s + residual - k)

            if s > h and not alarm_triggered:
                alarm_time = bin_times_rel[i] + breath_times_raw[0]
                alarm_triggered = True

        cusum[i] = s

    peak_cusum = np.max(cusum)
    final_cusum = cusum[-1] if len(cusum) > 0 else 0
    recovered_threshold = h / 2

    # Slope estimation
    analysis_mask = bin_times_rel >= cal_end
    n_analysis_points = np.sum(analysis_mask)
    interval_end_rel = interval.end_time - breath_times_raw[0]

    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]
        slope, intercept = fit_single_slope(analysis_times_min, analysis_ve)
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
    else:
        slope = 0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])

    # Second hinge detection
    post_phase3_mask = bin_times_rel >= phase3_onset_rel
    post_phase3_times = bin_times_rel[post_phase3_mask]
    post_phase3_ve = ve_binned[post_phase3_mask]

    hinge2_time_rel = None
    slope1_pct = None
    slope2_pct = None
    split_slope_ratio = None
    hinge2_detected = False
    segment2_times_rel = None
    segment2_ve = None
    segment3_times_rel = None
    segment3_ve = None

    if len(post_phase3_times) >= 4:
        tau2, h2_b0, h2_b1, h2_b2, _, hinge2_detected = fit_second_hinge(
            post_phase3_times, post_phase3_ve, phase3_onset_rel, interval_end_rel
        )

        hinge2_time_rel = tau2

        slope1_abs = h2_b1 * 60.0
        slope2_abs = (h2_b1 + h2_b2) * 60.0

        if cal_ve_mean > 0:
            slope1_pct = (slope1_abs / cal_ve_mean) * 100.0
            slope2_pct = (slope2_abs / cal_ve_mean) * 100.0
        else:
            slope1_pct = 0.0
            slope2_pct = 0.0

        if slope1_pct <= 0:
            slope1_for_ratio = 0.001
        else:
            slope1_for_ratio = slope1_pct

        split_slope_ratio = slope2_pct / slope1_for_ratio

        segment2_times_rel = np.array([phase3_onset_rel, hinge2_time_rel])
        segment2_ve = h2_b0 + h2_b1 * segment2_times_rel + h2_b2 * np.maximum(0, segment2_times_rel - hinge2_time_rel)

        segment3_times_rel = np.array([hinge2_time_rel, interval_end_rel])
        segment3_ve = h2_b0 + h2_b1 * segment3_times_rel + h2_b2 * np.maximum(0, segment3_times_rel - hinge2_time_rel)

        segment1_ve[-1] = segment2_ve[0]

    # Last 60s average
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0
    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start

    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    # Classification
    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)
    overall_slope_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0
    split_ratio = split_slope_ratio if split_slope_ratio is not None else 1.0

    low_threshold = expected_drift_pct
    high_threshold = max_drift_threshold
    split_ratio_threshold = 1.2

    if not cusum_alarm or cusum_recovered:
        if overall_slope_pct < low_threshold:
            status = IntervalStatus.BELOW_THRESHOLD
        elif overall_slope_pct < high_threshold:
            if split_ratio < split_ratio_threshold:
                status = IntervalStatus.BELOW_THRESHOLD
            else:
                status = IntervalStatus.BORDERLINE
        else:
            status = IntervalStatus.BORDERLINE
    else:
        if overall_slope_pct < low_threshold:
            status = IntervalStatus.BORDERLINE
        elif overall_slope_pct < high_threshold:
            if split_ratio < split_ratio_threshold:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
        else:
            status = IntervalStatus.ABOVE_THRESHOLD

    # Convert to absolute times
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])
    abs_segment1_times = segment1_times_rel + breath_times_raw[0]
    abs_segment2_times = segment2_times_rel + breath_times_raw[0] if segment2_times_rel is not None else None
    abs_segment3_times = segment3_times_rel + breath_times_raw[0] if segment3_times_rel is not None else None

    ve_drift_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0
    initial_ve = cal_ve_mean

    post_cal_mask = bin_times_rel >= cal_end
    if np.sum(post_cal_mask) > 0:
        avg_ve = np.mean(ve_binned[post_cal_mask])
    else:
        avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    terminal_ve = last_60s_avg_ve

    # Extend to interval end
    if len(abs_bin_times) > 0 and abs_bin_times[-1] < interval.end_time:
        abs_bin_times = np.append(abs_bin_times, interval.end_time)
        cusum = np.append(cusum, cusum[-1])
        ve_binned = np.append(ve_binned, ve_binned[-1])
        ve_expected = np.append(ve_expected, ve_expected[-1])

    # Build ChartData
    chart_data = ChartData(
        time_values=abs_bin_times.tolist(),
        ve_binned=ve_binned.tolist(),
        cusum_values=cusum.tolist(),
        expected_ve=ve_expected.tolist(),
        segment1_times=abs_segment1_times.tolist() if abs_segment1_times is not None else None,
        segment1_ve=segment1_ve.tolist() if segment1_ve is not None else None,
        segment2_times=abs_segment2_times.tolist() if abs_segment2_times is not None else None,
        segment2_ve=segment2_ve.tolist() if segment2_ve is not None else None,
        segment3_times=abs_segment3_times.tolist() if abs_segment3_times is not None else None,
        segment3_ve=segment3_ve.tolist() if segment3_ve is not None else None,
        slope_line_times=abs_slope_line_times.tolist() if len(abs_slope_line_times) > 0 else [],
        slope_line_ve=slope_line_ve.tolist() if len(slope_line_ve) > 0 else []
    )

    return IntervalResult(
        interval_num=interval.interval_num,
        start_time=interval.start_time,
        end_time=interval.end_time,
        status=status,
        baseline_ve=cal_ve_mean,
        avg_ve=avg_ve,
        peak_ve=float(np.max(ve_binned)) if len(ve_binned) > 0 else 0,
        initial_ve=initial_ve,
        terminal_ve=terminal_ve,
        last_60s_avg_ve=last_60s_avg_ve,
        ve_drift_rate=slope,
        ve_drift_pct=ve_drift_pct,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        cusum_threshold=h,
        alarm_time=alarm_time,
        cusum_recovered=cusum_recovered,
        is_ceiling_based=False,
        is_segmented=True,
        phase3_onset_rel=phase3_onset_rel,
        hinge2_time_rel=hinge2_time_rel,
        slope1_pct=slope1_pct,
        slope2_pct=slope2_pct,
        split_slope_ratio=split_slope_ratio,
        hinge2_detected=hinge2_detected,
        speed=speed,
        chart_data=chart_data,
        breath_times=breath_times_raw.tolist(),
        ve_median=ve_median.tolist()
    )


def analyze_interval_ceiling(
    breath_df: pd.DataFrame,
    interval: Interval,
    params: AnalysisParams,
    run_type: RunType,
    speed: Optional[float] = None
) -> IntervalResult:
    """
    Perform ceiling-based CUSUM analysis on a single interval.

    Uses user-provided VT1/VT2 VE ceiling as the threshold.
    No blanking/calibration period - starts accumulating after brief warm-up.
    Detects when VE exceeds the ceiling (absolute position, not drift).

    Best for short intervals (<6 min) where drift analysis is impractical.

    Args:
        breath_df: DataFrame with breath-by-breath data
        interval: Interval to analyze
        params: Analysis parameters
        run_type: Type of run (VT1 or VT2)
        speed: Optional speed for this interval

    Returns:
        IntervalResult with analysis results
    """
    idx_start = interval.start_idx
    idx_end = interval.end_idx

    if idx_end <= idx_start:
        return _create_default_result(interval, is_ceiling_based=True, is_segmented=False, speed=speed)

    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end]
    breath_times_raw = breath_df['breath_time'].values[idx_start:idx_end]

    rel_breath_times = breath_times_raw - breath_times_raw[0]

    ve_median, bin_times_rel, ve_binned = apply_hybrid_filtering(ve_raw, rel_breath_times, params)

    bin_times_min = bin_times_rel / 60.0

    # Get ceiling
    if run_type == RunType.VT1_STEADY:
        ceiling_ve = params.vt1_ve_ceiling
        h_mult = params.h_multiplier_vt1
        sigma_pct = params.sigma_pct_vt1
    else:
        ceiling_ve = params.vt2_ve_ceiling
        h_mult = params.h_multiplier_vt2
        sigma_pct = params.sigma_pct_vt2

    sigma_ref = (sigma_pct / 100.0) * ceiling_ve
    k = params.slack_multiplier * sigma_ref
    h = h_mult * sigma_ref

    ve_expected = np.full(len(bin_times_rel), ceiling_ve)

    # CUSUM calculation
    cusum = np.zeros(len(bin_times_rel))
    s = 0.0
    alarm_time = None
    alarm_triggered = False

    for i in range(len(bin_times_rel)):
        if bin_times_rel[i] >= params.ceiling_warmup_sec:
            residual = ve_binned[i] - ceiling_ve
            s = max(0, s + residual - k)

            if s > h and not alarm_triggered:
                alarm_time = bin_times_rel[i] + breath_times_raw[0]
                alarm_triggered = True

        cusum[i] = s

    peak_cusum = np.max(cusum)
    final_cusum = cusum[-1] if len(cusum) > 0 else 0
    recovered_threshold = h / 2

    # Slope estimation
    analysis_mask = bin_times_rel >= params.ceiling_warmup_sec
    n_analysis_points = np.sum(analysis_mask)

    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]
        slope, intercept = fit_single_slope(analysis_times_min, analysis_ve)
        interval_end_rel = interval.end_time - breath_times_raw[0]
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
    else:
        slope = 0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])

    # Last 60s average
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0
    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start

    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    # Classification
    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)

    if not cusum_alarm or cusum_recovered:
        status = IntervalStatus.BELOW_THRESHOLD
    else:
        status = IntervalStatus.ABOVE_THRESHOLD

    # Convert to absolute times
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])

    ve_drift_pct = (slope / ceiling_ve * 100.0) if ceiling_ve > 0 else 0.0
    initial_ve = ceiling_ve
    avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0
    terminal_ve = last_60s_avg_ve

    # Extend to interval end
    if len(abs_bin_times) > 0 and abs_bin_times[-1] < interval.end_time:
        abs_bin_times = np.append(abs_bin_times, interval.end_time)
        cusum = np.append(cusum, cusum[-1])
        ve_binned = np.append(ve_binned, ve_binned[-1])
        ve_expected = np.append(ve_expected, ceiling_ve)

    chart_data = ChartData(
        time_values=abs_bin_times.tolist(),
        ve_binned=ve_binned.tolist(),
        cusum_values=cusum.tolist(),
        expected_ve=ve_expected.tolist(),
        slope_line_times=abs_slope_line_times.tolist() if len(abs_slope_line_times) > 0 else [],
        slope_line_ve=slope_line_ve.tolist() if len(slope_line_ve) > 0 else []
    )

    return IntervalResult(
        interval_num=interval.interval_num,
        start_time=interval.start_time,
        end_time=interval.end_time,
        status=status,
        baseline_ve=ceiling_ve,
        avg_ve=avg_ve,
        peak_ve=float(np.max(ve_binned)) if len(ve_binned) > 0 else 0,
        initial_ve=initial_ve,
        terminal_ve=terminal_ve,
        last_60s_avg_ve=last_60s_avg_ve,
        ve_drift_rate=slope,
        ve_drift_pct=ve_drift_pct,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        cusum_threshold=h,
        alarm_time=alarm_time,
        cusum_recovered=cusum_recovered,
        is_ceiling_based=True,
        is_segmented=False,
        phase3_onset_rel=None,
        hinge2_time_rel=None,
        slope1_pct=None,
        slope2_pct=None,
        split_slope_ratio=None,
        hinge2_detected=False,
        speed=speed,
        chart_data=chart_data,
        breath_times=breath_times_raw.tolist(),
        ve_median=ve_median.tolist()
    )
