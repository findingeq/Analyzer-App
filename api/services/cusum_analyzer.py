"""
CUSUM Analysis Service for VT Threshold Analyzer

Implements two CUSUM analysis approaches:
1. Segmented analysis - For intervals >= 6 min, uses drift detection
2. Ceiling-based analysis - For short intervals, uses absolute VE threshold
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..models.enums import RunType, IntervalStatus
from ..models.params import AnalysisParams
from ..models.schemas import Interval, IntervalResult, ChartData

from .signal_filter import apply_hybrid_filtering
from .regression import (
    fit_single_slope,
    fit_robust_hinge,
    fit_second_hinge,
    # TESTING - Remove after slope model selection
    fit_second_hinge_constrained,
    fit_quadratic_slope,
)


def calculate_madsd_sigma(ve_binned: np.ndarray) -> float:
    """
    Calculate sigma using MAD of Successive Differences (MADSD).

    This method is robust to:
    - Linear drift (differencing removes trends)
    - Outliers (MAD has 50% breakdown point)

    Formula:
        diffs = ve[i+1] - ve[i]
        sigma = MAD(diffs) * 1.4826 / sqrt(2)

    The 1.4826 factor scales MAD to sigma for normal distributions.
    The sqrt(2) accounts for variance of differences being 2x variance of original.

    Args:
        ve_binned: Array of filtered/binned VE values

    Returns:
        Estimated sigma in same units as ve_binned (L/min)
    """
    if len(ve_binned) < 3:
        return 0.0

    # Calculate successive differences
    diffs = np.diff(ve_binned)

    if len(diffs) == 0:
        return 0.0

    # MAD of differences
    median_diff = np.median(diffs)
    absolute_deviations = np.abs(diffs - median_diff)
    mad = np.median(absolute_deviations)

    # Scale to sigma estimate
    # 1.4826 for normal distribution, /sqrt(2) for differenced data
    sigma = mad * 1.4826 / np.sqrt(2)

    return float(sigma)


def calculate_observed_sigma_pct(
    ve_binned: np.ndarray,
    baseline_ve: float,
    analysis_start_idx: int = 0
) -> Optional[float]:
    """
    Calculate observed sigma as percentage of baseline using MADSD.

    Args:
        ve_binned: Array of filtered/binned VE values
        baseline_ve: Baseline VE from calibration period (L/min)
        analysis_start_idx: Index to start analysis (skip calibration period)

    Returns:
        Observed sigma as percentage of baseline, or None if calculation fails
    """
    if baseline_ve <= 0:
        return None

    # Use post-calibration data only
    analysis_ve = ve_binned[analysis_start_idx:]

    if len(analysis_ve) < 3:
        return None

    sigma_abs = calculate_madsd_sigma(analysis_ve)

    # Convert to percentage of baseline
    sigma_pct = (sigma_abs / baseline_ve) * 100.0

    return float(sigma_pct)


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
        last_30s_avg_ve=0,
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
        speed=speed,
        observed_sigma_pct=None,
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

    # Phase III onset detection - all domains use same robust hinge model
    # Same window (90s-180s) and calibration period for Moderate, Heavy, and Severe
    tau, b0, b1, b2, _, detection_succeeded = fit_robust_hinge(bin_times_rel, ve_binned, params)
    phase3_onset_rel = tau
    phase3_onset_time = tau + breath_times_raw[0]

    # Generate segment 1 line - anchor to actual data at start
    seg1_t_start = bin_times_rel[0]
    seg1_t_end = phase3_onset_rel
    segment1_times_rel = np.array([seg1_t_start, seg1_t_end])

    # Use actual VE at start instead of model prediction
    start_ve = ve_binned[0]
    model_ve_at_start = b0 + b1 * seg1_t_start
    model_ve_at_end = b0 + b1 * seg1_t_end

    # Shift the line to pass through the actual start point
    offset = start_ve - model_ve_at_start
    segment1_ve = np.array([start_ve, model_ve_at_end + offset])

    # Set calibration window - all domains use same 60s calibration period
    cal_start = phase3_onset_rel
    cal_end = phase3_onset_rel + params.vt2_calibration_duration

    # Determine domain-specific parameters
    # SEVERE uses same parameters as VT2
    if run_type == RunType.MODERATE:
        h_mult = params.h_multiplier_vt1
        expected_drift_pct = params.expected_drift_pct_vt1
        sigma_pct = params.sigma_pct_vt1
        max_drift_threshold = params.max_drift_pct_vt1
    else:  # HEAVY or SEVERE
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
        slope, intercept = fit_single_slope(analysis_times_min, analysis_ve, params.huber_delta)
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
    else:
        slope = 0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])

    # Single slope line segment initialization (segment3 no longer used)
    segment2_times_rel = None
    segment2_ve = None
    segment3_times_rel = None
    segment3_ve = None

    # Single slope model for all run types (from Phase III onset to end)
    if n_analysis_points >= 3 and len(slope_line_times_rel) > 0:
        phase3_idx = np.argmin(np.abs(bin_times_rel - phase3_onset_rel))
        anchor_ve = ve_binned[phase3_idx]

        # Create segment 2 from phase3 onset to interval end using overall slope
        segment2_times_rel = np.array([phase3_onset_rel, interval_end_rel])
        slope_line_intercept = slope_line_ve[0] - slope * (slope_line_times_rel[0] / 60.0) if len(slope_line_ve) > 0 else cal_ve_mean
        segment2_ve = slope_line_intercept + slope * (segment2_times_rel / 60.0)

        # Adjust to anchor at phase3_onset
        ve_offset = anchor_ve - segment2_ve[0]
        segment2_ve = segment2_ve + ve_offset

        # Make segment1 end at the anchor point
        segment1_ve[-1] = anchor_ve

    # Last 60s and 30s averages
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0

    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start
    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    last_30s_start = max(0, interval_duration - 30)
    last_30s_mask = bin_times_rel >= last_30s_start
    if np.sum(last_30s_mask) > 0:
        last_30s_avg_ve = np.mean(ve_binned[last_30s_mask])
    else:
        last_30s_avg_ve = last_60s_avg_ve  # Fallback to 60s if not enough data

    # Classification
    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)
    overall_slope_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0
    sustained_alarm = cusum_alarm and not cusum_recovered

    if run_type == RunType.MODERATE:
        # Moderate domain classification (thresholds from cloud calibration)
        # low_threshold = expected_drift_pct (0.3%/min default)
        # high_threshold = max_drift_threshold (1.0%/min default for Moderate)
        low_threshold = expected_drift_pct
        high_threshold = max_drift_threshold

        if not sustained_alarm:
            # No cusum OR cusum recovered
            if overall_slope_pct < low_threshold:
                status = IntervalStatus.BELOW_THRESHOLD
            elif overall_slope_pct < high_threshold:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
        else:
            # Cusum triggered and NOT recovered
            if overall_slope_pct < low_threshold:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
    else:
        # Heavy/Severe domain classification (thresholds from cloud calibration)
        # low_threshold = expected_drift_pct (1.0%/min default)
        # high_threshold = max_drift_threshold (3.0%/min default)
        low_threshold = expected_drift_pct
        high_threshold = max_drift_threshold

        if not sustained_alarm:
            # No cusum OR cusum recovered
            if overall_slope_pct < low_threshold:
                status = IntervalStatus.BELOW_THRESHOLD
            elif overall_slope_pct < high_threshold:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
        else:
            # Cusum triggered and NOT recovered
            if overall_slope_pct < low_threshold:
                status = IntervalStatus.BORDERLINE
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

    # Calculate observed sigma using MADSD on post-calibration data
    post_cal_start_idx = np.argmax(post_cal_mask) if np.any(post_cal_mask) else 0
    observed_sigma_pct = calculate_observed_sigma_pct(ve_binned, cal_ve_mean, post_cal_start_idx)

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
        last_30s_avg_ve=last_30s_avg_ve,
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
        speed=speed,
        observed_sigma_pct=observed_sigma_pct,
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
    # SEVERE uses same parameters as VT2
    if run_type == RunType.MODERATE:
        ceiling_ve = params.vt1_ve_ceiling
        h_mult = params.h_multiplier_vt1
        sigma_pct = params.sigma_pct_vt1
    else:  # HEAVY or SEVERE
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

    # LOESS smoothing for ceiling-based analysis (provides curved trend line)
    analysis_mask = bin_times_rel >= params.ceiling_warmup_sec
    n_analysis_points = np.sum(analysis_mask)

    if n_analysis_points >= 4:  # Need enough points for LOESS
        analysis_times = bin_times_rel[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]

        # Apply LOESS smoothing with configurable frac for visual trend line
        loess_result = lowess(analysis_ve, analysis_times, frac=params.loess_frac)
        slope_line_times_rel = loess_result[:, 0]
        slope_line_ve = loess_result[:, 1]

        # Calculate slope from linear fit for drift metrics
        analysis_times_min = analysis_times / 60.0
        slope, _ = fit_single_slope(analysis_times_min, analysis_ve, params.huber_delta)
    else:
        slope = 0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])

    # Last 60s and 30s averages
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0

    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start
    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    last_30s_start = max(0, interval_duration - 30)
    last_30s_mask = bin_times_rel >= last_30s_start
    if np.sum(last_30s_mask) > 0:
        last_30s_avg_ve = np.mean(ve_binned[last_30s_mask])
    else:
        last_30s_avg_ve = last_60s_avg_ve  # Fallback to 60s if not enough data

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

    # Calculate observed sigma using MADSD on post-warmup data
    warmup_mask = bin_times_rel >= params.ceiling_warmup_sec
    warmup_start_idx = np.argmax(warmup_mask) if np.any(warmup_mask) else 0
    observed_sigma_pct = calculate_observed_sigma_pct(ve_binned, ceiling_ve, warmup_start_idx)

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
        last_30s_avg_ve=last_30s_avg_ve,
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
        speed=speed,
        observed_sigma_pct=observed_sigma_pct,
        chart_data=chart_data,
        breath_times=breath_times_raw.tolist(),
        ve_median=ve_median.tolist()
    )
