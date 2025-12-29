"""
VT Threshold Analyzer - Desktop Application
Analyzes Tymewear VitalPro respiratory data to assess VT1/VT2 compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import theilslopes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

class RunType(Enum):
    VT1_STEADY = "VT1 (Steady State)"
    VT2_INTERVAL = "VT2 (Intervals)"

class IntervalStatus(Enum):
    BELOW_THRESHOLD = "✅ Below Threshold"
    BORDERLINE = "⚠️ Borderline"
    ABOVE_THRESHOLD = "❌ Above Threshold"

@dataclass
class AnalysisParams:
    # Phase II typically completes by ~3 minutes; slow component emerges after
    blanking_period: float = 150.0  # seconds - start of calibration lookback window
    calibration_end: float = 180.0  # seconds - end of calibration window (minute 3)
    h_multiplier_vt1: float = 5.0
    h_multiplier_vt2: float = 5.0
    slack_multiplier: float = 0.5  # Hidden - not exposed in UI
    # Expected drift as percentage of baseline VE per minute
    expected_drift_pct_vt1: float = 0.25  # % per minute (minimal in moderate domain)
    expected_drift_pct_vt2: float = 5.0   # % per minute (slow component in heavy domain)
    # Sigma as percentage of baseline VE
    sigma_pct_vt1: float = 10.0  # % of cal_ve_mean
    sigma_pct_vt2: float = 5.0   # % of cal_ve_mean
    # Filtering parameters
    median_window: int = 5  # breaths
    bin_size: float = 5.0   # seconds

@dataclass
class Interval:
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    interval_num: int

@dataclass
class IntervalResult:
    interval: Interval
    status: IntervalStatus
    ve_drift_rate: float  # Theil-Sen slope
    slope_pvalue: float   # P-value testing if slope exceeds expected
    peak_ve: float
    peak_cusum: float
    final_cusum: float
    alarm_time: Optional[float]
    cusum_values: np.ndarray
    time_values: np.ndarray      # Bin timestamps (for CUSUM/binned line)
    ve_binned: np.ndarray        # Binned VE values
    ve_median: np.ndarray        # Median-filtered breath values (for dots)
    breath_times: np.ndarray     # Breath timestamps (for dots)
    expected_ve: np.ndarray
    # Theil-Sen slope line data
    slope_line_times: np.ndarray
    slope_line_ve: np.ndarray
    # Last 60s average for cumulative drift
    last_60s_avg_ve: float

@dataclass
class CumulativeDriftResult:
    """Result of cumulative drift analysis across all intervals."""
    slope_abs: float          # L/min per minute
    slope_pct: float          # % of baseline per minute
    baseline_ve: float        # Interval 1 last-60s average
    pvalue: float             # Statistical significance
    # Data for visualization
    interval_end_times: np.ndarray   # X-values (elapsed time at end of each interval)
    interval_avg_ve: np.ndarray      # Y-values (last-60s average VE)
    line_times: np.ndarray           # X-values for regression line
    line_ve: np.ndarray              # Y-values for regression line

# ============================================================================
# DATA PARSING
# ============================================================================

def parse_vitalpro_csv(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """Parse VitalPro CSV and extract breath-by-breath data with metadata."""
    # Read raw content
    content = uploaded_file.getvalue().decode('utf-8')
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    
    # Find header row (contains 'Time' and 'Breath by breath time')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Time' in line and 'Breath by breath time' in line:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find data header row")
    
    # Extract metadata from rows before header
    metadata = {}
    for i in range(header_idx):
        parts = lines[i].split(',')
        if len(parts) >= 2 and parts[0].strip():
            metadata[parts[0].strip()] = parts[1].strip()
    
    # Parse data starting from header
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, skiprows=header_idx, skip_blank_lines=True)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract relevant columns
    breath_cols = {
        'breath_time': 'Breath by breath time',
        'br_raw': 'BR breath by breath',
        'vt_raw': 'VT breath by breath',
        've_raw': 'VE breath by breath',
        'power': 'Power',
        'time': 'Time'
    }
    
    # Create cleaned dataframe with breath-by-breath data
    result = pd.DataFrame()
    
    for key, col in breath_cols.items():
        if col in df.columns:
            result[key] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where breath_time is NaN (no breath recorded)
    result = result.dropna(subset=['breath_time']).reset_index(drop=True)
    
    # Also keep original time-aligned power data for interval detection
    power_df = pd.DataFrame({
        'time': pd.to_numeric(df['Time'], errors='coerce') if 'Time' in df.columns else None,
        'power': pd.to_numeric(df['Power'], errors='coerce') if 'Power' in df.columns else None
    }).dropna()
    
    return result, metadata, power_df

# ============================================================================
# SIGNAL FILTERING (Hybrid Three-Stage)
# ============================================================================

def apply_breath_rejection(ve_raw: np.ndarray, breath_times: np.ndarray,
                           window: int = 9, threshold_mad: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 0: Reject outlier breaths using rolling MAD-based detection.

    For each breath, calculate its deviation from the local median using
    Median Absolute Deviation (MAD). Breaths that deviate by more than
    threshold_mad * MAD are marked as outliers and removed.

    This is more robust than non-overlapping windows because it considers
    each breath's local context independently, avoiding boundary effects.

    Args:
        ve_raw: Raw VE values (breath-by-breath)
        breath_times: Timestamps for each breath
        window: Size of the rolling window for local median/MAD (default 5)
        threshold_mad: Number of MADs to consider a breath an outlier (default 2.5)

    Returns:
        Tuple of (filtered VE values, filtered breath times)
    """
    n = len(ve_raw)
    if n == 0:
        return ve_raw.copy(), breath_times.copy()

    if n <= window:
        # Not enough data for windowed analysis, keep all
        return ve_raw.copy(), breath_times.copy()

    half_window = window // 2
    outlier_mask = np.zeros(n, dtype=bool)

    # MAD scale factor to make it comparable to standard deviation
    mad_scale = 1.4826

    for i in range(n):
        # Define window boundaries
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        window_data = ve_raw[start:end]

        # Calculate local median and MAD
        local_median = np.median(window_data)
        mad = np.median(np.abs(window_data - local_median))

        # If MAD is 0 (all same values), use a small default
        if mad < 1e-6:
            mad = 1.0

        # Calculate deviation in terms of scaled MAD
        deviation = np.abs(ve_raw[i] - local_median) / (mad * mad_scale)

        # Mark as outlier if deviation exceeds threshold
        if deviation > threshold_mad:
            outlier_mask[i] = True

    # Keep non-outliers
    keep_mask = ~outlier_mask

    return ve_raw[keep_mask], breath_times[keep_mask]


def apply_median_filter(ve_raw: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Stage 1: Apply rolling median filter in breath domain.
    Removes remaining non-physiological spikes after outlier rejection.
    """
    if len(ve_raw) < window:
        return ve_raw.copy()

    ve_median = median_filter(ve_raw, size=window, mode='nearest')
    return ve_median


def apply_time_binning(ve_clean: np.ndarray, breath_times: np.ndarray,
                       bin_size: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Convert irregular breath data to uniform time series via bin averaging.
    Uses bin-start timestamp convention.
    Empty bins are filled via linear interpolation.

    Returns:
        bin_times: Array of bin start timestamps
        bin_values: Array of averaged VE values per bin
    """
    if len(ve_clean) == 0:
        return np.array([]), np.array([])

    t_min = breath_times[0]
    t_max = breath_times[-1]

    # Create bin edges (start times)
    bin_starts = np.arange(t_min, t_max, bin_size)
    n_bins = len(bin_starts)

    if n_bins == 0:
        return np.array([t_min]), np.array([np.mean(ve_clean)])

    bin_values = np.full(n_bins, np.nan)

    # Assign breaths to bins and compute means
    for i in range(n_bins):
        bin_start = bin_starts[i]
        bin_end = bin_start + bin_size

        # Find breaths in this bin: [bin_start, bin_end)
        mask = (breath_times >= bin_start) & (breath_times < bin_end)

        if np.sum(mask) > 0:
            bin_values[i] = np.mean(ve_clean[mask])

    # Linear interpolation for empty bins
    if np.any(np.isnan(bin_values)):
        valid_mask = ~np.isnan(bin_values)
        if np.sum(valid_mask) >= 2:
            bin_values = np.interp(
                bin_starts,
                bin_starts[valid_mask],
                bin_values[valid_mask]
            )
        elif np.sum(valid_mask) == 1:
            # Only one valid bin - fill all with that value
            bin_values[:] = bin_values[valid_mask][0]
        else:
            # No valid bins - use overall mean
            bin_values[:] = np.mean(ve_clean)

    return bin_starts, bin_values


def apply_hybrid_filtering(ve_raw: np.ndarray, breath_times: np.ndarray,
                           params: AnalysisParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply three-stage hybrid filtering:
    Stage 0: Outlier rejection (remove breaths >2.5 MAD from local median)
    Stage 1: Rolling median (breath domain) - removes remaining artifacts
    Stage 2: Time binning (time domain) - standardizes accumulation rate

    Returns:
        ve_median: Median-filtered breath values (for visualization dots)
        breath_times_filtered: Timestamps after outlier rejection (for dots)
        bin_times: Bin start timestamps
        ve_binned: Binned VE values (for CUSUM analysis)
    """
    # Stage 0: Outlier rejection using rolling MAD
    ve_rejected, times_rejected = apply_breath_rejection(ve_raw, breath_times)

    # Stage 1: Median filter on remaining breaths
    ve_median = apply_median_filter(ve_rejected, params.median_window)

    # Stage 2: Time binning
    bin_times, ve_binned = apply_time_binning(ve_median, times_rejected, params.bin_size)

    return ve_median, times_rejected, bin_times, ve_binned

# ============================================================================
# INTERVAL DETECTION
# ============================================================================

def detect_intervals(power_df: pd.DataFrame, breath_df: pd.DataFrame) -> Tuple[RunType, int, float, float]:
    """
    Detect run format and intervals from power data.
    Returns run type, number of intervals, interval duration (min), and rest duration (min).

    Key assumptions:
    - First interval ALWAYS starts at t=0
    - Intervals are ALWAYS whole minutes (4m, 5m, 8m, etc.)
    - Rest periods are standard durations (0.5m, 1m, 1.5m, etc.)
    - Recording starts when first interval begins
    - Recording ends shortly after last recovery

    Detection approach:
    - Find where steady high power transitions to sustained ramp-down (interval end)
    - Fit detected end times to a regular grid of (interval_duration + rest_duration)
    """
    if power_df.empty or len(power_df) < 10:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    power = power_df['power'].values
    time = power_df['time'].values

    # Skip NaN values
    valid_mask = ~np.isnan(power)
    if np.sum(valid_mask) < 100:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # Check if this is interval training (large power range) or steady state
    valid_power = power[valid_mask]
    p10 = np.nanpercentile(valid_power, 10)
    p90 = np.nanpercentile(valid_power, 90)

    # If work and rest aren't clearly separated, it's steady state
    if p90 < p10 * 3:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # Detect interval end points: where steady high power starts sustained ramp-down
    # Look for points where power drops significantly from recent average and continues dropping
    window_size = 10  # seconds to look back for average

    interval_end_times = []
    last_end_time = -300  # Ensure we don't detect multiple ends for same interval

    for i in range(window_size + 5, len(time) - 4):
        if np.isnan(power[i]):
            continue

        # Calculate average power over previous window
        window_powers = [power[j] for j in range(i - window_size, i) if not np.isnan(power[j])]
        if len(window_powers) < window_size // 2:
            continue
        avg_power = np.mean(window_powers)

        # Check if we're in high power territory and starting to drop
        # Criteria: avg was high (>1000), current is 8%+ below avg, and next few points continue dropping
        if (avg_power > 1000 and
            power[i] < avg_power * 0.92 and
            power[i + 1] < power[i] and
            power[i + 2] < power[i + 1] and
            power[i + 3] < power[i + 2] and
            time[i] - last_end_time > 120):  # At least 2 min since last detected end

            interval_end_times.append(time[i])
            last_end_time = time[i]

    if len(interval_end_times) == 0:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # Convert end times to minutes
    end_times_min = [t / 60.0 for t in interval_end_times]

    # The first interval ends around end_times_min[0], and starts at 0
    # So interval duration is approximately end_times_min[0]
    # Round to nearest whole minute
    first_interval_approx = end_times_min[0]
    interval_duration = round(first_interval_approx)
    if interval_duration < 1:
        interval_duration = 1

    # Calculate the period (interval + rest) from spacing between consecutive ends
    if len(end_times_min) >= 2:
        periods = [end_times_min[i + 1] - end_times_min[i] for i in range(len(end_times_min) - 1)]
        median_period = np.median(periods)

        # Rest duration = period - interval duration
        rest_approx = median_period - interval_duration
        # Round to nearest 0.5 minute
        rest_duration = round(rest_approx * 2) / 2
        if rest_duration < 0.5:
            rest_duration = 0.5
    else:
        rest_duration = 1.0  # Default

    num_intervals = len(interval_end_times)

    return RunType.VT2_INTERVAL, num_intervals, float(interval_duration), float(rest_duration)


def create_intervals_from_params(breath_df: pd.DataFrame, run_type: RunType,
                                  num_intervals: int, interval_duration: float,
                                  recovery_duration: float) -> List[Interval]:
    """Create interval objects based on detected/specified parameters."""
    intervals = []
    breath_times = breath_df['breath_time'].values
    
    if run_type == RunType.VT1_STEADY:
        # Single interval spanning entire run
        total_duration = breath_df['breath_time'].max()
        return [Interval(
            start_time=0,
            end_time=total_duration,
            start_idx=0,
            end_idx=len(breath_df) - 1,
            interval_num=1
        )]
    
    # VT2 intervals - start at t=0
    current_time = 0.0
    for i in range(num_intervals):
        start_time = current_time
        end_time = start_time + interval_duration * 60  # Convert minutes to seconds
        
        breath_start_idx = np.searchsorted(breath_times, start_time)
        breath_end_idx = np.searchsorted(breath_times, end_time)
        
        if breath_start_idx < len(breath_times):
            intervals.append(Interval(
                start_time=start_time,
                end_time=min(end_time, breath_times[-1]),
                start_idx=breath_start_idx,
                end_idx=min(breath_end_idx, len(breath_df) - 1),
                interval_num=i + 1
            ))
        
        current_time = end_time + recovery_duration * 60
    
    return intervals

# ============================================================================
# CUSUM ANALYSIS
# ============================================================================

def compute_theil_sen_pvalue(slope: float, lo_slope: float, hi_slope: float, 
                             expected_drift: float) -> float:
    """
    Derive approximate p-value for testing if slope exceeds expected_drift.
    Based on whether the confidence interval excludes expected_drift.
    """
    if lo_slope > expected_drift:
        # Entire CI above expected_drift -> significant
        # If slope is much larger than CI half-width, more significant
        ci_half_width = (hi_slope - lo_slope) / 2
        if ci_half_width > 0 and (slope - expected_drift) > 2.5 * ci_half_width:
            return 0.01
        return 0.03
    elif hi_slope < expected_drift:
        # Entire CI below expected_drift -> slope significantly LESS than expected
        # This shouldn't trigger "above threshold" classification
        return 0.50
    else:
        # CI spans expected_drift -> not significantly different
        return 0.15


def analyze_interval(breath_df: pd.DataFrame, interval: Interval,
                     params: AnalysisParams, run_type: RunType) -> IntervalResult:
    """
    Perform CUSUM analysis on a single interval using hybrid filtering.
    
    Uses domain-expected drift rate (as % of baseline VE) as the baseline model slope,
    with self-calibrated intercept from the calibration window.
    Classification uses combined CUSUM + Theil-Sen slope logic.
    """
    
    # Extract interval data
    idx_start = interval.start_idx
    idx_end = interval.end_idx
    
    if idx_end <= idx_start:
        # Invalid interval
        return IntervalResult(
            interval=interval,
            status=IntervalStatus.BELOW_THRESHOLD,
            ve_drift_rate=0,
            slope_pvalue=1.0,
            peak_ve=0,
            peak_cusum=0,
            final_cusum=0,
            alarm_time=None,
            cusum_values=np.array([0]),
            time_values=np.array([interval.start_time]),
            ve_binned=np.array([0]),
            ve_median=np.array([0]),
            breath_times=np.array([interval.start_time]),
            expected_ve=np.array([0]),
            slope_line_times=np.array([]),
            slope_line_ve=np.array([]),
            last_60s_avg_ve=0
        )
    
    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end+1]
    breath_times_raw = breath_df['breath_time'].values[idx_start:idx_end+1]

    # Relative time within interval (in seconds)
    rel_breath_times = breath_times_raw - breath_times_raw[0]

    # Apply hybrid filtering (3-stage: outlier rejection, median filter, binning)
    ve_median, breath_times_filtered, bin_times_rel, ve_binned = apply_hybrid_filtering(
        ve_raw, rel_breath_times, params
    )
    
    # Convert bin times to minutes for drift calculations
    bin_times_min = bin_times_rel / 60.0
    
    # Determine domain-specific parameters
    if run_type == RunType.VT1_STEADY:
        h_mult = params.h_multiplier_vt1
        expected_drift_pct = params.expected_drift_pct_vt1  # % per minute
        sigma_pct = params.sigma_pct_vt1  # % of baseline
    else:
        h_mult = params.h_multiplier_vt2
        expected_drift_pct = params.expected_drift_pct_vt2  # % per minute
        sigma_pct = params.sigma_pct_vt2  # % of baseline
    
    # Find calibration window (in bin domain)
    cal_mask = (bin_times_rel >= params.blanking_period) & (bin_times_rel <= params.calibration_end)
    n_cal_points = np.sum(cal_mask)
    
    if n_cal_points < 3:
        # Extend calibration window if needed
        cal_mask = bin_times_rel >= params.blanking_period
        n_cal_points = np.sum(cal_mask)
        
        if n_cal_points < 2:
            cal_mask = np.arange(len(bin_times_rel)) >= len(bin_times_rel) // 3
    
    cal_ve = ve_binned[cal_mask]
    cal_times_min = bin_times_min[cal_mask]
    
    # Baseline model: VE_expected = alpha + expected_drift * t
    # Self-calibrate alpha (intercept) based on calibration window mean
    cal_midpoint_min = np.mean(cal_times_min) if len(cal_times_min) > 0 else 0
    cal_ve_mean = np.mean(cal_ve) if len(cal_ve) > 0 else np.mean(ve_binned)
    
    # Convert percentage drift to absolute drift (L/min per minute)
    expected_drift = (expected_drift_pct / 100.0) * cal_ve_mean
    
    # alpha is the VE at t=0 that would give cal_ve_mean at cal_midpoint given expected_drift
    alpha = cal_ve_mean - expected_drift * cal_midpoint_min
    
    # Calculate expected VE for all bin timepoints
    ve_expected = alpha + expected_drift * bin_times_min
    
    # Fixed sigma as percentage of baseline VE (not calculated from residuals)
    sigma_ref = (sigma_pct / 100.0) * cal_ve_mean
    
    # CUSUM parameters
    k = params.slack_multiplier * sigma_ref  # Slack
    h = h_mult * sigma_ref  # Threshold
    
    # CUSUM calculation - detect VE rising faster than expected
    # Iterate over bins (uniform time steps)
    cusum = np.zeros(len(bin_times_rel))
    s = 0.0
    alarm_time = None
    alarm_triggered = False
    
    for i in range(len(bin_times_rel)):
        if bin_times_rel[i] >= params.calibration_end:
            # Residual: actual - expected (positive means VE higher than expected)
            residual = ve_binned[i] - ve_expected[i]
            # One-sided upper CUSUM (detects increases beyond expected drift)
            s = max(0, s + residual - k)
            
            if s > h and not alarm_triggered:
                alarm_time = bin_times_rel[i] + breath_times_raw[0]  # Convert to absolute time
                alarm_triggered = True
        
        cusum[i] = s
    
    # CUSUM metrics
    peak_cusum = np.max(cusum)
    final_cusum = cusum[-1] if len(cusum) > 0 else 0
    recovered_threshold = h / 2
    
    # Theil-Sen slope estimation on post-calibration data
    analysis_mask = bin_times_rel >= params.calibration_end
    n_analysis_points = np.sum(analysis_mask)
    
    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]
        
        # Theil-Sen robust slope estimation
        slope, intercept, lo_slope, hi_slope = theilslopes(analysis_ve, analysis_times_min)
        
        # Calculate p-value for slope exceeding expected_drift
        slope_pvalue = compute_theil_sen_pvalue(slope, lo_slope, hi_slope, expected_drift)
        
        # Slope line for visualization (post-calibration only)
        slope_line_times_rel = bin_times_rel[analysis_mask]
        slope_line_ve = intercept + slope * analysis_times_min
    else:
        slope = 0
        slope_pvalue = 1.0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])
    
    # Calculate last 60 seconds average VE for cumulative drift
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0
    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start
    
    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0
    
    # Classification using combined CUSUM + Theil-Sen logic
    cusum_alarm = peak_cusum >= h
    cusum_recovered = final_cusum <= recovered_threshold
    slope_at_or_below = (slope <= expected_drift)
    
    if not cusum_alarm:
        # No CUSUM alarm
        if slope_at_or_below:
            status = IntervalStatus.BELOW_THRESHOLD
        else:
            status = IntervalStatus.BORDERLINE
    else:
        # CUSUM alarm triggered
        if cusum_recovered:
            if slope_at_or_below:
                status = IntervalStatus.BELOW_THRESHOLD  # CUSUM false alarm
            else:
                status = IntervalStatus.BORDERLINE
        else:
            status = IntervalStatus.ABOVE_THRESHOLD
    
    # Convert times to absolute for output
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])
    
    return IntervalResult(
        interval=interval,
        status=status,
        ve_drift_rate=slope,
        slope_pvalue=slope_pvalue,
        peak_ve=np.max(ve_binned) if len(ve_binned) > 0 else 0,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        alarm_time=alarm_time,
        cusum_values=cusum,
        time_values=abs_bin_times,
        ve_binned=ve_binned,
        ve_median=ve_median,
        breath_times=breath_times_filtered + breath_times_raw[0],  # Absolute times for filtered breath dots
        expected_ve=ve_expected,
        slope_line_times=abs_slope_line_times,
        slope_line_ve=slope_line_ve,
        last_60s_avg_ve=last_60s_avg_ve
    )


def compute_cumulative_drift(results: List[IntervalResult]) -> Optional[CumulativeDriftResult]:
    """
    Compute cumulative drift across all intervals using Theil-Sen regression.
    Only applicable for VT2 interval runs with 2+ intervals.
    
    Uses the last 60s average VE from each interval as data points.
    X-axis is elapsed time at end of each interval (including rest periods).
    """
    if len(results) < 2:
        return None
    
    # Extract data points: (end_time, last_60s_avg_ve) for each interval
    interval_end_times = np.array([r.interval.end_time / 60.0 for r in results])  # Convert to minutes
    interval_avg_ve = np.array([r.last_60s_avg_ve for r in results])
    
    # Baseline is interval 1's last 60s average
    baseline_ve = interval_avg_ve[0]
    
    if baseline_ve <= 0:
        return None
    
    # Theil-Sen regression on all intervals (including interval 1)
    if len(interval_end_times) >= 2:
        slope, intercept, lo_slope, hi_slope = theilslopes(interval_avg_ve, interval_end_times)
        
        # P-value: test if slope is significantly different from 0
        if lo_slope > 0:
            pvalue = 0.03
        elif hi_slope < 0:
            pvalue = 0.03
        else:
            pvalue = 0.15
        
        # Convert slope to percentage of baseline per minute
        slope_pct = (slope / baseline_ve) * 100.0
        
        # Regression line from interval 1 to final interval
        line_times = np.array([interval_end_times[0], interval_end_times[-1]])
        line_ve = intercept + slope * line_times
        
        return CumulativeDriftResult(
            slope_abs=slope,
            slope_pct=slope_pct,
            baseline_ve=baseline_ve,
            pvalue=pvalue,
            interval_end_times=interval_end_times,
            interval_avg_ve=interval_avg_ve,
            line_times=line_times * 60.0,  # Convert back to seconds for plotting
            line_ve=line_ve
        )
    
    return None

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_main_chart(breath_df: pd.DataFrame, results: List[IntervalResult],
                      intervals: List[Interval], params: AnalysisParams,
                      selected_interval: Optional[int] = None,
                      cumulative_drift: Optional[CumulativeDriftResult] = None,
                      run_type: RunType = RunType.VT1_STEADY) -> go.Figure:
    """Create the main visualization chart."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=('Filtered Minute Ventilation (VE)', 'CUSUM Score')
    )
    
    # Determine x-axis range
    if selected_interval is not None and selected_interval < len(results):
        result = results[selected_interval]
        x_min = result.interval.start_time - 10
        x_max = result.interval.end_time + 10
    else:
        x_min = breath_df['breath_time'].min()
        x_max = breath_df['breath_time'].max()
    
    # Color palette for intervals
    interval_colors = [
        'rgba(100, 149, 237, 0.15)',  # Cornflower blue
        'rgba(144, 238, 144, 0.15)',  # Light green
        'rgba(255, 182, 193, 0.15)',  # Light pink
        'rgba(255, 218, 185, 0.15)',  # Peach
        'rgba(221, 160, 221, 0.15)',  # Plum
        'rgba(176, 224, 230, 0.15)',  # Powder blue
    ]
    
    # Add interval shading
    for i, interval in enumerate(intervals):
        color = interval_colors[i % len(interval_colors)]
        fig.add_vrect(
            x0=interval.start_time,
            x1=interval.end_time,
            fillcolor=color,
            layer="below",
            line_width=0,
            row="all", col=1
        )
        # Add interval label
        fig.add_annotation(
            x=(interval.start_time + interval.end_time) / 2,
            y=1.02,
            yref="paper",
            text=f"Int {interval.interval_num}",
            showarrow=False,
            font=dict(size=10, color="#666")
        )
    
    # Plot each interval's results
    for result in results:
        # Calculate relative times within interval for tooltip
        interval_start = result.interval.start_time
        
        # Faint breath-by-breath dots (median-filtered)
        fig.add_trace(
            go.Scatter(
                x=result.breath_times,
                y=result.ve_median,
                mode='markers',
                name='Breaths',
                marker=dict(size=3, color='rgba(46, 134, 171, 0.2)'),
                showlegend=(result.interval.interval_num == 1),
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Create custom hover text for VE binned trace
        ve_hover_texts = []
        for i, (t, ve) in enumerate(zip(result.time_values, result.ve_binned)):
            rel_time = t - interval_start
            mins = int(rel_time // 60)
            secs = int(rel_time % 60)
            # Find corresponding CUSUM value
            cusum_val = result.cusum_values[i] if i < len(result.cusum_values) else 0
            ve_hover_texts.append(
                f"Interval: {result.interval.interval_num}<br>"
                f"Time: {mins}:{secs:02d}<br>"
                f"VE: {ve:.1f} L/min<br>"
                f"CUSUM: {cusum_val:.1f}"
            )
        
        # Bold binned VE line (what CUSUM analyzes) with enhanced tooltips
        fig.add_trace(
            go.Scatter(
                x=result.time_values,
                y=result.ve_binned,
                mode='lines',
                name='VE (5s bins)',
                line=dict(color='#2E86AB', width=2),
                showlegend=(result.interval.interval_num == 1),
                hoverinfo='text',
                hovertext=ve_hover_texts
            ),
            row=1, col=1
        )
        
        # Theil-Sen slope line (solid green, post-calibration only)
        if len(result.slope_line_times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=result.slope_line_times,
                    y=result.slope_line_ve,
                    mode='lines',
                    name='Observed Slope',
                    line=dict(color='#228B22', width=2),  # Forest green
                    showlegend=(result.interval.interval_num == 1)
                ),
                row=1, col=1
            )
        
        # Create custom hover text for CUSUM trace
        cusum_hover_texts = []
        for i, (t, cusum_val) in enumerate(zip(result.time_values, result.cusum_values)):
            rel_time = t - interval_start
            mins = int(rel_time // 60)
            secs = int(rel_time % 60)
            ve_val = result.ve_binned[i] if i < len(result.ve_binned) else 0
            cusum_hover_texts.append(
                f"Interval: {result.interval.interval_num}<br>"
                f"Time: {mins}:{secs:02d}<br>"
                f"VE: {ve_val:.1f} L/min<br>"
                f"CUSUM: {cusum_val:.1f}"
            )
        
        # CUSUM trace with enhanced tooltips
        fig.add_trace(
            go.Scatter(
                x=result.time_values,
                y=result.cusum_values,
                mode='lines',
                name='CUSUM',
                line=dict(color='#E94F37', width=1.5),
                showlegend=(result.interval.interval_num == 1),
                hoverinfo='text',
                hovertext=cusum_hover_texts
            ),
            row=2, col=1
        )
        
        # Add alarm line if triggered
        if result.alarm_time is not None:
            fig.add_vline(
                x=result.alarm_time,
                line=dict(color='red', width=2, dash='dot'),
                row="all", col=1
            )
            fig.add_annotation(
                x=result.alarm_time,
                y=result.peak_cusum,
                text="ALARM",
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                font=dict(color='red', size=10),
                row=2, col=1
            )
    
    # Add cumulative drift line (only in zoomed-out view for VT2)
    if (cumulative_drift is not None and 
        selected_interval is None and 
        run_type == RunType.VT2_INTERVAL):
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_drift.line_times,
                y=cumulative_drift.line_ve,
                mode='lines',
                name='Cumulative Drift',
                line=dict(color='#FF6B35', width=3),  # Orange
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        title=dict(
            text="VT Threshold Analysis",
            font=dict(size=20, color="#1a1a2e")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    fig.update_xaxes(
        title_text="Time (seconds)",
        row=2, col=1,
        range=[x_min, x_max] if selected_interval is not None else None
    )
    fig.update_yaxes(title_text="VE (L/min)", row=1, col=1)
    fig.update_yaxes(title_text="CUSUM Score", row=2, col=1)
    
    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="VT Threshold Analyzer",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        
        .main-header h1 {
            color: #fff;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .main-header p {
            color: #a0a0a0;
            margin: 0.5rem 0 0 0;
            font-size: 0.95rem;
        }
        
        .result-card {
            background: #fff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #2E86AB;
            margin-bottom: 0.8rem;
        }
        
        .status-below {
            color: #28a745;
            font-weight: 600;
        }
        
        .status-above {
            color: #dc3545;
            font-weight: 600;
        }
        
        .status-borderline {
            color: #ffc107;
            font-weight: 600;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.3rem 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.85rem;
        }
        
        .metric-value {
            font-weight: 500;
            color: #1a1a2e;
        }
        
        div[data-testid="stSidebar"] {
            background: #f8f9fa;
        }
        
        .sidebar-section {
            background: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        
        .sidebar-section h3 {
            margin-top: 0;
            color: #1a1a2e;
            font-size: 1rem;
            font-weight: 600;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 0.5rem;
        }
        
        .cumulative-drift-box {
            background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .cumulative-drift-box h4 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .cumulative-drift-box .value {
            font-size: 1.4rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'breath_df' not in st.session_state:
        st.session_state.breath_df = None
    if 'power_df' not in st.session_state:
        st.session_state.power_df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'intervals' not in st.session_state:
        st.session_state.intervals = None
    if 'run_type' not in st.session_state:
        st.session_state.run_type = None
    if 'selected_interval' not in st.session_state:
        st.session_state.selected_interval = None
    if 'detected_params' not in st.session_state:
        st.session_state.detected_params = None
    if 'cumulative_drift' not in st.session_state:
        st.session_state.cumulative_drift = None
    
    # Get default params from dataclass
    default_params = AnalysisParams()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        
        uploaded_file = st.file_uploader(
            "Upload VitalPro CSV",
            type=['csv'],
            help="Upload the CSV file exported from the Tymewear VitalPro"
        )
        
        if uploaded_file is not None:
            try:
                breath_df, metadata, power_df = parse_vitalpro_csv(uploaded_file)
                st.session_state.breath_df = breath_df
                st.session_state.power_df = power_df
                st.success(f"✓ Loaded {len(breath_df)} breaths")
                
                # Auto-detect run parameters
                run_type, num_int, int_dur, rest_dur = detect_intervals(power_df, breath_df)
                st.session_state.detected_params = {
                    'run_type': run_type,
                    'num_intervals': num_int,
                    'interval_duration': int_dur,
                    'recovery_duration': rest_dur
                }
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 🏃 Run Format")
        
        # Get detected params or use defaults
        detected = st.session_state.detected_params or {
            'run_type': RunType.VT1_STEADY,
            'num_intervals': 12,
            'interval_duration': 4.0,
            'recovery_duration': 1.0
        }
        
        # Run type selection
        run_type_options = ["VT1 (Steady State)", "VT2 (Intervals)"]
        default_idx = 0 if detected['run_type'] == RunType.VT1_STEADY else 1
        run_type_option = st.selectbox(
            "Run Type",
            options=run_type_options,
            index=default_idx,
            help="Auto-detected from power data; override if needed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            num_intervals = st.number_input(
                "# Intervals",
                min_value=1, max_value=30, 
                value=detected['num_intervals'],
                disabled=(run_type_option == "VT1 (Steady State)")
            )
            interval_duration = st.number_input(
                "Interval (min)",
                min_value=1.0, max_value=30.0, 
                value=detected['interval_duration'], 
                step=1.0,
                disabled=(run_type_option == "VT1 (Steady State)")
            )
        with col2:
            recovery_duration = st.number_input(
                "Recovery (min)",
                min_value=0.5, max_value=10.0, 
                value=detected['recovery_duration'], 
                step=0.5,
                disabled=(run_type_option == "VT1 (Steady State)")
            )
            run_speed = st.number_input(
                "Speed (mph)",
                min_value=0.0, max_value=15.0, value=7.6, step=0.1,
                help="Optional: for reference only"
            )
        
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Parameters")
        
        params = AnalysisParams()
        
        col1, col2 = st.columns(2)
        with col1:
            params.blanking_period = st.number_input(
                "Blanking (s)",
                min_value=30.0, max_value=180.0, 
                value=default_params.blanking_period, 
                step=10.0,
                help="Kinetic blanking period to ignore on-transient"
            )
        with col2:
            params.calibration_end = st.number_input(
                "Cal. End (s)",
                min_value=60.0, max_value=240.0, 
                value=default_params.calibration_end, 
                step=10.0,
                help="End of calibration window"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            params.h_multiplier_vt1 = st.number_input(
                "H (VT1)",
                min_value=3.0, max_value=10.0, 
                value=default_params.h_multiplier_vt1, 
                step=0.5,
                help="H threshold multiplier for VT1"
            )
        with col2:
            params.h_multiplier_vt2 = st.number_input(
                "H (VT2)",
                min_value=3.0, max_value=12.0, 
                value=default_params.h_multiplier_vt2, 
                step=0.5,
                help="H threshold multiplier for VT2"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            params.sigma_pct_vt1 = st.number_input(
                "Sigma VT1 (%)",
                min_value=1.0, max_value=25.0, 
                value=default_params.sigma_pct_vt1, 
                step=1.0,
                help="Sigma as % of baseline VE for VT1"
            )
        with col2:
            params.sigma_pct_vt2 = st.number_input(
                "Sigma VT2 (%)",
                min_value=1.0, max_value=25.0, 
                value=default_params.sigma_pct_vt2, 
                step=1.0,
                help="Sigma as % of baseline VE for VT2"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            params.expected_drift_pct_vt1 = st.number_input(
                "Drift VT1 (%)",
                min_value=0.0, max_value=5.0, 
                value=default_params.expected_drift_pct_vt1, 
                step=0.25,
                help="Expected VE drift for VT1 (% of baseline per min)"
            )
        with col2:
            params.expected_drift_pct_vt2 = st.number_input(
                "Drift VT2 (%)",
                min_value=0.5, max_value=15.0, 
                value=default_params.expected_drift_pct_vt2, 
                step=0.5,
                help="Expected VE drift for VT2 (% of baseline per min)"
            )
        
        st.markdown("---")
        
        # Analyze button
        analyze_btn = st.button("🔬 Analyze Run", type="primary", use_container_width=True)
        
        if analyze_btn and st.session_state.breath_df is not None:
            breath_df = st.session_state.breath_df
            power_df = st.session_state.power_df
            
            # Determine run type
            if run_type_option == "VT1 (Steady State)":
                run_type = RunType.VT1_STEADY
            else:
                run_type = RunType.VT2_INTERVAL
            
            # Create intervals based on parameters
            intervals = create_intervals_from_params(
                breath_df, run_type,
                num_intervals, interval_duration, recovery_duration
            )
            
            # Run analysis
            results = []
            for interval in intervals:
                result = analyze_interval(breath_df, interval, params, run_type)
                results.append(result)
            
            # Compute cumulative drift for VT2 runs
            if run_type == RunType.VT2_INTERVAL and len(results) >= 2:
                cumulative_drift = compute_cumulative_drift(results)
            else:
                cumulative_drift = None
            
            st.session_state.results = results
            st.session_state.intervals = intervals
            st.session_state.run_type = run_type
            st.session_state.selected_interval = None
            st.session_state.cumulative_drift = cumulative_drift
    
    # Main content area
    st.markdown("""
        <div class="main-header">
            <h1>🫁 VT Threshold Analyzer</h1>
            <p>Analyze Tymewear VitalPro respiratory data for VT1/VT2 compliance</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results is not None:
        results = st.session_state.results
        intervals = st.session_state.intervals
        run_type = st.session_state.run_type
        breath_df = st.session_state.breath_df
        cumulative_drift = st.session_state.cumulative_drift
        
        # Summary row
        col1, col2, col3, col4 = st.columns(4)
        
        below_count = sum(1 for r in results if r.status == IntervalStatus.BELOW_THRESHOLD)
        borderline_count = sum(1 for r in results if r.status == IntervalStatus.BORDERLINE)
        above_count = sum(1 for r in results if r.status == IntervalStatus.ABOVE_THRESHOLD)
        
        with col1:
            st.metric("Run Type", run_type.value)
        with col2:
            st.metric("Intervals", len(intervals))
        with col3:
            st.metric("Below/Borderline/Above", f"{below_count}/{borderline_count}/{above_count}")
        with col4:
            if above_count > 0:
                overall_status = "❌ Above"
            elif borderline_count > 0:
                overall_status = "⚠️ Borderline"
            else:
                overall_status = "✅ Below"
            st.metric("Overall", overall_status)
        
        # Chart section
        st.markdown("### 📊 Analysis Visualization")
        
        # Interval selector for zoom
        col1, col2 = st.columns([3, 1])
        with col2:
            interval_options = ["All Intervals"] + [f"Interval {i+1}" for i in range(len(intervals))]
            selected = st.selectbox("Zoom to:", interval_options)
            
            if selected == "All Intervals":
                st.session_state.selected_interval = None
            else:
                st.session_state.selected_interval = int(selected.split()[-1]) - 1
        
        # Show cumulative drift summary above chart (only in zoomed-out view for VT2)
        if (cumulative_drift is not None and 
            st.session_state.selected_interval is None and 
            run_type == RunType.VT2_INTERVAL):
            
            st.markdown(f"""
                <div class="cumulative-drift-box">
                    <h4>📈 Cumulative Drift Across All Intervals</h4>
                    <div class="value">
                        {cumulative_drift.slope_pct:+.2f}%/min ({cumulative_drift.slope_abs:+.2f} L/min/min)
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem;">
                        Baseline VE: {cumulative_drift.baseline_ve:.1f} L/min (Interval 1) · p = {cumulative_drift.pvalue:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        fig = create_main_chart(
            breath_df, results, intervals, params,
            st.session_state.selected_interval,
            cumulative_drift,
            run_type
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("### 📋 Interval Results")
        
        # Get expected drift for tooltip
        if run_type == RunType.VT1_STEADY:
            drift_pct = params.expected_drift_pct_vt1
        else:
            drift_pct = params.expected_drift_pct_vt2
        
        # Create results dataframe
        results_data = []
        for r in results:
            # Format drift with p-value
            drift_str = f"{r.ve_drift_rate:.2f} (p={r.slope_pvalue:.2f})"
            
            row_data = {
                "Interval": r.interval.interval_num,
                "Status": r.status.value,
                "VE Drift (L/min/min)": drift_str,
                "Peak VE (L/min)": f"{r.peak_ve:.1f}",
                "Peak CUSUM": f"{r.peak_cusum:.1f}",
                "Final CUSUM": f"{r.final_cusum:.1f}",
                "Alarm Time": f"{r.alarm_time:.1f}s" if r.alarm_time else "—"
            }
            
            # Add cumulative drift column for VT2 runs
            if run_type == RunType.VT2_INTERVAL and cumulative_drift is not None:
                # Calculate cumulative drift from baseline to this interval
                if r.interval.interval_num == 1:
                    row_data["Cumulative Drift"] = "— (baseline)"
                else:
                    ve_change = r.last_60s_avg_ve - cumulative_drift.baseline_ve
                    pct_change = (ve_change / cumulative_drift.baseline_ve) * 100
                    row_data["Cumulative Drift"] = f"{pct_change:+.1f}% ({ve_change:+.1f} L/min)"
            
            results_data.append(row_data)
        
        results_df = pd.DataFrame(results_data)
        
        # Build column config
        column_config = {
            "Status": st.column_config.TextColumn(
                "Status",
                width="medium",
                help="Classification based on CUSUM alarm state and slope relative to expected drift for this domain"
            ),
            "VE Drift (L/min/min)": st.column_config.TextColumn(
                "VE Drift (L/min/min)",
                help=f"Theil-Sen robust slope estimate. P-value tests whether drift significantly exceeds expected drift for this domain ({drift_pct:.1f}% per min)"
            ),
            "Peak VE (L/min)": st.column_config.TextColumn(
                "Peak VE (L/min)",
                help="Maximum filtered minute ventilation observed during this interval"
            ),
            "Peak CUSUM": st.column_config.TextColumn(
                "Peak CUSUM",
                help="Highest CUSUM score reached. Alarm triggers when this exceeds the threshold"
            ),
            "Final CUSUM": st.column_config.TextColumn(
                "Final CUSUM",
                help="CUSUM score at interval end. Used to determine if alarm 'recovered' (fell below half the threshold)"
            ),
            "Alarm Time": st.column_config.TextColumn(
                "Alarm Time",
                help="Time when CUSUM first exceeded threshold, accounting for system lag. Dash indicates no alarm triggered"
            )
        }
        
        # Add cumulative drift column config for VT2
        if run_type == RunType.VT2_INTERVAL and cumulative_drift is not None:
            column_config["Cumulative Drift"] = st.column_config.TextColumn(
                "Cumulative Drift",
                help="Change in last-60s average VE from Interval 1 baseline. Measures drift accumulation across the workout."
            )
        
        # Display as interactive table with tooltips via column_config
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        # Detailed cards for borderline/above intervals
        attention_intervals = [r for r in results if r.status != IntervalStatus.BELOW_THRESHOLD]
        if attention_intervals:
            st.markdown("### ⚠️ Intervals Requiring Attention")
            
            for r in attention_intervals:
                status_color = "#dc3545" if r.status == IntervalStatus.ABOVE_THRESHOLD else "#ffc107"
                st.markdown(f"""
                    <div class="result-card" style="border-left-color: {status_color};">
                        <h4 style="margin:0 0 0.5rem 0;">Interval {r.interval.interval_num} - {r.status.value}</h4>
                        <div class="metric-row">
                            <span class="metric-label">Duration</span>
                            <span class="metric-value">{(r.interval.end_time - r.interval.start_time)/60:.1f} min</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">VE Drift Rate</span>
                            <span class="metric-value">{r.ve_drift_rate:.2f} L/min per min (p={r.slope_pvalue:.2f})</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Peak CUSUM</span>
                            <span class="metric-value">{r.peak_cusum:.1f}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Alarm Triggered</span>
                            <span class="metric-value">{f"{r.alarm_time:.1f}s" if r.alarm_time else "N/A"}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        # No data loaded - show instructions
        st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
                <h2 style="color: #1a1a2e;">👈 Upload a VitalPro CSV to begin</h2>
                <p style="color: #666; max-width: 500px; margin: 1rem auto;">
                    Upload your Tymewear VitalPro data file using the sidebar. 
                    The app will automatically detect intervals and analyze VE drift 
                    to determine if your run was performed within the target threshold zone.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show example of what the app does
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **1. Data Processing**
            - Extracts breath-by-breath VE data from VitalPro CSV
            - Stage 0: Outlier rejection - removes breaths >2.5 MAD from rolling local median (inspired by Tymewear validation study)
            - Stage 1: Rolling median filter (5 breaths) removes remaining artifacts
            - Stage 2: 5-second bin averaging standardizes time series
            - Uses breath timestamps (not row index) for accurate timing
            
            **2. Interval Detection**
            - Auto-detects work/recovery intervals from power data
            - Finds where steady high power transitions to sustained ramp-down
            - First interval always starts at t=0
            - Intervals are always whole minutes; rest rounds to 0.5-minute increments
            
            **3. CUSUM Analysis (per interval)**
            - 150s kinetic blanking period
            - Self-calibrates baseline from 150-180s window
            - Expected drift as % of baseline VE per minute
            - Sigma fixed as % of baseline VE (VT1: 10%, VT2: 5%)
            - Detects sustained drift exceeding threshold
            
            **4. Theil-Sen Slope Analysis**
            - Robust slope estimation on post-calibration data
            - Compares observed drift to expected drift for domain
            - P-value indicates statistical significance
            
            **5. Cumulative Drift (VT2 only)**
            - Tracks VE drift across all intervals
            - Uses last 60s average from each interval
            - Theil-Sen regression across elapsed workout time
            - Shows overall fatigue/drift accumulation
            
            **6. Classification**
            - **Below Threshold**: CUSUM no alarm + slope ≤ expected, OR CUSUM recovered + slope ≤ expected
            - **Borderline**: Either CUSUM or slope indicates drift (but not both conclusively)
            - **Above Threshold**: CUSUM alarm sustained (not recovered)
            """)

if __name__ == "__main__":
    main()
