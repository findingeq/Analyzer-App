"""
VT Threshold Analyzer - Desktop Application
Analyzes Tymewear VitalPro respiratory data to assess VT1/VT2 compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import theilslopes
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from streamlit_js_eval import streamlit_js_eval

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

class RunType(Enum):
    VT1_STEADY = "VT1 (Steady State)"
    VT2_INTERVAL = "VT2 (Intervals)"

class IntervalStatus(Enum):
    BELOW_THRESHOLD = "âœ… Below Threshold"
    BORDERLINE = "âš ï¸ Borderline"
    ABOVE_THRESHOLD = "âŒ Above Threshold"

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
    expected_drift_pct_vt2: float = 2.5   # % per minute (slow component in heavy domain)
    # Sigma as percentage of baseline VE
    sigma_pct_vt1: float = 7.0   # % of cal_ve_mean
    sigma_pct_vt2: float = 4.0   # % of cal_ve_mean
    # Filtering parameters
    median_window: int = 9  # breaths
    bin_size: float = 4.0   # seconds
    # Ceiling-based analysis parameters
    vt1_ve_ceiling: float = 100.0  # L/min - user-provided VT1 VE ceiling
    vt2_ve_ceiling: float = 120.0  # L/min - user-provided VT2 VE ceiling
    use_thresholds_for_all: bool = False  # If True, use ceiling-based for all intervals
    ceiling_warmup_sec: float = 20.0  # seconds - warm-up period for ceiling-based CUSUM

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
    ve_drift_rate: float  # Theil-Sen slope (L/min per minute)
    ve_drift_pct: float   # Drift as % of baseline per minute
    slope_pvalue: float   # P-value testing if slope exceeds expected
    baseline_ve: float    # Calibration baseline VE
    peak_ve: float
    peak_cusum: float
    final_cusum: float
    alarm_time: Optional[float]
    cusum_recovered: bool  # True if CUSUM alarm triggered but then recovered
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
    # CUSUM threshold for percentage display
    cusum_threshold: float = 0.0
    # New fields for table display
    initial_ve: float = 0.0      # Average VE in calibration window
    avg_ve: float = 0.0          # Average VE in post-blanking period
    terminal_ve: float = 0.0     # Average VE in last minute of interval
    # Flag to indicate if ceiling-based analysis was used
    is_ceiling_based: bool = False

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
# SIGNAL FILTERING (Two-Stage)
# ============================================================================

def apply_median_filter(ve_raw: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Stage 1: Apply rolling median filter in breath domain.
    Removes non-physiological spikes (coughs, sensor errors).
    
    A 9-breath window at ~55 br/min covers ~10 seconds, providing
    robust outlier rejection while preserving physiological trends.
    """
    if len(ve_raw) < window:
        return ve_raw.copy()

    ve_median = median_filter(ve_raw, size=window, mode='nearest')
    return ve_median


def apply_time_binning(ve_clean: np.ndarray, breath_times: np.ndarray,
                       bin_size: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Convert irregular breath data to uniform time series via bin averaging.
    Uses bin-start timestamp convention.
    Empty bins are filled via linear interpolation.
    
    A 4-second bin at ~55 br/min captures ~3-4 breaths per bin,
    providing reliable averaging while maintaining temporal resolution.

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
                           params: AnalysisParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply two-stage hybrid filtering:
    Stage 1: Rolling median (breath domain) - removes outliers/artifacts
    Stage 2: Time binning (time domain) - standardizes accumulation rate

    Returns:
        ve_median: Median-filtered breath values (for visualization dots)
        bin_times: Bin start timestamps
        ve_binned: Binned VE values (for CUSUM analysis)
    """
    # Stage 1: Median filter
    ve_median = apply_median_filter(ve_raw, params.median_window)

    # Stage 2: Time binning
    bin_times, ve_binned = apply_time_binning(ve_median, breath_times, params.bin_size)

    return ve_median, bin_times, ve_binned

# ============================================================================
# INTERVAL DETECTION
# ============================================================================

def detect_intervals(power_df: pd.DataFrame, breath_df: pd.DataFrame) -> Tuple[RunType, int, float, float]:
    """
    Detect run format and intervals from power data using plateau counting and validation.
    Returns run type, number of intervals, interval duration (min), and rest duration (min).

    Key assumptions:
    - First interval ALWAYS starts at t=0
    - Intervals are ALWAYS whole minutes (1m, 2m, 3m, 4m, etc.)
    - Rest periods are in 0.5m increments (0.5m, 1m, 1.5m, etc.)
    - Recording starts when first interval begins
    - Recording ends after last recovery
    - Equal number of intervals and recoveries
    - Each cycle follows inverted parabola pattern: ramp-up â†’ plateau â†’ ramp-down â†’ trough

    Detection approach:
    1. Count high plateaus (intervals) and low plateaus (recoveries) - must be equal
    2. Measure plateau durations, add ~10s for ramp-up/ramp-down to estimate actual durations
    3. Round intervals to nearest 1 min, recoveries to nearest 0.5 min
    4. Validate against total duration: total â‰ˆ N Ã— (interval + recovery)
    5. If validation fails, try adjacent recovery durations (Â±0.5 min) to find best fit
    """
    if power_df.empty or len(power_df) < 10:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    power = power_df['power'].values
    time = power_df['time'].values

    # Skip NaN and negative values (negative values are calibration artifacts)
    valid_mask = ~np.isnan(power) & (power >= 0)
    if np.sum(valid_mask) < 100:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # Check if this is interval training (large power range) or steady state
    valid_power = power[valid_mask]
    valid_time = time[valid_mask]
    p10 = np.nanpercentile(valid_power, 10)
    p90 = np.nanpercentile(valid_power, 90)

    # Calculate power variability - intervals have high variability
    power_range = p90 - p10

    # If work and rest aren't clearly separated, it's steady state
    is_interval = (p90 > p10 * 1.5 and power_range > 50) or (p90 > p10 * 2)

    if not is_interval:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # Smooth power data with rolling median (10-second window) to reduce noise
    # Median preserves edges better than mean and is more robust to outliers
    window_size = min(10, len(valid_power) // 10)
    if window_size < 3:
        window_size = 3
    # Use pandas for rolling median on the valid power values
    power_series = pd.Series(valid_power)
    power_smooth = power_series.rolling(window=window_size, center=True).median()
    power_smooth = power_smooth.bfill().ffill().values

    # Calculate threshold using K-Means clustering to find work/rest separation
    # K-Means finds natural cluster centers, making threshold robust to distribution skew
    X = power_smooth.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(X)
    # Threshold is midpoint between the two cluster centers
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = (cluster_centers[0] + cluster_centers[1]) / 2

    # Classify each point as high (1) or low (0)
    is_high = (power_smooth > threshold).astype(int)

    # Find transitions (changes between high and low)
    transitions = np.diff(is_high)
    # +1 = low-to-high (start of interval ramp-up), -1 = high-to-low (start of recovery ramp-down)

    low_to_high_indices = np.where(transitions == 1)[0] + 1  # Start of interval ramp-ups
    high_to_low_indices = np.where(transitions == -1)[0] + 1  # Start of recovery ramp-downs

    # If no clear transitions found, treat as steady state
    if len(high_to_low_indices) < 1:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.VT1_STEADY, 1, total_duration, 0.0

    # STEP 1: Count plateaus
    # Number of high plateaus = number of high-to-low transitions (each interval ends with one)
    # First interval starts at t=0, so we count high_to_low transitions as interval count
    num_high_plateaus = len(high_to_low_indices)

    # Number of low plateaus = number of recoveries
    # Check if recording starts high (first interval) or low
    starts_high = is_high[0] == 1

    if starts_high:
        # Recording starts in interval 1, so:
        # - num_intervals = num_high_to_low transitions
        # - num_recoveries should equal num_intervals
        num_low_plateaus = len(low_to_high_indices)
        # If last transition was high-to-low, there's a final recovery plateau
        if len(high_to_low_indices) > 0:
            last_transition_idx = max(
                high_to_low_indices[-1] if len(high_to_low_indices) > 0 else 0,
                low_to_high_indices[-1] if len(low_to_high_indices) > 0 else 0
            )
            if len(high_to_low_indices) > 0 and high_to_low_indices[-1] == last_transition_idx:
                num_low_plateaus += 1  # Final recovery after last interval

    # STEP 2: Measure plateau durations and add ramp adjustment
    RAMP_ADJUSTMENT_SEC = 10  # ~10 seconds for ramp-up/ramp-down combined

    interval_plateau_durations = []
    recovery_plateau_durations = []

    # First interval: from t=0 to first high-to-low transition
    if len(high_to_low_indices) > 0 and starts_high:
        first_plateau_end = valid_time[high_to_low_indices[0]]
        # This is approximately the plateau duration (slightly less than actual interval)
        interval_plateau_durations.append(first_plateau_end)

    # Measure subsequent plateaus
    for i in range(len(high_to_low_indices)):
        recovery_start_idx = high_to_low_indices[i]
        recovery_start_time = valid_time[recovery_start_idx]

        # Find the next low-to-high transition (end of this recovery plateau)
        next_interval_starts = low_to_high_indices[low_to_high_indices > recovery_start_idx]

        if len(next_interval_starts) > 0:
            recovery_end_idx = next_interval_starts[0]
            recovery_end_time = valid_time[recovery_end_idx]
            recovery_plateau_dur = recovery_end_time - recovery_start_time
            recovery_plateau_durations.append(recovery_plateau_dur)

            # Next interval plateau: from low-to-high to next high-to-low
            next_recovery_starts = high_to_low_indices[high_to_low_indices > recovery_end_idx]
            if len(next_recovery_starts) > 0:
                interval_end_idx = next_recovery_starts[0]
                interval_end_time = valid_time[interval_end_idx]
                interval_plateau_dur = interval_end_time - recovery_end_time
                interval_plateau_durations.append(interval_plateau_dur)
        else:
            # This is the final recovery (no more intervals after)
            # Estimate duration to end of recording
            final_recovery_dur = valid_time[-1] - recovery_start_time
            if final_recovery_dur > 15:  # At least 15 seconds
                recovery_plateau_durations.append(final_recovery_dur)

    # Calculate median plateau durations and add ramp adjustment
    if len(interval_plateau_durations) > 0:
        median_interval_plateau_sec = np.median(interval_plateau_durations)
        # Add ramp adjustment and convert to minutes
        interval_duration_estimate = (median_interval_plateau_sec + RAMP_ADJUSTMENT_SEC) / 60.0
        # Round to nearest whole minute
        interval_duration = round(interval_duration_estimate)
        if interval_duration < 1:
            interval_duration = 1
    else:
        interval_duration = 4  # Default

    if len(recovery_plateau_durations) > 0:
        median_recovery_plateau_sec = np.median(recovery_plateau_durations)
        # Add ramp adjustment and convert to minutes
        recovery_duration_estimate = (median_recovery_plateau_sec + RAMP_ADJUSTMENT_SEC) / 60.0
        # Round to nearest 0.5 minutes
        rest_duration = round(recovery_duration_estimate * 2) / 2
        if rest_duration < 0.5:
            rest_duration = 0.5
    else:
        rest_duration = 1.0  # Default

    # STEP 3: Validate against total duration and find best fit
    total_duration_min = breath_df['breath_time'].max() / 60.0

    # Try different recovery durations to find the best fit
    # Recovery has more rounding uncertainty (0.5 min increments)
    candidate_recoveries = [
        rest_duration - 0.5,
        rest_duration,
        rest_duration + 0.5
    ]
    candidate_recoveries = [r for r in candidate_recoveries if r >= 0.5]  # Must be at least 0.5

    best_fit_recovery = rest_duration
    best_fit_error = float('inf')
    best_fit_n = 0

    for candidate_recovery in candidate_recoveries:
        period = interval_duration + candidate_recovery
        # N = total_time / period
        n_calculated = total_duration_min / period
        n_rounded = round(n_calculated)

        if n_rounded < 1:
            continue

        # Calculate what the total duration would be with this N
        expected_total = n_rounded * period
        error = abs(expected_total - total_duration_min)

        # Also check that n_rounded is close to n_calculated (good fit)
        n_error = abs(n_calculated - n_rounded)

        # Prefer solutions where n_calculated is close to a whole number
        if n_error < 0.2 and error < best_fit_error:
            best_fit_error = error
            best_fit_recovery = candidate_recovery
            best_fit_n = n_rounded

    # Use the best fit values
    rest_duration = best_fit_recovery
    num_intervals = best_fit_n if best_fit_n > 0 else round(total_duration_min / (interval_duration + rest_duration))

    # Ensure at least 1 interval
    num_intervals = max(1, num_intervals)

    # Sanity check: don't exceed what's physically possible
    max_possible = int(total_duration_min / interval_duration) + 1
    num_intervals = min(num_intervals, max_possible)

    # STEP 5: Validate plateau alignment
    # Verify that intervals contain high plateaus and recoveries contain low plateaus
    # If validation fails, try different interval durations

    def find_plateau_centers(is_high_arr, valid_time_arr):
        """Find the center time of each high and low plateau."""
        high_plateau_centers = []
        low_plateau_centers = []

        # Find contiguous segments
        in_high = is_high_arr[0] == 1
        segment_start = 0

        for i in range(1, len(is_high_arr)):
            if is_high_arr[i] != is_high_arr[i-1]:
                # Transition occurred - record the segment
                segment_end = i - 1
                segment_center_time = (valid_time_arr[segment_start] + valid_time_arr[segment_end]) / 2

                if in_high:
                    high_plateau_centers.append(segment_center_time)
                else:
                    low_plateau_centers.append(segment_center_time)

                segment_start = i
                in_high = is_high_arr[i] == 1

        # Don't forget the last segment
        segment_end = len(is_high_arr) - 1
        segment_center_time = (valid_time_arr[segment_start] + valid_time_arr[segment_end]) / 2
        if in_high:
            high_plateau_centers.append(segment_center_time)
        else:
            low_plateau_centers.append(segment_center_time)

        return high_plateau_centers, low_plateau_centers

    def validate_alignment(n_intervals, int_dur, rec_dur, high_centers, low_centers, tolerance_sec=30):
        """
        Validate that grid-based intervals/recoveries align with detected plateaus.
        Returns (is_valid, alignment_score) where lower score is better.

        Rules:
        - Each interval must contain a high plateau center
        - Each recovery must contain a low plateau center
        - Intervals cannot contain low plateau centers
        - Recoveries cannot contain high plateau centers
        """
        period = (int_dur + rec_dur) * 60  # Convert to seconds
        int_dur_sec = int_dur * 60
        rec_dur_sec = rec_dur * 60

        total_error = 0
        violations = 0

        for i in range(n_intervals):
            # Interval i spans: [i * period, i * period + int_dur_sec]
            int_start = i * period
            int_end = int_start + int_dur_sec

            # Recovery i spans: [i * period + int_dur_sec, (i+1) * period]
            rec_start = int_end
            rec_end = (i + 1) * period

            # Check if this interval contains at least one high plateau center
            high_in_interval = [c for c in high_centers if int_start <= c <= int_end]
            if len(high_in_interval) == 0:
                # No high plateau in this interval - violation
                # Find distance to nearest high plateau
                distances = [min(abs(c - int_start), abs(c - int_end)) for c in high_centers]
                if distances:
                    total_error += min(distances)
                violations += 1

            # Check if this interval contains any low plateau centers (violation)
            low_in_interval = [c for c in low_centers if int_start + tolerance_sec <= c <= int_end - tolerance_sec]
            if len(low_in_interval) > 0:
                violations += len(low_in_interval)

            # Check if this recovery contains at least one low plateau center
            low_in_recovery = [c for c in low_centers if rec_start <= c <= rec_end]
            if len(low_in_recovery) == 0:
                # No low plateau in this recovery - violation
                distances = [min(abs(c - rec_start), abs(c - rec_end)) for c in low_centers]
                if distances:
                    total_error += min(distances)
                violations += 1

            # Check if this recovery contains any high plateau centers (violation)
            high_in_recovery = [c for c in high_centers if rec_start + tolerance_sec <= c <= rec_end - tolerance_sec]
            if len(high_in_recovery) > 0:
                violations += len(high_in_recovery)

        is_valid = violations == 0
        return is_valid, violations, total_error

    # Find plateau centers
    high_plateau_centers, low_plateau_centers = find_plateau_centers(is_high, valid_time)

    # Validate current estimate
    is_valid, violations, alignment_error = validate_alignment(
        num_intervals, interval_duration, rest_duration,
        high_plateau_centers, low_plateau_centers
    )

    if not is_valid:
        # Try alternative interval durations (Â±1 min) and recovery durations (Â±0.5 min)
        best_config = (num_intervals, interval_duration, rest_duration)
        best_violations = violations
        best_alignment_error = alignment_error

        candidate_intervals = [interval_duration - 1, interval_duration, interval_duration + 1]
        candidate_intervals = [d for d in candidate_intervals if d >= 1]

        candidate_recoveries_step5 = [rest_duration - 0.5, rest_duration, rest_duration + 0.5]
        candidate_recoveries_step5 = [r for r in candidate_recoveries_step5 if r >= 0.5]

        for cand_int in candidate_intervals:
            for cand_rec in candidate_recoveries_step5:
                period = cand_int + cand_rec
                n_calc = total_duration_min / period
                n_rounded = round(n_calc)

                # Only consider if N is close to a whole number
                if n_rounded < 1 or abs(n_calc - n_rounded) > 0.15:
                    continue

                is_valid_cand, viol_cand, err_cand = validate_alignment(
                    n_rounded, cand_int, cand_rec,
                    high_plateau_centers, low_plateau_centers
                )

                # Prefer configurations with fewer violations, then lower error
                if viol_cand < best_violations or (viol_cand == best_violations and err_cand < best_alignment_error):
                    best_violations = viol_cand
                    best_alignment_error = err_cand
                    best_config = (n_rounded, cand_int, cand_rec)

        # Use the best configuration found
        num_intervals, interval_duration, rest_duration = best_config

    # Single interval = VT1, Multiple intervals = VT2
    if num_intervals == 1:
        return RunType.VT1_STEADY, 1, float(interval_duration), 0.0
    else:
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
            ve_drift_pct=0,
            slope_pvalue=1.0,
            baseline_ve=0,
            peak_ve=0,
            peak_cusum=0,
            final_cusum=0,
            alarm_time=None,
            cusum_recovered=False,
            cusum_values=np.array([0]),
            time_values=np.array([interval.start_time]),
            ve_binned=np.array([0]),
            ve_median=np.array([0]),
            breath_times=np.array([interval.start_time]),
            expected_ve=np.array([0]),
            slope_line_times=np.array([]),
            slope_line_ve=np.array([]),
            last_60s_avg_ve=0,
            cusum_threshold=0,
            initial_ve=0,
            avg_ve=0,
            terminal_ve=0,
            is_ceiling_based=False
        )

    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end]
    breath_times_raw = breath_df['breath_time'].values[idx_start:idx_end]

    # Relative time within interval (in seconds)
    rel_breath_times = breath_times_raw - breath_times_raw[0]

    # Apply hybrid filtering (2-stage: median filter, binning)
    ve_median, bin_times_rel, ve_binned = apply_hybrid_filtering(
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

        # Slope line for visualization - extend from calibration end to full interval end
        interval_end_rel = interval.end_time - breath_times_raw[0]
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
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
    #
    # Classification rules:
    # BELOW:      (no alarm + slope â‰¤ expected) OR (alarm + recovered + slope â‰¤ expected)
    # BORDERLINE: (no alarm + slope > expected) OR (alarm + recovered + slope > expected) OR (alarm + no recovery + slope â‰¤ expected)
    # ABOVE:      (alarm + no recovery + slope > expected)
    #
    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)
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
            # Alarm triggered but recovered
            if slope_at_or_below:
                status = IntervalStatus.BELOW_THRESHOLD
            else:
                status = IntervalStatus.BORDERLINE
        else:
            # Alarm triggered, no recovery
            if slope_at_or_below:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
    
    # Convert times to absolute for output
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])

    # Calculate drift as percentage of baseline
    ve_drift_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0

    # Calculate new metrics for table display BEFORE extending arrays
    # Initial VE: average VE in calibration window (already have as cal_ve_mean)
    initial_ve = cal_ve_mean

    # Avg VE: average VE in post-blanking period
    post_blanking_mask = bin_times_rel >= params.blanking_period
    if np.sum(post_blanking_mask) > 0:
        avg_ve = np.mean(ve_binned[post_blanking_mask])
    else:
        avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    # Terminal VE: average VE in last minute (already have as last_60s_avg_ve)
    terminal_ve = last_60s_avg_ve

    # Extend CUSUM line to the end of the interval
    # Add a final point at interval.end_time with the final CUSUM value
    if len(abs_bin_times) > 0 and abs_bin_times[-1] < interval.end_time:
        abs_bin_times = np.append(abs_bin_times, interval.end_time)
        cusum = np.append(cusum, cusum[-1])
        ve_binned = np.append(ve_binned, ve_binned[-1])
        ve_expected = np.append(ve_expected, ve_expected[-1])

    return IntervalResult(
        interval=interval,
        status=status,
        ve_drift_rate=slope,
        ve_drift_pct=ve_drift_pct,
        slope_pvalue=slope_pvalue,
        baseline_ve=cal_ve_mean,
        peak_ve=np.max(ve_binned) if len(ve_binned) > 0 else 0,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        alarm_time=alarm_time,
        cusum_recovered=cusum_recovered,
        cusum_values=cusum,
        time_values=abs_bin_times,
        ve_binned=ve_binned,
        ve_median=ve_median,
        breath_times=breath_times_raw,  # Absolute times for breath dots
        expected_ve=ve_expected,
        slope_line_times=abs_slope_line_times,
        slope_line_ve=slope_line_ve,
        last_60s_avg_ve=last_60s_avg_ve,
        cusum_threshold=h,
        initial_ve=initial_ve,
        avg_ve=avg_ve,
        terminal_ve=terminal_ve,
        is_ceiling_based=False
    )


def analyze_interval_ceiling(breath_df: pd.DataFrame, interval: Interval,
                             params: AnalysisParams, run_type: RunType) -> IntervalResult:
    """
    Perform ceiling-based CUSUM analysis on a single interval.
    
    Uses user-provided VT1/VT2 VE ceiling as the threshold.
    No blanking/calibration period - starts accumulating after brief warm-up.
    Detects when VE exceeds the ceiling (absolute position, not drift).
    
    Best for short intervals (<6 min) where drift analysis is impractical.
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
            ve_drift_pct=0,
            slope_pvalue=1.0,
            baseline_ve=0,
            peak_ve=0,
            peak_cusum=0,
            final_cusum=0,
            alarm_time=None,
            cusum_recovered=False,
            cusum_values=np.array([0]),
            time_values=np.array([interval.start_time]),
            ve_binned=np.array([0]),
            ve_median=np.array([0]),
            breath_times=np.array([interval.start_time]),
            expected_ve=np.array([0]),
            slope_line_times=np.array([]),
            slope_line_ve=np.array([]),
            last_60s_avg_ve=0,
            cusum_threshold=0,
            initial_ve=0,
            avg_ve=0,
            terminal_ve=0,
            is_ceiling_based=True
        )

    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end]
    breath_times_raw = breath_df['breath_time'].values[idx_start:idx_end]

    # Relative time within interval (in seconds)
    rel_breath_times = breath_times_raw - breath_times_raw[0]

    # Apply hybrid filtering (2-stage: median filter, binning)
    ve_median, bin_times_rel, ve_binned = apply_hybrid_filtering(
        ve_raw, rel_breath_times, params
    )

    # Convert bin times to minutes for drift calculations
    bin_times_min = bin_times_rel / 60.0

    # Get ceiling based on run type
    if run_type == RunType.VT1_STEADY:
        ceiling_ve = params.vt1_ve_ceiling
        h_mult = params.h_multiplier_vt1
        sigma_pct = params.sigma_pct_vt1
    else:
        ceiling_ve = params.vt2_ve_ceiling
        h_mult = params.h_multiplier_vt2
        sigma_pct = params.sigma_pct_vt2
    
    # Fixed parameters based on ceiling
    sigma_ref = (sigma_pct / 100.0) * ceiling_ve
    k = params.slack_multiplier * sigma_ref
    h = h_mult * sigma_ref
    
    # Expected VE is just the ceiling (flat line)
    ve_expected = np.full(len(bin_times_rel), ceiling_ve)
    
    # CUSUM calculation - detect VE exceeding ceiling
    cusum = np.zeros(len(bin_times_rel))
    s = 0.0
    alarm_time = None
    alarm_triggered = False
    
    for i in range(len(bin_times_rel)):
        # Start accumulating after warm-up period (just median filter warm-up)
        if bin_times_rel[i] >= params.ceiling_warmup_sec:
            # Residual: how much VE exceeds ceiling
            residual = ve_binned[i] - ceiling_ve
            # One-sided upper CUSUM (detects VE above ceiling)
            s = max(0, s + residual - k)
            
            if s > h and not alarm_triggered:
                alarm_time = bin_times_rel[i] + breath_times_raw[0]  # Convert to absolute time
                alarm_triggered = True
        
        cusum[i] = s
    
    # CUSUM metrics
    peak_cusum = np.max(cusum)
    final_cusum = cusum[-1] if len(cusum) > 0 else 0
    recovered_threshold = h / 2
    
    # Theil-Sen slope estimation on post-warmup data (for information only)
    analysis_mask = bin_times_rel >= params.ceiling_warmup_sec
    n_analysis_points = np.sum(analysis_mask)
    
    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]

        # Theil-Sen robust slope estimation
        slope, intercept, lo_slope, hi_slope = theilslopes(analysis_ve, analysis_times_min)

        # Slope line for visualization
        interval_end_rel = interval.end_time - breath_times_raw[0]
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
        
        # P-value not really meaningful for ceiling-based, but compute anyway
        slope_pvalue = 0.5  # Neutral - slope isn't the primary metric here
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
    
    # Classification for ceiling-based analysis
    # Simpler than drift-based: no slope analysis, just CUSUM alarm + recovery
    # Below = no alarm OR alarm triggered but recovered
    # Above = alarm triggered and no recovery
    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)

    if not cusum_alarm or cusum_recovered:
        # No alarm, or alarm triggered but recovered
        status = IntervalStatus.BELOW_THRESHOLD
    else:
        # Alarm triggered and no recovery
        status = IntervalStatus.ABOVE_THRESHOLD
    
    # Convert times to absolute for output
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])

    # Calculate drift as percentage of ceiling
    ve_drift_pct = (slope / ceiling_ve * 100.0) if ceiling_ve > 0 else 0.0

    # Calculate new metrics for table display BEFORE extending arrays
    # Initial VE: for ceiling-based, not used (will be hidden in table)
    initial_ve = ceiling_ve

    # Avg VE: for ceiling-based, average VE for entire interval (no warmup exclusion)
    avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    # Terminal VE: average VE in last minute
    terminal_ve = last_60s_avg_ve

    # Extend CUSUM line to the end of the interval
    if len(abs_bin_times) > 0 and abs_bin_times[-1] < interval.end_time:
        abs_bin_times = np.append(abs_bin_times, interval.end_time)
        cusum = np.append(cusum, cusum[-1])
        ve_binned = np.append(ve_binned, ve_binned[-1])
        ve_expected = np.append(ve_expected, ceiling_ve)

    return IntervalResult(
        interval=interval,
        status=status,
        ve_drift_rate=slope,
        ve_drift_pct=ve_drift_pct,
        slope_pvalue=slope_pvalue,
        baseline_ve=ceiling_ve,  # Use ceiling as "baseline" for display
        peak_ve=np.max(ve_binned) if len(ve_binned) > 0 else 0,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        alarm_time=alarm_time,
        cusum_recovered=cusum_recovered,
        cusum_values=cusum,
        time_values=abs_bin_times,
        ve_binned=ve_binned,
        ve_median=ve_median,
        breath_times=breath_times_raw,
        expected_ve=ve_expected,
        slope_line_times=abs_slope_line_times,
        slope_line_ve=slope_line_ve,
        last_60s_avg_ve=last_60s_avg_ve,
        cusum_threshold=h,
        initial_ve=initial_ve,
        avg_ve=avg_ve,
        terminal_ve=terminal_ve,
        is_ceiling_based=True
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
    """Create the main visualization chart with CUSUM overlaid on secondary Y-axis."""

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Determine x-axis range
    if selected_interval is not None and selected_interval < len(results):
        result = results[selected_interval]
        x_min = result.interval.start_time - 10
        x_max = result.interval.end_time + 10
    else:
        x_min = breath_df['breath_time'].min()
        x_max = breath_df['breath_time'].max()

    # Add grey shading for periods outside work intervals
    # 1. Shading before first interval (from chart origin to first interval start)
    first_interval_start = intervals[0].start_time if intervals else 0
    # Use 0 as the start point to ensure shading from the very beginning of the chart
    chart_start = min(0, breath_df['breath_time'].min())
    if first_interval_start > chart_start:
        fig.add_shape(
            type="rect",
            x0=chart_start,
            x1=first_interval_start,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor='rgba(220, 220, 220, 0.6)',
            layer="below",
            line_width=0,
        )

    # 2. Recovery period shading (grey) - gaps between intervals
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            recovery_start = intervals[i].end_time
            recovery_end = intervals[i + 1].start_time
            if recovery_end > recovery_start:
                fig.add_shape(
                    type="rect",
                    x0=recovery_start,
                    x1=recovery_end,
                    y0=0,
                    y1=1,
                    yref="paper",  # Use paper coordinates to span full height
                    fillcolor='rgba(220, 220, 220, 0.6)',  # Light grey, more visible
                    layer="below",
                    line_width=0,
                )

    # 3. Shading after last interval to end of data
    last_interval_end = intervals[-1].end_time if intervals else 0
    data_end = breath_df['breath_time'].max()
    if data_end > last_interval_end:
        fig.add_shape(
            type="rect",
            x0=last_interval_end,
            x1=data_end,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor='rgba(220, 220, 220, 0.6)',
            layer="below",
            line_width=0,
        )

    # Add interval labels (no colored shading for work intervals)
    for i, interval in enumerate(intervals):
        # Add interval label
        fig.add_annotation(
            x=(interval.start_time + interval.end_time) / 2,
            y=1.02,
            yref="paper",
            text=f"Int {interval.interval_num}",
            showarrow=False,
            font=dict(size=10, color="#666")
        )

    # Calculate VE range for scaling CUSUM
    all_ve_values = []
    for result in results:
        all_ve_values.extend(result.ve_binned)
    ve_min = min(all_ve_values) if all_ve_values else 0
    ve_max = max(all_ve_values) if all_ve_values else 100

    # Calculate max CUSUM for scaling
    max_cusum = max(r.peak_cusum for r in results) if results else 1
    if max_cusum < 1:
        max_cusum = 1

    # Scale factor: map CUSUM so its max is at ~30% of VE range (below VE data)
    # CUSUM will be displayed in the lower portion of the chart
    cusum_display_max = ve_min * 0.7  # Show CUSUM in lower 30% of chart
    cusum_scale = cusum_display_max / max_cusum if max_cusum > 0 else 1

    # First, plot the entire breath-by-breath data including recoveries
    breath_times = breath_df['breath_time'].values
    ve_raw = breath_df['ve_raw'].values

    # Apply median filter to all breath data
    ve_median_all = apply_median_filter(ve_raw, params.median_window)

    # Plot all breath dots (including recoveries)
    fig.add_trace(
        go.Scatter(
            x=breath_times,
            y=ve_median_all,
            mode='markers',
            name='Breaths',
            marker=dict(size=3, color='rgba(46, 134, 171, 0.2)'),
            showlegend=True,
            hoverinfo='skip'
        ),
        secondary_y=False
    )

    # Apply binning to entire dataset for continuous VE line
    _, bin_times_all, ve_binned_all = apply_hybrid_filtering(ve_raw, breath_times, params)

    # Create hover text for VE line with interval-relative time
    ve_hover_texts = []
    for t, ve_val in zip(bin_times_all, ve_binned_all):
        # Find which interval this time belongs to
        interval_num = None
        rel_time = t
        for interval in intervals:
            if interval.start_time <= t <= interval.end_time:
                interval_num = interval.interval_num
                rel_time = t - interval.start_time
                break

        mins = int(rel_time // 60)
        secs = int(rel_time % 60)

        if interval_num is not None:
            ve_hover_texts.append(
                f"Interval: {interval_num}<br>"
                f"Time: {mins}:{secs:02d}<br>"
                f"VE: {ve_val:.1f} L/min"
            )
        else:
            # Recovery period - show as recovery time
            ve_hover_texts.append(
                f"Recovery<br>"
                f"VE: {ve_val:.1f} L/min"
            )

    # Plot continuous VE smoothed line (including recoveries)
    fig.add_trace(
        go.Scatter(
            x=bin_times_all,
            y=ve_binned_all,
            mode='lines',
            name='VE (smoothed)',
            line=dict(color='#2E86AB', width=2),
            showlegend=True,
            hoverinfo='text',
            hovertext=ve_hover_texts
        ),
        secondary_y=False
    )

    # Track if we've shown legend for ceiling baseline vs observed slope
    shown_ceiling_legend = False
    shown_slope_legend = False

    # Plot each interval's analysis results (slope lines and CUSUM)
    for result in results:
        interval_start = result.interval.start_time

        # For ceiling-based analysis: show horizontal dotted green line at ceiling VE
        # For drift-based analysis: show Theil-Sen slope line (solid green)
        if result.is_ceiling_based:
            # Draw horizontal dotted line at the ceiling VE (baseline_ve holds ceiling value)
            ceiling_ve = result.baseline_ve
            fig.add_trace(
                go.Scatter(
                    x=[result.interval.start_time, result.interval.end_time],
                    y=[ceiling_ve, ceiling_ve],
                    mode='lines',
                    name='VT Ceiling',
                    line=dict(color='#FFA500', width=2, dash='dot'),
                    showlegend=(not shown_ceiling_legend),
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
            shown_ceiling_legend = True

            # Add LOESS trend line through the VE data for this interval
            if len(result.time_values) > 3:  # Need enough points for LOESS
                loess_result = lowess(result.ve_binned, result.time_values, frac=0.4)
                fig.add_trace(
                    go.Scatter(
                        x=loess_result[:, 0],
                        y=loess_result[:, 1],
                        mode='lines',
                        name='VE Trend',
                        line=dict(color='#228B22', width=2),
                        showlegend=(not shown_slope_legend),
                        hoverinfo='skip'
                    ),
                    secondary_y=False
                )
                shown_slope_legend = True
        elif len(result.slope_line_times) > 0:
            # Theil-Sen slope line (solid green, post-calibration only)
            fig.add_trace(
                go.Scatter(
                    x=result.slope_line_times,
                    y=result.slope_line_ve,
                    mode='lines',
                    name='Observed Slope',
                    line=dict(color='#228B22', width=2),
                    showlegend=(not shown_slope_legend),
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
            shown_slope_legend = True

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

        # CUSUM trace on secondary y-axis (scaled to fit below VE)
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
            secondary_y=True
        )

        # Add alarm line if triggered (vertical dotted line only, no label)
        if result.alarm_time is not None:
            fig.add_vline(
                x=result.alarm_time,
                line=dict(color='red', width=2, dash='dot'),
            )

    # Add cumulative drift line (only in zoomed-out view for VT2)
    # Compare by value to handle Streamlit reruns
    is_vt2 = run_type.value == RunType.VT2_INTERVAL.value if hasattr(run_type, 'value') else run_type == RunType.VT2_INTERVAL
    if (cumulative_drift is not None and
        selected_interval is None and
        is_vt2):

        fig.add_trace(
            go.Scatter(
                x=cumulative_drift.line_times,
                y=cumulative_drift.line_ve,
                mode='lines',
                name='Cumulative Drift',
                line=dict(color='rgba(128, 90, 213, 0.2)', width=2),  # Very faint purple
                showlegend=True
            ),
            secondary_y=False
        )

    # Update layout - no title
    fig.update_layout(
        height=500,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        range=[x_min, x_max] if selected_interval is not None else None
    )
    fig.update_yaxes(title_text="VE (L/min)", secondary_y=False)
    # Set CUSUM y-axis range so it stays in lower portion of chart
    fig.update_yaxes(
        title_text="CUSUM Score",
        secondary_y=True,
        showgrid=False,
        range=[0, max_cusum * 3]  # Make CUSUM appear smaller by extending its axis range
    )

    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="VT Threshold Analyzer",
        page_icon="ðŸ«",
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

        div[data-testid="stSidebar"] {
            background: #f8f9fa;
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
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

    # Get default params from dataclass
    default_params = AnalysisParams()

    # Load VT Thresholds from localStorage (persists across sessions)
    if 'vt_thresholds_loaded' not in st.session_state:
        st.session_state.vt_thresholds_loaded = False

    if not st.session_state.vt_thresholds_loaded:
        # Try to load from localStorage
        stored_vt1 = streamlit_js_eval(js_expressions="localStorage.getItem('vt1_ve_ceiling')", key="load_vt1")
        stored_vt2 = streamlit_js_eval(js_expressions="localStorage.getItem('vt2_ve_ceiling')", key="load_vt2")

        if stored_vt1 is not None and stored_vt1 != 'null':
            try:
                st.session_state.saved_vt1_ve = float(stored_vt1)
            except (ValueError, TypeError):
                st.session_state.saved_vt1_ve = default_params.vt1_ve_ceiling
        else:
            st.session_state.saved_vt1_ve = default_params.vt1_ve_ceiling

        if stored_vt2 is not None and stored_vt2 != 'null':
            try:
                st.session_state.saved_vt2_ve = float(stored_vt2)
            except (ValueError, TypeError):
                st.session_state.saved_vt2_ve = default_params.vt2_ve_ceiling
        else:
            st.session_state.saved_vt2_ve = default_params.vt2_ve_ceiling

        # Load "Use Thresholds for All" checkbox state
        stored_use_thresh = streamlit_js_eval(js_expressions="localStorage.getItem('use_thresholds_for_all')", key="load_use_thresh")
        if stored_use_thresh is not None and stored_use_thresh != 'null':
            st.session_state.saved_use_thresholds = stored_use_thresh == 'true'
        else:
            st.session_state.saved_use_thresholds = False

        st.session_state.vt_thresholds_loaded = True
    
    # Sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload VitalPro CSV",
            type=['csv'],
            help="Upload the CSV file exported from the Tymewear VitalPro"
        )
        
        if uploaded_file is not None:
            # Create unique file identifier using name + size
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Check if this is a new file
            if st.session_state.get('current_file_id') != file_id:
                # New file uploaded - clear ALL previous analysis results
                st.session_state.results = None
                st.session_state.intervals = None
                st.session_state.run_type = None
                st.session_state.selected_interval = None
                st.session_state.cumulative_drift = None
                st.session_state.detected_params = None
                st.session_state.breath_df = None
                st.session_state.power_df = None
                st.session_state.current_file_id = file_id
                st.session_state.current_file_name = uploaded_file.name

            # Only parse if we haven't already (or if data was cleared)
            if st.session_state.breath_df is None:
                try:
                    breath_df, metadata, power_df = parse_vitalpro_csv(uploaded_file)
                    st.session_state.breath_df = breath_df
                    st.session_state.power_df = power_df
                    st.success(f"âœ“ Loaded {len(breath_df)} breaths")

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
            else:
                st.success(f"âœ“ Loaded {len(st.session_state.breath_df)} breaths")

        # Analyze button (placed after upload, before Run Format)
        analyze_btn = st.button("Analyze Run", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### Run Format")

        # Get detected params or use defaults
        detected = st.session_state.detected_params or {
            'run_type': RunType.VT1_STEADY,
            'num_intervals': 12,
            'interval_duration': 4.0,
            'recovery_duration': 1.0
        }

        # Use file_id in widget keys to force refresh when file changes
        file_key = st.session_state.get('current_file_id', 'default')

        # Run type selection
        run_type_options = ["VT1 (Steady State)", "VT2 (Intervals)"]
        default_idx = 0 if detected['run_type'] == RunType.VT1_STEADY else 1
        run_type_option = st.selectbox(
            "Run Type",
            options=run_type_options,
            index=default_idx,
            key=f"run_type_{file_key}",
            help="Auto-detected from power data; override if needed"
        )

        col1, col2 = st.columns(2)
        with col1:
            num_intervals = st.number_input(
                "# Intervals",
                min_value=1,
                value=int(detected['num_intervals']),
                step=1,
                key=f"num_int_{file_key}",
                disabled=(run_type_option == "VT1 (Steady State)")
            )
            interval_duration = st.number_input(
                "Interval (min)",
                min_value=1.0,
                value=detected['interval_duration'],
                step=1.0,
                key=f"int_dur_{file_key}",
                disabled=(run_type_option == "VT1 (Steady State)")
            )
        with col2:
            recovery_duration = st.number_input(
                "Recovery (min)",
                min_value=0.0,
                value=detected['recovery_duration'],
                step=0.5,
                key=f"rec_dur_{file_key}",
                disabled=(run_type_option == "VT1 (Steady State)")
            )
            run_speed = st.number_input(
                "Speed (mph)",
                min_value=0.0, max_value=15.0, value=7.6, step=0.1,
                key=f"speed_{file_key}",
                help="Optional: for reference only"
            )
        
        st.markdown("---")
        st.markdown("### VT Thresholds")

        # Use saved values from localStorage if available
        vt1_default = st.session_state.get('saved_vt1_ve', default_params.vt1_ve_ceiling)
        vt2_default = st.session_state.get('saved_vt2_ve', default_params.vt2_ve_ceiling)

        col1, col2 = st.columns(2)
        with col1:
            vt1_ve_ceiling = st.number_input(
                "VT1 VE (L/min)",
                min_value=50.0, max_value=200.0,
                value=vt1_default,
                step=5.0,
                key="vt1_ve_input",
                help="VE ceiling for VT1 runs (from ramp test)"
            )
        with col2:
            vt2_ve_ceiling = st.number_input(
                "VT2 VE (L/min)",
                min_value=50.0, max_value=250.0,
                value=vt2_default,
                step=5.0,
                key="vt2_ve_input",
                help="VE ceiling for VT2 runs (from ramp test)"
            )

        # Save to localStorage when values change
        if vt1_ve_ceiling != st.session_state.get('saved_vt1_ve'):
            streamlit_js_eval(js_expressions=f"localStorage.setItem('vt1_ve_ceiling', '{vt1_ve_ceiling}')", key=f"save_vt1_{vt1_ve_ceiling}")
            st.session_state.saved_vt1_ve = vt1_ve_ceiling

        if vt2_ve_ceiling != st.session_state.get('saved_vt2_ve'):
            streamlit_js_eval(js_expressions=f"localStorage.setItem('vt2_ve_ceiling', '{vt2_ve_ceiling}')", key=f"save_vt2_{vt2_ve_ceiling}")
            st.session_state.saved_vt2_ve = vt2_ve_ceiling
        
        use_thresholds_for_all = st.checkbox(
            "Use Thresholds for All Analysis",
            value=st.session_state.get('saved_use_thresholds', False),
            key=f"use_thresh_{file_key}",
            help="If checked, uses VT1/VT2 VE ceilings for all intervals. "
                 "If unchecked, only uses ceilings for intervals < 6 min."
        )

        # Save checkbox state to localStorage when it changes
        if use_thresholds_for_all != st.session_state.get('saved_use_thresholds'):
            streamlit_js_eval(js_expressions=f"localStorage.setItem('use_thresholds_for_all', '{str(use_thresholds_for_all).lower()}')", key=f"save_use_thresh_{use_thresholds_for_all}")
            st.session_state.saved_use_thresholds = use_thresholds_for_all
        
        st.markdown("---")
        st.markdown("### Analysis Parameters")
        
        params = AnalysisParams()
        # Set ceiling parameters
        params.vt1_ve_ceiling = vt1_ve_ceiling
        params.vt2_ve_ceiling = vt2_ve_ceiling
        params.use_thresholds_for_all = use_thresholds_for_all
        
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
            
            # Run analysis - choose algorithm based on interval duration and checkbox
            results = []
            for interval in intervals:
                interval_duration_sec = interval.end_time - interval.start_time
                interval_duration_min = interval_duration_sec / 60.0
                
                # Use ceiling-based CUSUM if:
                # 1. Checkbox is checked (use_thresholds_for_all), OR
                # 2. Interval duration is less than 6 minutes
                use_ceiling_based = params.use_thresholds_for_all or (interval_duration_min < 6.0)
                
                if use_ceiling_based:
                    result = analyze_interval_ceiling(breath_df, interval, params, run_type)
                else:
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
    if st.session_state.results is not None:
        results = st.session_state.results
        intervals = st.session_state.intervals
        run_type = st.session_state.run_type
        breath_df = st.session_state.breath_df
        cumulative_drift = st.session_state.cumulative_drift
        
        # Summary row - compare by value to handle Streamlit reruns
        below_count = sum(1 for r in results if r.status.value == IntervalStatus.BELOW_THRESHOLD.value)
        borderline_count = sum(1 for r in results if r.status.value == IntervalStatus.BORDERLINE.value)
        above_count = sum(1 for r in results if r.status.value == IntervalStatus.ABOVE_THRESHOLD.value)

        # Calculate Average VE across all post-blanking periods
        all_avg_ve = [r.avg_ve for r in results if r.avg_ve > 0]
        overall_avg_ve = np.mean(all_avg_ve) if all_avg_ve else 0

        # Show 5 columns if cumulative drift available, otherwise 3
        if cumulative_drift is not None:
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Run Type", run_type.value)
        with col2:
            st.metric("Intervals", len(intervals))
        with col3:
            st.metric("Below/Borderline/Above", f"{below_count}/{borderline_count}/{above_count}")

        if cumulative_drift is not None:
            with col4:
                drift_str = f"{cumulative_drift.slope_pct:+.2f}%/min ({cumulative_drift.baseline_ve:.1f} L/min)"
                st.metric("Cumulative Drift", drift_str)
            with col5:
                st.metric("Average VE", f"{overall_avg_ve:.1f} L/min")
        
        fig = create_main_chart(
            breath_df, results, intervals, params,
            st.session_state.selected_interval,
            cumulative_drift,
            run_type
        )
        st.plotly_chart(fig, use_container_width=True)

        # Build interval results table with clickable rows
        # Inject CSS to remove column and row gaps for seamless row coloring
        st.markdown("""
        <style>
        /* Remove gaps between columns */
        [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
        }
        /* Remove vertical gaps between rows */
        [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Check if all intervals use ceiling-based analysis
        all_ceiling_based = all(r.is_ceiling_based for r in results)

        # Different table layout for ceiling-based vs drift-based
        if all_ceiling_based:
            # Ceiling-based: #, Status, Peak, Avg VE, Terminal VE (no VE Drift, no Initial VE)
            col_widths = [0.4, 1.6, 0.8, 0.8, 0.8]
            headers = ["#", "Status", "Peak", "Avg VE", "Terminal VE"]
        else:
            # Drift-based: #, Status, VE Drift, Peak, Initial VE, Avg VE, Terminal VE
            col_widths = [0.4, 1.4, 1.0, 0.6, 0.7, 0.7, 0.8]
            headers = ["#", "Status", "VE Drift", "Peak", "Initial VE", "Avg VE", "Terminal VE"]

        # Center the table
        table_left, table_center, table_right = st.columns([0.3, 4, 0.3])
        with table_center:
            # Header - add top margin to ensure visibility
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            header_cols = st.columns(col_widths)
            for i, h in enumerate(headers):
                header_cols[i].markdown(
                    f"<div style='font-weight:600;font-size:11px;text-align:center;padding:8px 2px;border-bottom:2px solid #ccc;background:#f5f5f5;'>{h}</div>",
                    unsafe_allow_html=True
                )

            # Data rows
            for r in results:
                # Determine row color
                if r.status.value == IntervalStatus.BELOW_THRESHOLD.value:
                    row_color = "#d4edda"
                elif r.status.value == IntervalStatus.BORDERLINE.value:
                    row_color = "#fff3cd"
                else:
                    row_color = "#f8d7da"

                # Create row with container for background
                row_container = st.container()
                with row_container:
                    row_cols = st.columns(col_widths, gap="small")

                    # Column 0: Clickable button
                    with row_cols[0]:
                        is_selected = st.session_state.selected_interval == (r.interval.interval_num - 1)
                        btn_label = f"**{r.interval.interval_num}**" if is_selected else str(r.interval.interval_num)
                        if st.button(btn_label, key=f"int_{r.interval.interval_num}", use_container_width=True):
                            if is_selected:
                                st.session_state.selected_interval = None
                            else:
                                st.session_state.selected_interval = r.interval.interval_num - 1
                            st.rerun()

                    # Cell style
                    cell = f"background:{row_color};font-size:11px;text-align:center;padding:10px 4px;margin:0;"

                    if all_ceiling_based:
                        # Ceiling-based columns: Status, Peak, Avg VE, Terminal VE
                        row_cols[1].markdown(f"<div style='{cell}'>{r.status.value}</div>", unsafe_allow_html=True)
                        row_cols[2].markdown(f"<div style='{cell}'>{r.peak_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[3].markdown(f"<div style='{cell}'>{r.avg_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[4].markdown(f"<div style='{cell}'>{r.terminal_ve:.0f}</div>", unsafe_allow_html=True)
                    else:
                        # Drift-based columns: Status, VE Drift, Peak, Initial VE, Avg VE, Terminal VE
                        row_cols[1].markdown(f"<div style='{cell}'>{r.status.value}</div>", unsafe_allow_html=True)
                        row_cols[2].markdown(f"<div style='{cell}'>{r.ve_drift_pct:+.1f}% ({r.baseline_ve:.0f})</div>", unsafe_allow_html=True)
                        row_cols[3].markdown(f"<div style='{cell}'>{r.peak_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[4].markdown(f"<div style='{cell}'>{r.initial_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[5].markdown(f"<div style='{cell}'>{r.avg_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[6].markdown(f"<div style='{cell}'>{r.terminal_ve:.0f}</div>", unsafe_allow_html=True)

    else:
        # No data loaded - show instructions
        st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
                <h2 style="color: #1a1a2e;">ðŸ‘ˆ Upload a VitalPro CSV to begin</h2>
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
            - Stage 1: Rolling median filter (9 breaths) removes outliers/artifacts
            - Stage 2: 4-second bin averaging standardizes time series (~3-4 breaths/bin at 55 br/min)
            - Uses breath timestamps (not row index) for accurate timing
            
            **2. Interval Detection**
            - Auto-detects work/recovery intervals from power data
            - Finds where steady high power transitions to sustained ramp-down
            - First interval always starts at t=0
            - Intervals are always whole minutes; rest rounds to 0.5-minute increments
            
            **3. CUSUM Analysis - Two Algorithms**
            
            *Self-Calibrating CUSUM (for intervals >= 6 min):*
            - 150s kinetic blanking period
            - Self-calibrates baseline from 150-180s window
            - Expected drift as % of baseline VE per minute
            - Detects VE drifting *faster* than expected
            
            *Ceiling-Based CUSUM (for intervals < 6 min):*
            - Uses user-provided VT1/VT2 VE ceilings
            - Minimal warm-up (~20s for median filter)
            - Detects VE *exceeding* the threshold ceiling
            - Better for short intervals where drift analysis is impractical
            
            *"Use Thresholds for All Analysis" checkbox:*
            - When checked, uses ceiling-based CUSUM for ALL intervals
            - When unchecked (default), only uses ceiling-based for intervals < 6 min
            
            **4. Theil-Sen Slope Analysis**
            - Robust slope estimation on post-calibration/warmup data
            - Compares observed drift to expected drift for domain
            
            **5. Cumulative Drift (VT2 only)**
            - Tracks VE drift across all intervals
            - Uses last 60s average from each interval
            - Theil-Sen regression across elapsed workout time
            - Shows overall fatigue/drift accumulation
            
            **6. Classification**
            - **Below Threshold**: CUSUM no alarm (or alarm + recovered for ceiling-based)
            - **Borderline**: Alarm triggered but recovered
            - **Above Threshold**: CUSUM alarm sustained (not recovered)
            """)

if __name__ == "__main__":
    main()