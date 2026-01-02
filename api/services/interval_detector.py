"""
Interval Detection Service for VT Threshold Analyzer

Detects work intervals and recovery periods from power data using:
- K-Means clustering for work/rest separation
- Plateau counting and duration analysis
- Validation against expected workout structure
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..models.enums import RunType
from ..models.schemas import Interval


def detect_intervals(
    power_df: pd.DataFrame,
    breath_df: pd.DataFrame
) -> Tuple[RunType, int, float, float]:
    """
    Detect run format and intervals from power data using plateau counting and validation.

    Key assumptions:
    - First interval ALWAYS starts at t=0
    - Intervals are ALWAYS whole minutes (1m, 2m, 3m, 4m, etc.)
    - Rest periods are in 0.5m increments (0.5m, 1m, 1.5m, etc.)
    - Recording starts when first interval begins
    - Recording ends after last recovery
    - Equal number of intervals and recoveries
    - Each cycle follows inverted parabola pattern: ramp-up -> plateau -> ramp-down -> trough

    Args:
        power_df: DataFrame with 'time' and 'power' columns
        breath_df: DataFrame with 'breath_time' column

    Returns:
        Tuple of (run_type, num_intervals, interval_duration_min, recovery_duration_min)
    """
    if power_df.empty or len(power_df) < 10:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.MODERATE, 1, total_duration, 0.0

    power = power_df['power'].values
    time = power_df['time'].values

    # Skip NaN and negative values (negative values are calibration artifacts)
    valid_mask = ~np.isnan(power) & (power >= 0)
    if np.sum(valid_mask) < 100:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.MODERATE, 1, total_duration, 0.0

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
        return RunType.MODERATE, 1, total_duration, 0.0

    # Smooth power data with rolling median (10-second window) to reduce noise
    window_size = min(10, len(valid_power) // 10)
    if window_size < 3:
        window_size = 3
    power_series = pd.Series(valid_power)
    power_smooth = power_series.rolling(window=window_size, center=True).median()
    power_smooth = power_smooth.bfill().ffill().values

    # Calculate threshold using K-Means clustering to find work/rest separation
    X = power_smooth.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(X)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = (cluster_centers[0] + cluster_centers[1]) / 2

    # Classify each point as high (1) or low (0)
    is_high = (power_smooth > threshold).astype(int)

    # Find transitions (changes between high and low)
    transitions = np.diff(is_high)
    low_to_high_indices = np.where(transitions == 1)[0] + 1
    high_to_low_indices = np.where(transitions == -1)[0] + 1

    # If no clear transitions found, treat as steady state
    if len(high_to_low_indices) < 1:
        total_duration = breath_df['breath_time'].max() / 60.0
        return RunType.MODERATE, 1, total_duration, 0.0

    # Count plateaus and measure durations
    num_high_plateaus = len(high_to_low_indices)
    starts_high = is_high[0] == 1

    # Measure plateau durations
    RAMP_ADJUSTMENT_SEC = 10

    interval_plateau_durations = []
    recovery_plateau_durations = []

    # First interval: from t=0 to first high-to-low transition
    if len(high_to_low_indices) > 0 and starts_high:
        first_plateau_end = valid_time[high_to_low_indices[0]]
        interval_plateau_durations.append(first_plateau_end)

    # Measure subsequent plateaus
    for i in range(len(high_to_low_indices)):
        recovery_start_idx = high_to_low_indices[i]
        recovery_start_time = valid_time[recovery_start_idx]

        next_interval_starts = low_to_high_indices[low_to_high_indices > recovery_start_idx]

        if len(next_interval_starts) > 0:
            recovery_end_idx = next_interval_starts[0]
            recovery_end_time = valid_time[recovery_end_idx]
            recovery_plateau_dur = recovery_end_time - recovery_start_time
            recovery_plateau_durations.append(recovery_plateau_dur)

            next_recovery_starts = high_to_low_indices[high_to_low_indices > recovery_end_idx]
            if len(next_recovery_starts) > 0:
                interval_end_idx = next_recovery_starts[0]
                interval_end_time = valid_time[interval_end_idx]
                interval_plateau_dur = interval_end_time - recovery_end_time
                interval_plateau_durations.append(interval_plateau_dur)
        else:
            final_recovery_dur = valid_time[-1] - recovery_start_time
            if final_recovery_dur > 15:
                recovery_plateau_durations.append(final_recovery_dur)

    # Calculate median plateau durations and add ramp adjustment
    if len(interval_plateau_durations) > 0:
        median_interval_plateau_sec = np.median(interval_plateau_durations)
        interval_duration_estimate = (median_interval_plateau_sec + RAMP_ADJUSTMENT_SEC) / 60.0
        interval_duration = round(interval_duration_estimate)
        if interval_duration < 1:
            interval_duration = 1
    else:
        interval_duration = 4

    if len(recovery_plateau_durations) > 0:
        median_recovery_plateau_sec = np.median(recovery_plateau_durations)
        recovery_duration_estimate = (median_recovery_plateau_sec + RAMP_ADJUSTMENT_SEC) / 60.0
        rest_duration = round(recovery_duration_estimate * 2) / 2
        if rest_duration < 0.5:
            rest_duration = 0.5
    else:
        rest_duration = 1.0

    # Validate against total duration and find best fit
    total_duration_min = breath_df['breath_time'].max() / 60.0

    candidate_recoveries = [
        rest_duration - 0.5,
        rest_duration,
        rest_duration + 0.5
    ]
    candidate_recoveries = [r for r in candidate_recoveries if r >= 0.5]

    best_fit_recovery = rest_duration
    best_fit_error = float('inf')
    best_fit_n = 0

    for candidate_recovery in candidate_recoveries:
        period = interval_duration + candidate_recovery
        n_calculated = total_duration_min / period
        n_rounded = round(n_calculated)

        if n_rounded < 1:
            continue

        expected_total = n_rounded * period
        error = abs(expected_total - total_duration_min)
        n_error = abs(n_calculated - n_rounded)

        if n_error < 0.2 and error < best_fit_error:
            best_fit_error = error
            best_fit_recovery = candidate_recovery
            best_fit_n = n_rounded

    rest_duration = best_fit_recovery
    num_intervals = best_fit_n if best_fit_n > 0 else round(total_duration_min / (interval_duration + rest_duration))

    num_intervals = max(1, num_intervals)
    max_possible = int(total_duration_min / interval_duration) + 1
    num_intervals = min(num_intervals, max_possible)

    # Validate plateau alignment
    def find_plateau_centers(is_high_arr, valid_time_arr):
        high_plateau_centers = []
        low_plateau_centers = []
        in_high = is_high_arr[0] == 1
        segment_start = 0

        for i in range(1, len(is_high_arr)):
            if is_high_arr[i] != is_high_arr[i-1]:
                segment_end = i - 1
                segment_center_time = (valid_time_arr[segment_start] + valid_time_arr[segment_end]) / 2
                if in_high:
                    high_plateau_centers.append(segment_center_time)
                else:
                    low_plateau_centers.append(segment_center_time)
                segment_start = i
                in_high = is_high_arr[i] == 1

        segment_end = len(is_high_arr) - 1
        segment_center_time = (valid_time_arr[segment_start] + valid_time_arr[segment_end]) / 2
        if in_high:
            high_plateau_centers.append(segment_center_time)
        else:
            low_plateau_centers.append(segment_center_time)

        return high_plateau_centers, low_plateau_centers

    def validate_alignment(n_intervals, int_dur, rec_dur, high_centers, low_centers, tolerance_sec=30):
        period = (int_dur + rec_dur) * 60
        int_dur_sec = int_dur * 60
        total_error = 0
        violations = 0

        for i in range(n_intervals):
            int_start = i * period
            int_end = int_start + int_dur_sec
            rec_start = int_end
            rec_end = (i + 1) * period

            high_in_interval = [c for c in high_centers if int_start <= c <= int_end]
            if len(high_in_interval) == 0:
                distances = [min(abs(c - int_start), abs(c - int_end)) for c in high_centers]
                if distances:
                    total_error += min(distances)
                violations += 1

            low_in_interval = [c for c in low_centers if int_start + tolerance_sec <= c <= int_end - tolerance_sec]
            if len(low_in_interval) > 0:
                violations += len(low_in_interval)

            low_in_recovery = [c for c in low_centers if rec_start <= c <= rec_end]
            if len(low_in_recovery) == 0:
                distances = [min(abs(c - rec_start), abs(c - rec_end)) for c in low_centers]
                if distances:
                    total_error += min(distances)
                violations += 1

            high_in_recovery = [c for c in high_centers if rec_start + tolerance_sec <= c <= rec_end - tolerance_sec]
            if len(high_in_recovery) > 0:
                violations += len(high_in_recovery)

        is_valid = violations == 0
        return is_valid, violations, total_error

    high_plateau_centers, low_plateau_centers = find_plateau_centers(is_high, valid_time)

    is_valid, violations, alignment_error = validate_alignment(
        num_intervals, interval_duration, rest_duration,
        high_plateau_centers, low_plateau_centers
    )

    if not is_valid:
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

                if n_rounded < 1 or abs(n_calc - n_rounded) > 0.15:
                    continue

                is_valid_cand, viol_cand, err_cand = validate_alignment(
                    n_rounded, cand_int, cand_rec,
                    high_plateau_centers, low_plateau_centers
                )

                if viol_cand < best_violations or (viol_cand == best_violations and err_cand < best_alignment_error):
                    best_violations = viol_cand
                    best_alignment_error = err_cand
                    best_config = (n_rounded, cand_int, cand_rec)

        num_intervals, interval_duration, rest_duration = best_config

    # Single interval = VT1, Multiple intervals = VT2
    if num_intervals == 1:
        return RunType.MODERATE, 1, float(interval_duration), 0.0
    else:
        return RunType.HEAVY, num_intervals, float(interval_duration), float(rest_duration)


def create_intervals_from_params(
    breath_df: pd.DataFrame,
    run_type: RunType,
    num_intervals: int,
    interval_duration: float,
    recovery_duration: float
) -> List[Interval]:
    """
    Create interval objects based on detected/specified parameters.

    Args:
        breath_df: DataFrame with breath-by-breath data
        run_type: Type of run (VT1 or VT2)
        num_intervals: Number of intervals
        interval_duration: Interval duration in minutes
        recovery_duration: Recovery duration in minutes

    Returns:
        List of Interval objects
    """
    intervals = []
    breath_times = breath_df['breath_time'].values

    if run_type == RunType.MODERATE:
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

        breath_start_idx = int(np.searchsorted(breath_times, start_time))
        breath_end_idx = int(np.searchsorted(breath_times, end_time))

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
