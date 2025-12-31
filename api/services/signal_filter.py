"""
Signal Filtering Service for VT Threshold Analyzer

Implements a three-stage hybrid filtering pipeline:
1. Rolling median filter (breath domain) - removes single-breath outliers
2. Time binning (time domain) - standardizes accumulation rate
3. Hampel filter (time domain) - removes consecutive outlier clusters
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import median_filter

from ..models.params import AnalysisParams


def apply_median_filter(ve_raw: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Stage 1: Apply rolling median filter in breath domain.

    Removes non-physiological spikes (coughs, sensor errors).
    A 9-breath window at ~55 br/min covers ~10 seconds, providing
    robust outlier rejection while preserving physiological trends.

    Args:
        ve_raw: Raw VE values (L/min)
        window: Filter window size in breaths

    Returns:
        Median-filtered VE values
    """
    if len(ve_raw) < window:
        return ve_raw.copy()

    ve_median = median_filter(ve_raw, size=window, mode='nearest')
    return ve_median


def apply_time_binning(
    ve_clean: np.ndarray,
    breath_times: np.ndarray,
    bin_size: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Convert irregular breath data to uniform time series via bin averaging.

    Uses bin-start timestamp convention. Empty bins are filled via linear interpolation.
    A 4-second bin at ~55 br/min captures ~3-4 breaths per bin,
    providing reliable averaging while maintaining temporal resolution.

    Args:
        ve_clean: Cleaned VE values (after median filter)
        breath_times: Timestamps for each breath
        bin_size: Bin size in seconds

    Returns:
        Tuple of (bin_times, bin_values)
        - bin_times: Array of bin start timestamps
        - bin_values: Array of averaged VE values per bin
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


def apply_hampel_filter(
    bin_times: np.ndarray,
    bin_values: np.ndarray,
    window_sec: float = 30.0,
    n_sigma: float = 3.0
) -> np.ndarray:
    """
    Stage 3: Hampel filter for removing consecutive outlier clusters from binned data.

    Uses a time-based window (default 30 seconds) centered on each bin.
    Bins deviating more than n_sigma MAD-scaled standard deviations from
    the window median are replaced with the window median.

    This handles cases where the median filter fails due to consecutive
    outliers (e.g., 5+ spike values in a row that dominate the median window).

    Args:
        bin_times: Uniformly-spaced bin timestamps
        bin_values: Binned VE values (already median-filtered and binned)
        window_sec: Window size in seconds (total, centered on each point)
        n_sigma: Number of scaled MAD units for outlier threshold

    Returns:
        VE values with outliers replaced by window median
    """
    if len(bin_values) < 3:
        return bin_values.copy()

    ve_cleaned = bin_values.copy()
    half_window = window_sec / 2.0

    for i in range(len(bin_values)):
        t = bin_times[i]
        v = bin_values[i]

        # Get all bins within the time window
        window_mask = (bin_times >= t - half_window) & (bin_times <= t + half_window)
        window_ve = bin_values[window_mask]

        if len(window_ve) < 3:
            continue  # Not enough points for robust statistics

        # Calculate median and MAD (Median Absolute Deviation)
        window_median = np.median(window_ve)
        window_mad = np.median(np.abs(window_ve - window_median))
        window_mad_scaled = 1.4826 * window_mad  # Scale factor for normal distribution

        if window_mad_scaled < 1e-6:
            continue  # Avoid division by zero (all values identical)

        # Calculate deviation in sigma units
        deviation = abs(v - window_median)
        sigma_dev = deviation / window_mad_scaled

        # Replace outliers with window median
        if sigma_dev > n_sigma:
            ve_cleaned[i] = window_median

    return ve_cleaned


def apply_hybrid_filtering(
    ve_raw: np.ndarray,
    breath_times: np.ndarray,
    params: AnalysisParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply three-stage hybrid filtering pipeline.

    Stage 1: Rolling median (breath domain) - removes single-breath outliers/artifacts
    Stage 2: Time binning (time domain) - standardizes accumulation rate
    Stage 3: Hampel filter (time domain) - removes consecutive outlier clusters

    Args:
        ve_raw: Raw VE values
        breath_times: Breath timestamps
        params: Analysis parameters containing filter settings

    Returns:
        Tuple of (ve_median, bin_times, ve_binned)
        - ve_median: Median-filtered breath values (for visualization dots)
        - bin_times: Bin start timestamps
        - ve_binned: Binned VE values after Hampel filtering (for CUSUM analysis)
    """
    # Stage 1: Median filter (removes single-breath spikes)
    ve_median = apply_median_filter(ve_raw, params.median_window)

    # Stage 2: Time binning
    bin_times, ve_binned = apply_time_binning(ve_median, breath_times, params.bin_size)

    # Stage 3: Hampel filter (removes consecutive outlier clusters)
    ve_binned = apply_hampel_filter(
        bin_times,
        ve_binned,
        params.hampel_window_sec,
        params.hampel_n_sigma
    )

    return ve_median, bin_times, ve_binned
