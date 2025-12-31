"""
VT Threshold Analyzer - Desktop Application
Analyzes Tymewear VitalPro respiratory data to assess VT1/VT2 compliance
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
from streamlit_js_eval import streamlit_js_eval
from pathlib import Path
import json
from io import StringIO

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

class RunType(Enum):
    VT1_STEADY = "VT1 (Steady State)"
    VT2_INTERVAL = "VT2 (Intervals)"

class IntervalStatus(Enum):
    BELOW_THRESHOLD = "Below Threshold"
    BORDERLINE = "Borderline"
    ABOVE_THRESHOLD = "Above Threshold"

@dataclass
class AnalysisParams:
    # Ramp-up period (blanking) parameters
    phase3_onset_override: Optional[float] = None  # User override for ramp-up period end (seconds)
    phase3_min_time: float = 90.0  # Minimum time for Phase III detection for VT2 (seconds)
    phase3_max_time: float = 180.0  # Maximum time for Phase III detection for VT2 (seconds)
    phase3_default: float = 150.0  # Default Phase III onset if detection fails (seconds)
    # VT1-specific: blanking always at 6 min, calibration depends on run length
    vt1_blanking_time: float = 360.0  # 6 minutes in seconds - fixed for VT1
    vt1_calibration_short: float = 60.0  # 1 min calibration for runs < 15 min
    vt1_calibration_long: float = 240.0  # 4 min calibration for runs >= 15 min
    # VT2-specific: calibration after Phase III onset
    vt2_calibration_duration: float = 60.0  # 1 minute calibration for VT2
    calibration_duration: float = 30.0  # seconds - legacy, used as fallback
    h_multiplier_vt1: float = 5.0
    h_multiplier_vt2: float = 5.0
    slack_multiplier: float = 0.5  # Hidden - not exposed in UI
    # Expected drift as percentage of baseline VE per minute (for CUSUM)
    expected_drift_pct_vt1: float = 0.3  # % per minute (minimal in moderate domain)
    expected_drift_pct_vt2: float = 1.0   # % per minute (slow component in heavy domain)
    # Max drift thresholds for slope classification
    max_drift_pct_vt1: float = 1.0  # % per minute - slope threshold for VT1 classification
    max_drift_pct_vt2: float = 3.0  # % per minute - slope threshold for VT2 classification
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
    ve_drift_rate: float  # Slope (L/min per minute)
    ve_drift_pct: float   # Drift as % of baseline per minute
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
    # Slope line data (from regression)
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
    # Segmented regression fields
    is_segmented: bool = False                    # Flag for segmented regression analysis
    phase3_onset_time: Optional[float] = None     # Detected Phase III onset (absolute time)
    phase3_onset_rel: Optional[float] = None      # Phase III onset relative to interval start (seconds)
    segment1_times: Optional[np.ndarray] = None   # Times for Phase II segment line
    segment1_ve: Optional[np.ndarray] = None      # VE values for Phase II segment line
    segment2_times: Optional[np.ndarray] = None   # Times for segment 2 (Phase III onset to 2nd hinge)
    segment2_ve: Optional[np.ndarray] = None      # VE values for segment 2
    segment3_times: Optional[np.ndarray] = None   # Times for segment 3 (2nd hinge to end)
    segment3_ve: Optional[np.ndarray] = None      # VE values for segment 3
    # Second hinge fields (slope change detection after Phase III)
    hinge2_time: Optional[float] = None           # 2nd hinge point (absolute time)
    hinge2_time_rel: Optional[float] = None       # 2nd hinge point relative to interval start (seconds)
    slope1_pct: Optional[float] = None            # Slope before 2nd hinge (% of baseline per minute)
    slope2_pct: Optional[float] = None            # Slope after 2nd hinge (% of baseline per minute)
    split_slope_ratio: Optional[float] = None     # Ratio: slope2 / slope1 (for classification)
    hinge2_detected: bool = False                 # True if 2nd hinge was successfully detected
    # Speed for this interval (from iOS CSV)
    speed: Optional[float] = None

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
# CLOUD SESSION HELPERS
# ============================================================================

import requests

# API URL - change this to your Cloud Run URL after deployment
# For local development: http://localhost:8000
# For production: https://your-service-name-xxxxx-uc.a.run.app
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def list_cloud_sessions() -> list:
    """
    List all uploaded sessions from the API.
    Returns list of dicts with session_id, filename, uploaded_at.
    """
    try:
        response = requests.get(f"{API_URL}/api/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def get_cloud_session_content(session_id: str) -> Optional[str]:
    """
    Get CSV content for a specific session from the API.
    """
    try:
        response = requests.get(f"{API_URL}/api/sessions/{session_id}", timeout=10)
        if response.status_code == 200:
            return response.json().get("csv_content")
        return None
    except Exception:
        return None


class CloudSessionFile:
    """Wrapper to make a cloud session look like an uploaded file."""

    def __init__(self, session_id: str, filename: str, content: str):
        self.name = filename
        self.size = len(content)
        self._content = content.encode('utf-8')
        self._position = 0

    def getvalue(self) -> bytes:
        return self._content

    def seek(self, position: int):
        self._position = position

    def read(self) -> bytes:
        return self._content[self._position:]


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


def detect_csv_format(uploaded_file) -> str:
    """Detect whether the CSV is iOS app format or VitalPro format."""
    content = uploaded_file.getvalue().decode('utf-8')
    uploaded_file.seek(0)  # Reset for later parsing

    # iOS format starts with # comments containing metadata
    if content.startswith('# Date:') or '# Run Type:' in content[:500]:
        return 'ios'
    # VitalPro format has 'Breath by breath time' column
    elif 'Breath by breath time' in content:
        return 'vitalpro'
    else:
        return 'unknown'


def parse_ios_csv(uploaded_file) -> Tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """
    Parse iOS app CSV format and extract breath-by-breath data with metadata.

    iOS format has:
    - Header comments starting with # containing metadata
    - Data columns: timestamp, elapsed_sec, VE, HR, [phase, speed]

    Returns:
        breath_df: DataFrame with breath-by-breath data
        metadata: Dict with parsed metadata
        power_df: Empty DataFrame (no power data in iOS format)
        run_params: Dict with run parameters (run_type, intervals, durations, speeds)
    """
    content = uploaded_file.getvalue().decode('utf-8')
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # Parse header metadata
    metadata = {}
    run_params = {}
    header_end_idx = 0

    for i, line in enumerate(lines):
        if line.startswith('#'):
            # Parse metadata line: # Key: Value
            if ':' in line:
                key_value = line[1:].strip()  # Remove leading #
                colon_idx = key_value.find(':')
                if colon_idx > 0:
                    key = key_value[:colon_idx].strip().lower()
                    value = key_value[colon_idx + 1:].strip()
                    metadata[key] = value

                    # Parse specific run parameters
                    if key == 'run type':
                        run_params['run_type'] = RunType.VT1_STEADY if value.lower() == 'vt1' else RunType.VT2_INTERVAL
                    elif key == 'speed':
                        # Speed can be single value or comma-separated for intervals
                        speeds = [float(s.strip().replace(' mph', '')) for s in value.replace(' mph', '').split(',')]
                        run_params['speeds'] = speeds
                    elif key == 'vt1 threshold':
                        run_params['vt1_threshold'] = float(value.replace(' L/min', ''))
                    elif key == 'vt2 threshold':
                        run_params['vt2_threshold'] = float(value.replace(' L/min', ''))
                    elif key == 'intervals':
                        run_params['num_intervals'] = int(value)
                    elif key == 'interval duration':
                        run_params['interval_duration'] = float(value.replace(' min', ''))
                    elif key == 'recovery duration':
                        run_params['recovery_duration'] = float(value.replace(' min', ''))
                    elif key == 'phase duration':
                        run_params['phase_duration'] = float(value.replace(' min', ''))
            header_end_idx = i + 1
        elif line.strip() and not line.startswith('#'):
            # Found data header row
            break

    # Parse data using pandas
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, comment='#', skip_blank_lines=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Map iOS columns to internal format
    result = pd.DataFrame()

    if 'elapsed_sec' in df.columns:
        result['breath_time'] = pd.to_numeric(df['elapsed_sec'], errors='coerce')
    elif 'elapsed' in df.columns:
        result['breath_time'] = pd.to_numeric(df['elapsed'], errors='coerce')

    if 've' in df.columns:
        result['ve_raw'] = pd.to_numeric(df['ve'], errors='coerce')

    if 'hr' in df.columns:
        result['hr'] = pd.to_numeric(df['hr'], errors='coerce')

    # Speed column (per-breath, can vary per interval)
    if 'speed' in df.columns:
        result['speed'] = pd.to_numeric(df['speed'], errors='coerce')

    # Phase column (workout, recovery, etc.)
    if 'phase' in df.columns:
        result['phase'] = df['phase']

    # Drop rows where breath_time is NaN
    result = result.dropna(subset=['breath_time']).reset_index(drop=True)

    # Empty power_df since iOS format doesn't have power data
    power_df = pd.DataFrame()

    # Set defaults for VT1 runs if not specified
    if 'run_type' not in run_params:
        run_params['run_type'] = RunType.VT1_STEADY

    if run_params['run_type'] == RunType.VT1_STEADY:
        run_params['num_intervals'] = 1
        if 'phase_duration' in run_params:
            run_params['interval_duration'] = run_params['phase_duration']
        elif 'interval_duration' not in run_params:
            # Calculate from data
            run_params['interval_duration'] = result['breath_time'].max() / 60.0
        run_params['recovery_duration'] = 0.0

    return result, metadata, power_df, run_params


def parse_csv_auto(uploaded_file) -> Tuple[pd.DataFrame, dict, pd.DataFrame, Optional[dict]]:
    """
    Auto-detect CSV format and parse accordingly.

    Returns:
        breath_df: DataFrame with breath-by-breath data
        metadata: Dict with file metadata
        power_df: DataFrame with power data (empty for iOS format)
        run_params: Dict with run parameters (only for iOS format, None for VitalPro)
    """
    csv_format = detect_csv_format(uploaded_file)

    if csv_format == 'ios':
        breath_df, metadata, power_df, run_params = parse_ios_csv(uploaded_file)
        return breath_df, metadata, power_df, run_params
    else:
        # VitalPro format
        breath_df, metadata, power_df = parse_vitalpro_csv(uploaded_file)
        return breath_df, metadata, power_df, None


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

def fit_single_slope(t: np.ndarray, ve: np.ndarray) -> Tuple[float, float]:
    """
    Fit a single linear slope using robust Huber regression.

    Used for:
    - Phase III slope estimation after calibration window
    - Cumulative drift across intervals

    Returns: (slope, intercept)
    """

    def huber_loss(params, t_arr, ve_arr, delta=5.0):
        """Huber loss for robust linear regression."""
        intercept, slope = params
        pred = intercept + slope * t_arr
        residuals = ve_arr - pred
        abs_res = np.abs(residuals)
        loss = np.where(abs_res <= delta,
                        0.5 * residuals**2,
                        delta * (abs_res - 0.5 * delta))
        return np.sum(loss)

    if len(t) < 2:
        return 0.0, np.mean(ve) if len(ve) > 0 else 0.0

    # Initial guess from simple linear fit
    slope_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
    intercept_init = ve[0] - slope_init * t[0]

    try:
        result = minimize(
            huber_loss,
            [intercept_init, slope_init],
            args=(t, ve),
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        intercept, slope = result.x
    except Exception:
        slope = slope_init
        intercept = intercept_init

    return slope, intercept


def fit_robust_hinge(t: np.ndarray, ve: np.ndarray,
                     params: AnalysisParams) -> Tuple[float, float, float, float, float, bool]:
    """
    Fit a robust hinge model to detect Phase III onset.

    Model: VE(t) = β₀ + β₁*t + β₂*max(0, t - τ)

    Where:
    - τ (tau): The Phase III onset breakpoint
    - β₁: Slope of Phase II (initial ramp)
    - β₁ + β₂: Slope of Phase III (drift)

    Uses Huber loss for robustness to outliers.
    Constrains breakpoint detection to params.phase3_min_time - params.phase3_max_time.
    If detection fails or breakpoint is at bounds, returns default Phase III onset.

    Returns: (tau, beta0, beta1, beta2, loss, detection_succeeded)
    """

    def hinge_model(t_arr, tau, b0, b1, b2):
        """Piecewise linear hinge function."""
        return b0 + b1 * t_arr + b2 * np.maximum(0, t_arr - tau)

    def huber_loss(params_opt, t_arr, ve_arr, delta=5.0):
        """Huber loss: quadratic for small errors, linear for large (robust to outliers)."""
        tau, b0, b1, b2 = params_opt
        pred = hinge_model(t_arr, tau, b0, b1, b2)
        residuals = ve_arr - pred
        # Huber loss
        abs_res = np.abs(residuals)
        loss = np.where(abs_res <= delta,
                        0.5 * residuals**2,
                        delta * (abs_res - 0.5 * delta))
        return np.sum(loss)

    # Use user override if provided
    if params.phase3_onset_override is not None:
        tau = params.phase3_onset_override
        # Fit linear model for Phase II (before tau) and Phase III (after tau)
        mask_before = t < tau
        mask_after = t >= tau

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            # Fit Phase II slope
            t_before = t[mask_before]
            ve_before = ve[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            # Fit Phase III slope change
            t_after = t[mask_after]
            ve_after = ve[mask_after]
            phase3_slope = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = phase3_slope - b1
        else:
            b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
            b0 = ve[0] - b1 * t[0]
            b2 = 0

        final_loss = huber_loss([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, True  # User override counts as success

    # Set tau bounds based on params (90-240 seconds constraint)
    tau_min = params.phase3_min_time
    tau_max = params.phase3_max_time

    # Ensure bounds are within data range
    t_min = t[0]
    t_max = t[-1]
    tau_min = max(tau_min, t_min + 10)  # At least 10s into the interval
    tau_max = min(tau_max, t_max - 30)  # At least 30s before end (for calibration)

    # Check if we have enough data for constrained detection
    if tau_max <= tau_min:
        # Not enough data range - use default
        tau = params.phase3_default
        b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0 = ve[0] - b1 * t[0]
        b2 = 0
        final_loss = huber_loss([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False

    # Initial guess: tau at midpoint of allowed range
    tau_init = (tau_min + tau_max) / 2

    # Simple linear fit for initial b0, b1
    if len(t) >= 2:
        b1_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0_init = ve[0] - b1_init * t[0]
    else:
        b0_init = np.mean(ve)
        b1_init = 0
    b2_init = 0  # Initially assume no slope change

    initial_params = [tau_init, b0_init, b1_init, b2_init]

    # Optimize with bounds on tau
    bounds = [(tau_min, tau_max),  # tau constrained to 90-240s
              (None, None),  # b0
              (None, None),  # b1 (Phase II slope)
              (None, None)]  # b2 (slope change)

    detection_succeeded = True

    try:
        result = minimize(
            huber_loss,
            initial_params,
            args=(t, ve),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        tau, b0, b1, b2 = result.x
        final_loss = result.fun

        # Check if tau is at the bounds (indicates detection failed)
        if abs(tau - tau_min) < 1.0 or abs(tau - tau_max) < 1.0:
            # Breakpoint at boundary - detection likely failed
            detection_succeeded = False
            tau = params.phase3_default
    except Exception:
        # Optimization failed - use default
        detection_succeeded = False
        tau = params.phase3_default
        b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0 = ve[0] - b1 * t[0]
        b2 = 0
        final_loss = huber_loss([tau, b0, b1, b2], t, ve)

    return tau, b0, b1, b2, final_loss, detection_succeeded


def fit_second_hinge(t: np.ndarray, ve: np.ndarray,
                     phase3_onset: float, interval_end: float) -> Tuple[float, float, float, float, float, bool]:
    """
    Fit a second robust hinge model to detect slope change after Phase III onset.

    Model: VE(t) = β₀ + β₁*t + β₂*max(0, t - τ)

    Constraint window: (Phase III onset + 120s) to (interval end - 120s)
    If window is invalid, places hinge at midpoint between Phase III onset and interval end.
    If detection fails, falls back to midpoint of constraint window.

    Returns: (tau2, beta0, beta1, beta2, loss, detection_succeeded)
    """

    def hinge_model(t_arr, tau, b0, b1, b2):
        """Piecewise linear hinge function."""
        return b0 + b1 * t_arr + b2 * np.maximum(0, t_arr - tau)

    def huber_loss(params_opt, t_arr, ve_arr, delta=5.0):
        """Huber loss: quadratic for small errors, linear for large (robust to outliers)."""
        tau, b0, b1, b2 = params_opt
        pred = hinge_model(t_arr, tau, b0, b1, b2)
        residuals = ve_arr - pred
        abs_res = np.abs(residuals)
        loss = np.where(abs_res <= delta,
                        0.5 * residuals**2,
                        delta * (abs_res - 0.5 * delta))
        return np.sum(loss)

    # Define constraint window: 2 min after Phase III onset to 2 min before interval end
    tau_min = phase3_onset + 120.0  # 2 minutes after Phase III onset
    tau_max = interval_end - 120.0   # 2 minutes before interval end

    # Calculate midpoint between Phase III onset and interval end (fallback position)
    midpoint_fallback = (phase3_onset + interval_end) / 2.0

    # Check if constraint window is valid
    if tau_max <= tau_min:
        # Invalid window - use midpoint between Phase III onset and interval end
        tau = midpoint_fallback
        # Fit coefficients for this fixed tau
        mask_before = t < tau
        mask_after = t >= tau

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            t_before = t[mask_before]
            ve_before = ve[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            t_after = t[mask_after]
            ve_after = ve[mask_after]
            slope_after = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = slope_after - b1
        else:
            b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
            b0 = ve[0] - b1 * t[0]
            b2 = 0

        final_loss = huber_loss([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False  # Window invalid, used fallback

    # Constraint window midpoint (for fallback if detection fails)
    window_midpoint = (tau_min + tau_max) / 2.0

    # Ensure bounds are within data range
    t_min = t[0]
    t_max = t[-1]
    tau_min = max(tau_min, t_min + 10)
    tau_max = min(tau_max, t_max - 10)

    if tau_max <= tau_min:
        # Not enough data in window - use window midpoint
        tau = window_midpoint
        mask_before = t < tau
        mask_after = t >= tau

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            t_before = t[mask_before]
            ve_before = ve[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            t_after = t[mask_after]
            ve_after = ve[mask_after]
            slope_after = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = slope_after - b1
        else:
            b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
            b0 = ve[0] - b1 * t[0]
            b2 = 0

        final_loss = huber_loss([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False

    # Initial guess: tau at midpoint of constraint window
    tau_init = (tau_min + tau_max) / 2

    # Simple linear fit for initial b0, b1
    if len(t) >= 2:
        b1_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0_init = ve[0] - b1_init * t[0]
    else:
        b0_init = np.mean(ve)
        b1_init = 0
    b2_init = 0

    initial_params = [tau_init, b0_init, b1_init, b2_init]

    # Optimize with bounds on tau
    bounds = [(tau_min, tau_max),
              (None, None),
              (None, None),
              (None, None)]

    detection_succeeded = True

    try:
        result = minimize(
            huber_loss,
            initial_params,
            args=(t, ve),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        tau, b0, b1, b2 = result.x
        final_loss = result.fun

        # Check if tau is at the bounds (indicates detection failed)
        if abs(tau - tau_min) < 1.0 or abs(tau - tau_max) < 1.0:
            detection_succeeded = False
            tau = window_midpoint
            # Recompute coefficients for fallback tau
            mask_before = t < tau
            mask_after = t >= tau

            if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
                t_before = t[mask_before]
                ve_before = ve[mask_before]
                b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
                b0 = ve_before[0] - b1 * t_before[0]

                t_after = t[mask_after]
                ve_after = ve[mask_after]
                slope_after = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
                b2 = slope_after - b1
            final_loss = huber_loss([tau, b0, b1, b2], t, ve)

    except Exception:
        detection_succeeded = False
        tau = window_midpoint
        mask_before = t < tau
        mask_after = t >= tau

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            t_before = t[mask_before]
            ve_before = ve[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            t_after = t[mask_after]
            ve_after = ve[mask_after]
            slope_after = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = slope_after - b1
        else:
            b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
            b0 = ve[0] - b1 * t[0]
            b2 = 0
        final_loss = huber_loss([tau, b0, b1, b2], t, ve)

    return tau, b0, b1, b2, final_loss, detection_succeeded


def analyze_interval_segmented(breath_df: pd.DataFrame, interval: Interval,
                               params: AnalysisParams, run_type: RunType,
                               speed: Optional[float] = None) -> IntervalResult:
    """
    Perform CUSUM analysis using robust hinge model to detect Phase III onset.

    Uses a piecewise linear "hinge" model with Huber loss to find the breakpoint
    where Phase II (on-kinetics) transitions to Phase III (steady state/drift).

    Calibration window is then set to [breakpoint, breakpoint + calibration_duration].

    Phase III detection is constrained to 90-240 seconds. If detection fails,
    falls back to 150 seconds default.
    """

    # Extract interval data
    idx_start = interval.start_idx
    idx_end = interval.end_idx

    if idx_end <= idx_start:
        # Invalid interval - return default result
        return IntervalResult(
            interval=interval,
            status=IntervalStatus.BELOW_THRESHOLD,
            ve_drift_rate=0,
            ve_drift_pct=0,
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
            is_ceiling_based=False,
            is_segmented=True,
            phase3_onset_time=None,
            phase3_onset_rel=None,
            speed=speed
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

    # =========================================================================
    # RAMP-UP PERIOD (BLANKING) AND PHASE III ONSET
    # VT1: Fixed at 6 minutes, VT2: Auto-detected (constrained to 90-180s)
    # =========================================================================

    # Calculate interval duration for VT1 calibration window determination
    interval_duration_sec = interval.end_time - interval.start_time
    interval_duration_min = interval_duration_sec / 60.0

    if run_type == RunType.VT1_STEADY:
        # VT1: Fixed blanking at 6 minutes (no hinge detection)
        phase3_onset_rel = params.vt1_blanking_time  # 360 seconds = 6 minutes
        phase3_onset_time = phase3_onset_rel + breath_times_raw[0]
        detection_succeeded = True  # Fixed value, always "succeeds"

        # Fit a simple linear model for segment 1 (ramp-up visualization)
        ramp_mask = bin_times_rel <= phase3_onset_rel
        if np.sum(ramp_mask) >= 2:
            ramp_times = bin_times_rel[ramp_mask]
            ramp_ve = ve_binned[ramp_mask]
            b1 = (ramp_ve[-1] - ramp_ve[0]) / (ramp_times[-1] - ramp_times[0]) if (ramp_times[-1] - ramp_times[0]) > 0 else 0
            b0 = ramp_ve[0] - b1 * ramp_times[0]
        else:
            b0 = np.mean(ve_binned) if len(ve_binned) > 0 else 0
            b1 = 0
        b2 = 0  # No slope change for segment 1
    else:
        # VT2: Use robust hinge model to detect Phase III onset (constrained to 90-180s)
        tau, b0, b1, b2, _, detection_succeeded = fit_robust_hinge(bin_times_rel, ve_binned, params)
        phase3_onset_rel = tau  # In seconds relative to interval start
        phase3_onset_time = tau + breath_times_raw[0]  # Absolute time

    # Generate segment 1 line for visualization (Ramp-up: start to Phase III onset)
    seg1_t_start = bin_times_rel[0]
    seg1_t_end = phase3_onset_rel
    segment1_times_rel = np.array([seg1_t_start, seg1_t_end])
    segment1_ve = b0 + b1 * segment1_times_rel  # Ramp-up line

    # Segments 2 and 3 will be generated after 2nd hinge detection below
    segment2_times_rel = None
    segment2_ve = None
    segment3_times_rel = None
    segment3_ve = None

    # =========================================================================
    # Set calibration window based on run type and duration
    # =========================================================================

    if run_type == RunType.VT1_STEADY:
        # VT1: Calibration window depends on run duration
        cal_start = phase3_onset_rel  # Start at 6 minutes
        if interval_duration_min >= 15.0:
            # Long runs (>= 15 min): 4-minute calibration window (6:00-10:00)
            cal_duration = params.vt1_calibration_long
        else:
            # Short runs (< 15 min): 1-minute calibration window (6:00-7:00)
            cal_duration = params.vt1_calibration_short
        cal_end = cal_start + cal_duration
    else:
        # VT2: 1-minute calibration after Phase III onset
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

    # Find calibration window (in bin domain)
    cal_mask = (bin_times_rel >= cal_start) & (bin_times_rel <= cal_end)
    n_cal_points = np.sum(cal_mask)

    if n_cal_points < 3:
        # Extend calibration window if needed
        cal_mask = (bin_times_rel >= cal_start) & (bin_times_rel <= cal_start + 60)
        n_cal_points = np.sum(cal_mask)
        if n_cal_points < 2:
            # Fallback: use points around the breakpoint
            cal_mask = np.abs(bin_times_rel - phase3_onset_rel) <= 30

    cal_ve = ve_binned[cal_mask]
    cal_times_min = bin_times_min[cal_mask]

    # Baseline model using calibration window
    cal_midpoint_min = np.mean(cal_times_min) if len(cal_times_min) > 0 else phase3_onset_rel / 60.0
    cal_ve_mean = np.mean(cal_ve) if len(cal_ve) > 0 else np.mean(ve_binned)

    # Convert percentage drift to absolute drift (L/min per minute)
    expected_drift = (expected_drift_pct / 100.0) * cal_ve_mean

    # Alpha is the VE at t=0 that would give cal_ve_mean at cal_midpoint given expected_drift
    alpha = cal_ve_mean - expected_drift * cal_midpoint_min

    # Calculate expected VE for all bin timepoints
    ve_expected = alpha + expected_drift * bin_times_min

    # Fixed sigma as percentage of baseline VE
    sigma_ref = (sigma_pct / 100.0) * cal_ve_mean

    # CUSUM parameters
    k = params.slack_multiplier * sigma_ref
    h = h_mult * sigma_ref

    # =========================================================================
    # CUSUM calculation - starts after calibration window
    # =========================================================================

    cusum = np.zeros(len(bin_times_rel))
    s = 0.0
    alarm_time = None
    alarm_triggered = False

    for i in range(len(bin_times_rel)):
        if bin_times_rel[i] >= cal_end:
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

    # Slope estimation on post-calibration data using robust Huber regression
    analysis_mask = bin_times_rel >= cal_end
    n_analysis_points = np.sum(analysis_mask)
    interval_end_rel = interval.end_time - breath_times_raw[0]

    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]

        # Robust slope estimation (overall slope)
        slope, intercept = fit_single_slope(analysis_times_min, analysis_ve)

        # Slope line for visualization - extend from calibration end to full interval end
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
    else:
        slope = 0
        slope_line_times_rel = np.array([])
        slope_line_ve = np.array([])

    # =========================================================================
    # SECOND HINGE: Detect slope change after Phase III onset
    # =========================================================================

    # Get post-Phase III data for 2nd hinge fitting
    post_phase3_mask = bin_times_rel >= phase3_onset_rel
    post_phase3_times = bin_times_rel[post_phase3_mask]
    post_phase3_ve = ve_binned[post_phase3_mask]

    # Initialize 2nd hinge variables
    hinge2_time_rel = None
    hinge2_time = None
    slope1_pct = None
    slope2_pct = None
    split_slope_ratio = None
    hinge2_detected = False

    if len(post_phase3_times) >= 4:
        # Fit 2nd hinge model
        tau2, h2_b0, h2_b1, h2_b2, _, hinge2_detected = fit_second_hinge(
            post_phase3_times, post_phase3_ve, phase3_onset_rel, interval_end_rel
        )

        hinge2_time_rel = tau2
        hinge2_time = tau2 + breath_times_raw[0]

        # Calculate slopes before and after 2nd hinge (in L/min per minute)
        # h2_b1 is slope in (L/min) per second, multiply by 60 to get per minute
        # Slope before 2nd hinge: h2_b1 (from hinge model)
        # Slope after 2nd hinge: h2_b1 + h2_b2
        slope1_abs = h2_b1 * 60.0  # Convert from per-second to per-minute
        slope2_abs = (h2_b1 + h2_b2) * 60.0

        # Convert to percentage of baseline
        if cal_ve_mean > 0:
            slope1_pct = (slope1_abs / cal_ve_mean) * 100.0
            slope2_pct = (slope2_abs / cal_ve_mean) * 100.0
        else:
            slope1_pct = 0.0
            slope2_pct = 0.0

        # Calculate split slope ratio (slope2 / slope1)
        # If slope1 <= 0, treat as small positive value
        if slope1_pct <= 0:
            slope1_for_ratio = 0.001  # Small positive value
        else:
            slope1_for_ratio = slope1_pct

        split_slope_ratio = slope2_pct / slope1_for_ratio

        # Generate segment 2 (Phase III onset to 2nd hinge) and segment 3 (2nd hinge to end)
        # Using 2nd hinge model: VE(t) = h2_b0 + h2_b1*t + h2_b2*max(0, t - tau2)
        segment2_times_rel = np.array([phase3_onset_rel, hinge2_time_rel])
        segment2_ve = h2_b0 + h2_b1 * segment2_times_rel + h2_b2 * np.maximum(0, segment2_times_rel - hinge2_time_rel)

        segment3_times_rel = np.array([hinge2_time_rel, interval_end_rel])
        segment3_ve = h2_b0 + h2_b1 * segment3_times_rel + h2_b2 * np.maximum(0, segment3_times_rel - hinge2_time_rel)

        # Ensure continuity: adjust segment1 end to match segment2 start
        segment1_ve[-1] = segment2_ve[0]

    # Calculate last 60 seconds average VE for cumulative drift
    interval_duration = bin_times_rel[-1] if len(bin_times_rel) > 0 else 0
    last_60s_start = max(0, interval_duration - 60)
    last_60s_mask = bin_times_rel >= last_60s_start

    if np.sum(last_60s_mask) > 0:
        last_60s_avg_ve = np.mean(ve_binned[last_60s_mask])
    else:
        last_60s_avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    # =========================================================================
    # CLASSIFICATION: Based on CUSUM status, overall slope %, and Max Drift threshold
    # =========================================================================

    cusum_alarm = peak_cusum >= h
    cusum_recovered = cusum_alarm and (final_cusum <= recovered_threshold)

    # Overall slope as percentage of baseline (already calculated as ve_drift_pct later, compute here)
    overall_slope_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0

    # Use split_slope_ratio for classification (default to 1.0 if not computed)
    split_ratio = split_slope_ratio if split_slope_ratio is not None else 1.0

    # Classification thresholds: use max_drift_threshold from params
    # max_drift_threshold is the "high" threshold (1% for VT1, 3% for VT2)
    # low_threshold is expected_drift_pct (0.3% for VT1, 1% for VT2)
    low_threshold = expected_drift_pct
    high_threshold = max_drift_threshold

    split_ratio_threshold = 1.2

    # Apply classification tree
    if not cusum_alarm or cusum_recovered:
        # No CUSUM alarm OR alarm + recovered
        if overall_slope_pct < low_threshold:
            status = IntervalStatus.BELOW_THRESHOLD
        elif overall_slope_pct < high_threshold:
            # Middle range: check split slope ratio
            if split_ratio < split_ratio_threshold:
                status = IntervalStatus.BELOW_THRESHOLD
            else:
                status = IntervalStatus.BORDERLINE
        else:
            # >= high_threshold (Max Drift)
            status = IntervalStatus.BORDERLINE
    else:
        # CUSUM alarm + no recovery
        if overall_slope_pct < low_threshold:
            status = IntervalStatus.BORDERLINE
        elif overall_slope_pct < high_threshold:
            # Middle range: check split slope ratio
            if split_ratio < split_ratio_threshold:
                status = IntervalStatus.BORDERLINE
            else:
                status = IntervalStatus.ABOVE_THRESHOLD
        else:
            # >= high_threshold (Max Drift)
            status = IntervalStatus.ABOVE_THRESHOLD

    # Convert times to absolute for output
    abs_bin_times = bin_times_rel + breath_times_raw[0]
    abs_slope_line_times = slope_line_times_rel + breath_times_raw[0] if len(slope_line_times_rel) > 0 else np.array([])
    abs_segment1_times = segment1_times_rel + breath_times_raw[0]
    abs_segment2_times = segment2_times_rel + breath_times_raw[0] if segment2_times_rel is not None else None
    abs_segment3_times = segment3_times_rel + breath_times_raw[0] if segment3_times_rel is not None else None

    # Calculate drift as percentage of baseline
    ve_drift_pct = (slope / cal_ve_mean * 100.0) if cal_ve_mean > 0 else 0.0

    # Calculate metrics for table display
    initial_ve = cal_ve_mean

    # Avg VE: average VE in post-calibration period
    post_cal_mask = bin_times_rel >= cal_end
    if np.sum(post_cal_mask) > 0:
        avg_ve = np.mean(ve_binned[post_cal_mask])
    else:
        avg_ve = np.mean(ve_binned) if len(ve_binned) > 0 else 0

    terminal_ve = last_60s_avg_ve

    # Extend CUSUM line to the end of the interval
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
        breath_times=breath_times_raw,
        expected_ve=ve_expected,
        slope_line_times=abs_slope_line_times,
        slope_line_ve=slope_line_ve,
        last_60s_avg_ve=last_60s_avg_ve,
        cusum_threshold=h,
        initial_ve=initial_ve,
        avg_ve=avg_ve,
        terminal_ve=terminal_ve,
        is_ceiling_based=False,
        is_segmented=True,
        phase3_onset_time=phase3_onset_time,
        phase3_onset_rel=phase3_onset_rel,
        segment1_times=abs_segment1_times,
        segment1_ve=segment1_ve,
        segment2_times=abs_segment2_times,
        segment2_ve=segment2_ve,
        segment3_times=abs_segment3_times,
        segment3_ve=segment3_ve,
        hinge2_time=hinge2_time,
        hinge2_time_rel=hinge2_time_rel,
        slope1_pct=slope1_pct,
        slope2_pct=slope2_pct,
        split_slope_ratio=split_slope_ratio,
        hinge2_detected=hinge2_detected,
        speed=speed
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
    
    # Robust slope estimation on post-warmup data (for information only)
    analysis_mask = bin_times_rel >= params.ceiling_warmup_sec
    n_analysis_points = np.sum(analysis_mask)

    if n_analysis_points >= 3:
        analysis_times_min = bin_times_min[analysis_mask]
        analysis_ve = ve_binned[analysis_mask]

        # Robust Huber slope estimation
        slope, intercept = fit_single_slope(analysis_times_min, analysis_ve)

        # Slope line for visualization
        interval_end_rel = interval.end_time - breath_times_raw[0]
        slope_line_times_rel = np.array([bin_times_rel[analysis_mask][0], interval_end_rel])
        slope_line_times_min = slope_line_times_rel / 60.0
        slope_line_ve = intercept + slope * slope_line_times_min
    else:
        slope = 0
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
    Compute cumulative drift across all intervals as a segmented polyline.
    Only applicable for VT2 interval runs with 2+ intervals.

    Uses the terminal VE (last 60s average) from each interval as hinge points.
    The cumulative drift is the sum of individual segment slopes, expressed as
    total VE change from first to last interval as % of baseline per minute.
    """
    if len(results) < 2:
        return None

    # Extract data points: (end_time, terminal_ve) for each interval
    interval_end_times = np.array([r.interval.end_time for r in results])  # Keep in seconds
    interval_avg_ve = np.array([r.last_60s_avg_ve for r in results])

    # Baseline is interval 1's terminal VE
    baseline_ve = interval_avg_ve[0]

    if baseline_ve <= 0:
        return None

    # Calculate cumulative drift as total change from first to last interval
    # expressed as % of baseline per minute of actual work time
    total_ve_change = interval_avg_ve[-1] - interval_avg_ve[0]

    # Calculate total work time (sum of all interval durations, excluding recoveries)
    total_work_time_min = sum((r.interval.end_time - r.interval.start_time) / 60.0 for r in results)

    if total_work_time_min > 0:
        # Slope as L/min per minute of work
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

    # Polyline: connect all terminal VE points directly (no regression)
    # line_times and line_ve are the actual points to connect
    line_times = interval_end_times  # Already in seconds
    line_ve = interval_avg_ve

    return CumulativeDriftResult(
        slope_abs=slope_abs,
        slope_pct=slope_pct,
        baseline_ve=baseline_ve,
        pvalue=pvalue,
        interval_end_times=interval_end_times / 60.0,  # Store in minutes for display
        interval_avg_ve=interval_avg_ve,
        line_times=line_times,  # In seconds for plotting
        line_ve=line_ve
    )

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

    # 2. Recovery period shading (grey) - gaps between intervals with labels
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
                # Add "Rest" label with duration at top of chart
                recovery_duration_sec = recovery_end - recovery_start
                recovery_min = int(recovery_duration_sec // 60)
                recovery_sec = int(recovery_duration_sec % 60)
                fig.add_annotation(
                    x=(recovery_start + recovery_end) / 2,
                    y=1.02,
                    yref="paper",
                    text=f"Rest<br>{recovery_min}:{recovery_sec:02d}",
                    showarrow=False,
                    font=dict(size=10, color="#666")
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

    # Add interval labels centered in interval with duration
    for i, interval in enumerate(intervals):
        # Calculate interval duration
        interval_duration_sec = interval.end_time - interval.start_time
        interval_min = int(interval_duration_sec // 60)
        interval_sec = int(interval_duration_sec % 60)

        # Add interval label with duration
        fig.add_annotation(
            x=(interval.start_time + interval.end_time) / 2,
            y=1.02,
            yref="paper",
            text=f"Int {interval.interval_num}<br>{interval_min}:{interval_sec:02d}",
            showarrow=False,
            font=dict(size=10, color="#666")
        )

    # Add yellow ramp-up period shading for each interval
    for i, result in enumerate(results):
        if result.is_segmented and result.phase3_onset_time is not None:
            interval_start = result.interval.start_time
            ramp_end = result.phase3_onset_time
            ramp_duration_sec = ramp_end - interval_start
            ramp_duration_min = int(ramp_duration_sec // 60)
            ramp_duration_s = int(ramp_duration_sec % 60)

            # Add yellow shading for ramp-up period
            fig.add_shape(
                type="rect",
                x0=interval_start,
                x1=ramp_end,
                y0=0,
                y1=1,
                yref="paper",
                fillcolor='rgba(255, 255, 200, 0.5)',  # Light yellow
                layer="below",
                line_width=0,
            )

            # Add "Ramp" label with duration at top of chart
            fig.add_annotation(
                x=(interval_start + ramp_end) / 2,
                y=1.02,
                yref="paper",
                text=f"Ramp<br>{ramp_duration_min}:{ramp_duration_s:02d}",
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

    # Plot continuous VE smoothed line (including recoveries) - lighter blue
    fig.add_trace(
        go.Scatter(
            x=bin_times_all,
            y=ve_binned_all,
            mode='lines',
            name='VE (smoothed)',
            line=dict(color='#5DADE2', width=2),  # Lighter blue (but darker than breath dots)
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
        # For drift-based analysis: show slope line (solid green)
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
                        name='Slope',
                        line=dict(color='#8B5CF6', width=2),  # Purple to match slope
                        showlegend=(not shown_slope_legend),
                        hoverinfo='skip'
                    ),
                    secondary_y=False
                )
                shown_slope_legend = True
        # Note: Segmented slope lines (3 segments) are drawn later in the is_segmented block

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

        # CUSUM trace colored by alarm state: green when OK/recovered, red when in alarm
        # Get the CUSUM threshold (H) for detecting alarm/recovery
        h_threshold = result.cusum_threshold if result.cusum_threshold > 0 else result.peak_cusum
        recovery_threshold = h_threshold / 2

        # Build segments with colors based on alarm state
        in_alarm_state = False
        segment_start_idx = 0

        # Track state changes to create colored segments
        for i, (t, c) in enumerate(zip(result.time_values, result.cusum_values)):
            state_changed = False
            new_alarm_state = in_alarm_state

            if not in_alarm_state and c >= h_threshold:
                # Alarm triggered
                new_alarm_state = True
                state_changed = True
            elif in_alarm_state and c <= recovery_threshold:
                # Recovered
                new_alarm_state = False
                state_changed = True

            if state_changed and i > segment_start_idx:
                # Draw segment up to this point with previous color
                segment_times = result.time_values[segment_start_idx:i+1]
                segment_values = result.cusum_values[segment_start_idx:i+1]
                segment_color = 'rgba(255, 0, 0, 0.5)' if in_alarm_state else 'rgba(0, 128, 0, 0.5)'

                fig.add_trace(
                    go.Scatter(
                        x=segment_times,
                        y=segment_values,
                        mode='lines',
                        name='CUSUM',
                        line=dict(color=segment_color, width=1.5),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=cusum_hover_texts[segment_start_idx:i+1]
                    ),
                    secondary_y=True
                )
                segment_start_idx = i
                in_alarm_state = new_alarm_state

        # Draw final segment
        if segment_start_idx < len(result.time_values) - 1:
            segment_times = result.time_values[segment_start_idx:]
            segment_values = result.cusum_values[segment_start_idx:]
            segment_color = 'rgba(255, 0, 0, 0.5)' if in_alarm_state else 'rgba(0, 128, 0, 0.5)'

            fig.add_trace(
                go.Scatter(
                    x=segment_times,
                    y=segment_values,
                    mode='lines',
                    name='CUSUM',
                    line=dict(color=segment_color, width=1.5),
                    showlegend=(result.interval.interval_num == 1 and segment_start_idx == 0),
                    hoverinfo='text',
                    hovertext=cusum_hover_texts[segment_start_idx:]
                ),
                secondary_y=True
            )

        # Note: Phase III onset line removed - replaced by yellow ramp-up shading

        # Plot segment lines for segmented analysis (purple slope line)
        if result.is_segmented:
            # Build continuous line by concatenating all available segments
            all_times = []
            all_ve = []

            # Segment 1 (start to Phase III onset) - part of ramp-up, don't include in main slope line
            # Only include segments 2 and 3 (after ramp-up period)

            # Segment 2 (Phase III onset to 2nd hinge)
            if result.segment2_times is not None and len(result.segment2_times) > 0:
                all_times.extend(result.segment2_times)
                all_ve.extend(result.segment2_ve)

            # Segment 3 (2nd hinge to end)
            if result.segment3_times is not None and len(result.segment3_times) > 0:
                # Skip first point if it overlaps with segment2 end
                if len(all_times) > 0 and len(result.segment3_times) > 0:
                    all_times.extend(result.segment3_times[1:])
                    all_ve.extend(result.segment3_ve[1:])
                else:
                    all_times.extend(result.segment3_times)
                    all_ve.extend(result.segment3_ve)

            # Draw the continuous slope line (purple)
            if len(all_times) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=all_times,
                        y=all_ve,
                        mode='lines',
                        name='Slope',
                        line=dict(color='#8E44AD', width=2),  # Purple
                        showlegend=(not shown_slope_legend and result.interval.interval_num == 1),
                        hoverinfo='skip'
                    ),
                    secondary_y=False
                )
                shown_slope_legend = True

            # Add slope%/min annotations for each segment after blanking
            # Segment 2 (slope1_pct) - from Phase III onset to 2nd hinge
            if result.slope1_pct is not None and result.segment2_times is not None and len(result.segment2_times) >= 2:
                seg2_mid_x = (result.segment2_times[0] + result.segment2_times[-1]) / 2
                seg2_mid_y = (result.segment2_ve[0] + result.segment2_ve[-1]) / 2
                fig.add_annotation(
                    x=seg2_mid_x,
                    y=seg2_mid_y,
                    text=f"{result.slope1_pct:+.1f}%/min",
                    showarrow=False,
                    font=dict(size=8, color="#8E44AD"),  # Purple (matches slope line)
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    yshift=10
                )

            # Segment 3 (slope2_pct) - from 2nd hinge to end
            if result.slope2_pct is not None and result.segment3_times is not None and len(result.segment3_times) >= 2:
                seg3_mid_x = (result.segment3_times[0] + result.segment3_times[-1]) / 2
                seg3_mid_y = (result.segment3_ve[0] + result.segment3_ve[-1]) / 2
                fig.add_annotation(
                    x=seg3_mid_x,
                    y=seg3_mid_y,
                    text=f"{result.slope2_pct:+.1f}%/min",
                    showarrow=False,
                    font=dict(size=8, color="#8E44AD"),  # Purple (matches slope line)
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    yshift=10
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

    # Create tick values and labels for m:s format on x-axis
    # Determine tick interval based on data range
    data_range = x_max - x_min if selected_interval is not None else (breath_df['breath_time'].max() - breath_df['breath_time'].min())
    if data_range <= 600:  # 10 minutes or less
        tick_interval = 60  # Every minute
    elif data_range <= 1800:  # 30 minutes or less
        tick_interval = 120  # Every 2 minutes
    else:
        tick_interval = 300  # Every 5 minutes

    # Generate tick values starting from 0
    tick_start = 0
    tick_end = int(breath_df['breath_time'].max()) + tick_interval
    tick_vals = list(range(tick_start, tick_end, tick_interval))

    # Create m:s labels
    tick_labels = [f"{int(t // 60)}:{int(t % 60):02d}" for t in tick_vals]

    fig.update_xaxes(
        title_text="Time (m:s)",
        range=[x_min, x_max] if selected_interval is not None else None,
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_labels
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
        # Data source selection
        data_source = st.radio(
            "Data Source",
            options=["Upload File", "Fetch from Cloud"],
            horizontal=True,
            help="Upload a local CSV or fetch from iOS app uploads"
        )

        uploaded_file = None

        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                help="Upload CSV from VitalPro or iOS app"
            )
        else:
            # Fetch from Cloud
            cloud_sessions = list_cloud_sessions()

            if not cloud_sessions:
                st.info("No cloud sessions available. Upload from iOS app first.")
            else:
                # Create display options
                session_options = ["-- Select a session --"] + [
                    f"{s['filename']} ({s['uploaded_at'][:10] if s['uploaded_at'] else 'unknown'})"
                    for s in cloud_sessions
                ]

                selected_idx = st.selectbox(
                    "Select Session",
                    options=range(len(session_options)),
                    format_func=lambda i: session_options[i],
                    help="Sessions uploaded from iOS app"
                )

                if selected_idx > 0:
                    # Load the selected session from API
                    session = cloud_sessions[selected_idx - 1]
                    csv_content = get_cloud_session_content(session['session_id'])

                    if csv_content:
                        uploaded_file = CloudSessionFile(
                            session['session_id'],
                            session['filename'],
                            csv_content
                        )
                    else:
                        st.error("Failed to load session from cloud")

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
                st.session_state.ios_run_params = None
                st.session_state.detected_phase3_onset = None
                st.session_state.current_file_id = file_id
                st.session_state.current_file_name = uploaded_file.name

            # Only parse if we haven't already (or if data was cleared)
            if st.session_state.breath_df is None:
                try:
                    # Auto-detect CSV format (iOS or VitalPro)
                    breath_df, metadata, power_df, run_params = parse_csv_auto(uploaded_file)
                    st.session_state.breath_df = breath_df
                    st.session_state.power_df = power_df
                    st.session_state.ios_run_params = run_params  # Store iOS params if present
                    st.success(f"Loaded {len(breath_df)} breaths")

                    # Use iOS params if available, otherwise auto-detect from power data
                    if run_params is not None:
                        # iOS format - use provided params
                        st.session_state.detected_params = {
                            'run_type': run_params.get('run_type', RunType.VT1_STEADY),
                            'num_intervals': run_params.get('num_intervals', 1),
                            'interval_duration': run_params.get('interval_duration', 30.0),
                            'recovery_duration': run_params.get('recovery_duration', 0.0),
                            'speeds': run_params.get('speeds', []),
                            'vt1_threshold': run_params.get('vt1_threshold'),
                            'vt2_threshold': run_params.get('vt2_threshold')
                        }
                        st.caption("iOS CSV format detected")
                    else:
                        # VitalPro format - auto-detect
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
                st.success(f"Loaded {len(st.session_state.breath_df)} breaths")

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
        
        # Initialize params with default values
        params = AnalysisParams()
        # Set ceiling parameters
        params.vt1_ve_ceiling = vt1_ve_ceiling
        params.vt2_ve_ceiling = vt2_ve_ceiling
        params.use_thresholds_for_all = use_thresholds_for_all

        st.markdown("---")

        # Collapsible Advanced section (collapsed by default)
        with st.expander("Advanced", expanded=False):
            st.markdown("#### Ramp-Up Period")

            # Show detected ramp-up period (Phase III onset) if available
            detected_phase3 = st.session_state.get('detected_phase3_onset')
            if detected_phase3 is not None:
                detected_min = int(detected_phase3 // 60)
                detected_sec = int(detected_phase3 % 60)
                st.info(f"Detected: {detected_min}:{detected_sec:02d}")
            else:
                st.caption("VT1: Fixed at 6:00. VT2: Auto-detected (90-180s)")

            col1, col2 = st.columns(2)
            with col1:
                phase3_override = st.number_input(
                    "Ramp-Up Period (s)",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    key=f"phase3_override_{file_key}",
                    help="Override ramp-up period end (0 = auto). VT1 default: 360s (6min). VT2: auto-detected."
                )
                # Set override if user entered a value
                if phase3_override > 0:
                    params.phase3_onset_override = phase3_override
            with col2:
                # Note: Calibration window is now automatic based on run type/duration
                st.caption("Cal. window: VT1 short (<15m): 1m, VT1 long: 4m, VT2: 1m")

            st.markdown("---")
            st.markdown("#### Max Drift Thresholds")
            st.caption("Slope threshold for classifying as 'above threshold'")

            col1, col2 = st.columns(2)
            with col1:
                params.max_drift_pct_vt1 = st.number_input(
                    "Max Drift VT1 (%)",
                    min_value=0.1, max_value=10.0,
                    value=default_params.max_drift_pct_vt1,
                    step=0.1,
                    key=f"max_drift_vt1_{file_key}",
                    help="Maximum allowable slope (%/min) before VT1 is classified as above threshold"
                )
            with col2:
                params.max_drift_pct_vt2 = st.number_input(
                    "Max Drift VT2 (%)",
                    min_value=0.5, max_value=15.0,
                    value=default_params.max_drift_pct_vt2,
                    step=0.5,
                    key=f"max_drift_vt2_{file_key}",
                    help="Maximum allowable slope (%/min) before VT2 is classified as above threshold"
                )

            st.markdown("---")
            st.markdown("#### CUSUM Parameters")

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
                    step=0.1,
                    help="Expected VE drift for VT1 (% of baseline per min) - used in CUSUM calculation"
                )
            with col2:
                params.expected_drift_pct_vt2 = st.number_input(
                    "Drift VT2 (%)",
                    min_value=0.5, max_value=15.0,
                    value=default_params.expected_drift_pct_vt2,
                    step=0.5,
                    help="Expected VE drift for VT2 (% of baseline per min) - used in CUSUM calculation"
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

            # Get speeds from iOS params if available
            ios_params = st.session_state.get('ios_run_params')
            speeds = ios_params.get('speeds', []) if ios_params else []

            # Run analysis - choose algorithm based on interval duration and checkbox
            results = []
            detected_phase3 = None  # Track detected Phase III from first long interval

            for i, interval in enumerate(intervals):
                interval_duration_sec = interval.end_time - interval.start_time
                interval_duration_min = interval_duration_sec / 60.0

                # Get speed for this interval (if available)
                speed = speeds[i] if i < len(speeds) else (speeds[0] if speeds else None)

                # Use ceiling-based CUSUM if:
                # 1. Checkbox is checked (use_thresholds_for_all), OR
                # 2. Interval duration is less than 6 minutes
                use_ceiling_based = params.use_thresholds_for_all or (interval_duration_min < 6.0)

                if use_ceiling_based:
                    result = analyze_interval_ceiling(breath_df, interval, params, run_type)
                    result.speed = speed
                else:
                    result = analyze_interval_segmented(breath_df, interval, params, run_type, speed)
                    # Track detected Phase III onset from first interval with segmented analysis
                    if detected_phase3 is None and result.phase3_onset_rel is not None:
                        detected_phase3 = result.phase3_onset_rel
                results.append(result)

            # Store detected Phase III onset for display in sidebar
            if detected_phase3 is not None:
                st.session_state.detected_phase3_onset = detected_phase3

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
        
        # Summary row - simplified to just Run Type, Cumulative Drift (VT2), and Average VE
        # Calculate Average VE across all post-blanking periods
        all_avg_ve = [r.avg_ve for r in results if r.avg_ve > 0]
        overall_avg_ve = np.mean(all_avg_ve) if all_avg_ve else 0

        # Show 3 columns if cumulative drift available, otherwise 2
        if cumulative_drift is not None:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)

        with col1:
            st.metric("Run Type", run_type.value)

        if cumulative_drift is not None:
            with col2:
                drift_str = f"{cumulative_drift.slope_pct:+.2f}%/min ({cumulative_drift.baseline_ve:.1f} L/min)"
                st.metric(
                    "Cumulative Drift",
                    drift_str,
                    help="Rate of VE increase across all intervals, expressed as % of baseline VE per minute"
                )
            with col3:
                st.metric(
                    "Average VE",
                    f"{overall_avg_ve:.1f} L/min",
                    help="Mean VE across all post-blanking periods"
                )
        else:
            with col2:
                st.metric(
                    "Average VE",
                    f"{overall_avg_ve:.1f} L/min",
                    help="Mean VE across all post-blanking periods"
                )
        
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

        # Check if any interval has speed data
        has_speed = any(r.speed is not None for r in results)

        # Different table layout for ceiling-based vs drift-based
        # Headers with tooltips: (display_name, tooltip)
        header_tooltips = {
            "#": "Interval number (click to zoom)",
            "Status": "Classification based on CUSUM and slope analysis",
            "Peak VE": "Maximum VE recorded during this interval (L/min)",
            "Avg VE": "Average VE during post-blanking period (L/min)",
            "Terminal VE": "Average VE in last 60 seconds of interval (L/min)",
            "Speed": "Running speed (mph)",
            "VE Drift": "Slope as % of baseline VE per minute (baseline L/min)",
            "Initial VE": "Average VE during calibration window (L/min)"
        }

        if all_ceiling_based:
            # Ceiling-based: #, Status, Peak VE, Avg VE, Terminal VE [, Speed] (no VE Drift, no Initial VE)
            if has_speed:
                col_widths = [0.4, 1.4, 0.7, 0.7, 0.8, 0.6]
                headers = ["#", "Status", "Peak VE", "Avg VE", "Terminal VE", "Speed"]
            else:
                col_widths = [0.4, 1.6, 0.8, 0.8, 0.8]
                headers = ["#", "Status", "Peak VE", "Avg VE", "Terminal VE"]
        else:
            # Drift-based: #, Status, VE Drift, Peak VE, Initial VE, Avg VE, Terminal VE [, Speed]
            if has_speed:
                col_widths = [0.4, 1.2, 0.9, 0.6, 0.7, 0.7, 0.8, 0.6]
                headers = ["#", "Status", "VE Drift", "Peak VE", "Initial VE", "Avg VE", "Terminal VE", "Speed"]
            else:
                col_widths = [0.4, 1.4, 1.0, 0.6, 0.7, 0.7, 0.8]
                headers = ["#", "Status", "VE Drift", "Peak VE", "Initial VE", "Avg VE", "Terminal VE"]

        # Center the table
        table_left, table_center, table_right = st.columns([0.3, 4, 0.3])
        with table_center:
            # Header - add top margin to ensure visibility
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            header_cols = st.columns(col_widths)
            for i, h in enumerate(headers):
                tooltip = header_tooltips.get(h, "")
                header_cols[i].markdown(
                    f"<div style='font-weight:600;font-size:11px;text-align:center;padding:8px 2px;border-bottom:2px solid #ccc;background:#f5f5f5;cursor:help;' title='{tooltip}'>{h}</div>",
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
                        # Ceiling-based columns: Status, Peak, Avg VE, Terminal VE [, Speed]
                        row_cols[1].markdown(f"<div style='{cell}'>{r.status.value}</div>", unsafe_allow_html=True)
                        row_cols[2].markdown(f"<div style='{cell}'>{r.peak_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[3].markdown(f"<div style='{cell}'>{r.avg_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[4].markdown(f"<div style='{cell}'>{r.terminal_ve:.0f}</div>", unsafe_allow_html=True)
                        if has_speed:
                            speed_str = f"{r.speed:.1f}" if r.speed is not None else "-"
                            row_cols[5].markdown(f"<div style='{cell}'>{speed_str}</div>", unsafe_allow_html=True)
                    else:
                        # Drift-based columns: Status, VE Drift, Peak, Initial VE, Avg VE, Terminal VE [, Speed]
                        row_cols[1].markdown(f"<div style='{cell}'>{r.status.value}</div>", unsafe_allow_html=True)
                        row_cols[2].markdown(f"<div style='{cell}'>{r.ve_drift_pct:+.1f}% ({r.baseline_ve:.0f})</div>", unsafe_allow_html=True)
                        row_cols[3].markdown(f"<div style='{cell}'>{r.peak_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[4].markdown(f"<div style='{cell}'>{r.initial_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[5].markdown(f"<div style='{cell}'>{r.avg_ve:.0f}</div>", unsafe_allow_html=True)
                        row_cols[6].markdown(f"<div style='{cell}'>{r.terminal_ve:.0f}</div>", unsafe_allow_html=True)
                        if has_speed:
                            speed_str = f"{r.speed:.1f}" if r.speed is not None else "-"
                            row_cols[7].markdown(f"<div style='{cell}'>{speed_str}</div>", unsafe_allow_html=True)

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
        with st.expander("How it works"):
            st.markdown("""
            **1. Data Processing**
            - Extracts breath-by-breath VE data from VitalPro or iOS app CSV
            - Stage 1: Rolling median filter (9 breaths) removes outliers/artifacts
            - Stage 2: 4-second bin averaging standardizes time series (~3-4 breaths/bin at 55 br/min)
            - Uses breath timestamps (not row index) for accurate timing

            **2. Interval Detection**
            - Auto-detects work/recovery intervals from power data (VitalPro)
            - Uses metadata from iOS app CSV if available
            - First interval always starts at t=0
            - Intervals are always whole minutes; rest rounds to 0.5-minute increments

            **3. Phase III Detection (for intervals >= 6 min)**
            - Uses segmented regression (hinge model) to detect Phase III onset
            - Constrained to 90-240 seconds range
            - Baseline calibrated from 30-second window after Phase III onset
            - Falls back to 150s default if detection fails

            **4. CUSUM Analysis - Two Algorithms**

            *Segmented CUSUM (for intervals >= 6 min):*
            - Auto-detects Phase III onset using segmented regression
            - Self-calibrates baseline from 30s window after Phase III onset
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

            **5. Cumulative Drift (VT2 only)**
            - Tracks VE drift across all intervals
            - Uses last 60s average from each interval
            - Robust Huber regression across elapsed workout time
            - Shows overall fatigue/drift accumulation

            **6. Classification**
            - **Below Threshold**: CUSUM no alarm (or alarm + recovered for ceiling-based)
            - **Borderline**: Alarm triggered but recovered
            - **Above Threshold**: CUSUM alarm sustained (not recovered)
            """)

if __name__ == "__main__":
    main()