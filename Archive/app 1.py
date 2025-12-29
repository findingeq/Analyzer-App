"""
VT Threshold Analyzer - Desktop Application
Analyzes Tymewear VitalPro respiratory data to assess VT1/VT2 compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
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
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    RECOVERED = "⚠️ RECOVERED"

@dataclass
class AnalysisParams:
    # Phase II typically completes by ~3 minutes; slow component emerges after
    blanking_period: float = 150.0  # seconds - start of calibration lookback window
    calibration_end: float = 180.0  # seconds - end of calibration window (minute 3)
    h_multiplier_vt1: float = 5.0
    h_multiplier_vt2: float = 7.0
    slack_multiplier: float = 0.5
    expected_drift_vt1: float = 0.25  # L/min per minute (minimal in moderate domain)
    expected_drift_vt2: float = 2.0   # L/min per minute (slow component in heavy domain)
    median_window: int = 5
    savgol_window: int = 15
    savgol_poly: int = 2

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
    ve_drift_rate: float
    peak_ve: float
    peak_cusum: float
    final_cusum: float
    alarm_time: Optional[float]
    cusum_values: np.ndarray
    time_values: np.ndarray
    ve_filtered: np.ndarray
    expected_ve: np.ndarray

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
# SIGNAL FILTERING
# ============================================================================

def apply_filtering(ve_raw: np.ndarray, params: AnalysisParams) -> np.ndarray:
    """Apply rolling median then Savitzky-Golay filter to VE data."""
    if len(ve_raw) < params.savgol_window:
        return ve_raw
    
    # Step 1: Rolling median for outlier removal
    ve_median = median_filter(ve_raw, size=params.median_window, mode='nearest')
    
    # Step 2: Savitzky-Golay for smoothing
    # Ensure window is odd
    window = params.savgol_window if params.savgol_window % 2 == 1 else params.savgol_window + 1
    if len(ve_median) < window:
        window = len(ve_median) if len(ve_median) % 2 == 1 else len(ve_median) - 1
    
    if window >= 3:
        ve_filtered = savgol_filter(ve_median, window, params.savgol_poly)
    else:
        ve_filtered = ve_median
    
    return ve_filtered

# ============================================================================
# INTERVAL DETECTION
# ============================================================================

def detect_intervals(power_df: pd.DataFrame, breath_df: pd.DataFrame,
                     interval_duration: float, recovery_duration: float,
                     num_intervals: int) -> Tuple[RunType, List[Interval]]:
    """
    Detect run format and intervals from power data.
    Returns run type and list of interval boundaries.
    """
    if power_df.empty or len(power_df) < 10:
        # Default to steady state if no power data
        return RunType.VT1_STEADY, []
    
    power = power_df['power'].values
    time = power_df['time'].values
    
    # Remove initial ramp-up (first 30 seconds often have transient power)
    stable_mask = time > 30
    if np.sum(stable_mask) < 100:
        stable_mask = np.ones(len(time), dtype=bool)
    
    power_stable = power[stable_mask]
    
    # Smooth power heavily to reduce noise (30-second window)
    window_size = min(30, len(power) // 4)
    if window_size < 3:
        window_size = 3
    power_smooth = median_filter(power, size=window_size, mode='nearest')
    
    # Calculate power statistics on stable portion
    power_mean = np.nanmean(power_stable)
    power_range = np.nanmax(power_smooth[stable_mask]) - np.nanmin(power_smooth[stable_mask])
    
    # If power range is small relative to mean, likely steady state
    if power_range < 0.25 * power_mean:
        total_duration = breath_df['breath_time'].max()
        return RunType.VT1_STEADY, [Interval(
            start_time=0,
            end_time=total_duration,
            start_idx=0,
            end_idx=len(breath_df) - 1,
            interval_num=1
        )]
    
    # For interval detection, use a threshold between work and recovery
    # Use percentiles to find the two modes
    p25 = np.nanpercentile(power_smooth[stable_mask], 25)
    p75 = np.nanpercentile(power_smooth[stable_mask], 75)
    power_threshold = (p25 + p75) / 2
    
    # Apply hysteresis to avoid chatter
    hysteresis = (p75 - p25) * 0.1
    
    # State machine for interval detection
    in_work = power_smooth[0] > power_threshold
    intervals = []
    current_start = 0 if in_work else None
    interval_num = 0
    
    min_interval_duration = 60  # Minimum 1 minute for a valid interval
    
    for i in range(1, len(time)):
        if in_work:
            # Check for transition to recovery
            if power_smooth[i] < power_threshold - hysteresis:
                # End of work interval
                duration = time[i] - time[current_start]
                if duration >= min_interval_duration:
                    interval_num += 1
                    breath_times = breath_df['breath_time'].values
                    breath_start_idx = np.searchsorted(breath_times, time[current_start])
                    breath_end_idx = np.searchsorted(breath_times, time[i])
                    
                    intervals.append(Interval(
                        start_time=time[current_start],
                        end_time=time[i],
                        start_idx=breath_start_idx,
                        end_idx=min(breath_end_idx, len(breath_df) - 1),
                        interval_num=interval_num
                    ))
                in_work = False
                current_start = None
        else:
            # Check for transition to work
            if power_smooth[i] > power_threshold + hysteresis:
                in_work = True
                current_start = i
    
    # Handle case where we end in a work interval
    if in_work and current_start is not None:
        duration = time[-1] - time[current_start]
        if duration >= min_interval_duration:
            interval_num += 1
            breath_times = breath_df['breath_time'].values
            breath_start_idx = np.searchsorted(breath_times, time[current_start])
            
            intervals.append(Interval(
                start_time=time[current_start],
                end_time=time[-1],
                start_idx=breath_start_idx,
                end_idx=len(breath_df) - 1,
                interval_num=interval_num
            ))
    
    if len(intervals) > 1:
        return RunType.VT2_INTERVAL, intervals
    elif len(intervals) == 1:
        return RunType.VT1_STEADY, intervals
    else:
        # Fallback to steady state
        total_duration = breath_df['breath_time'].max()
        return RunType.VT1_STEADY, [Interval(
            start_time=0,
            end_time=total_duration,
            start_idx=0,
            end_idx=len(breath_df) - 1,
            interval_num=1
        )]

def create_manual_intervals(breath_df: pd.DataFrame, num_intervals: int,
                           interval_duration: float, recovery_duration: float) -> List[Interval]:
    """Create intervals based on manual specification."""
    intervals = []
    breath_times = breath_df['breath_time'].values
    
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

def analyze_interval(breath_df: pd.DataFrame, interval: Interval,
                     params: AnalysisParams, run_type: RunType) -> IntervalResult:
    """
    Perform CUSUM analysis on a single interval.
    
    Uses domain-expected drift rate as the baseline model slope, with
    self-calibrated intercept from the calibration window. This makes
    detection more robust to local VE variability.
    """
    
    # Extract interval data
    idx_start = interval.start_idx
    idx_end = interval.end_idx
    
    if idx_end <= idx_start:
        # Invalid interval
        return IntervalResult(
            interval=interval,
            status=IntervalStatus.PASS,
            ve_drift_rate=0,
            peak_ve=0,
            peak_cusum=0,
            final_cusum=0,
            alarm_time=None,
            cusum_values=np.array([0]),
            time_values=np.array([interval.start_time]),
            ve_filtered=np.array([0]),
            expected_ve=np.array([0])
        )
    
    ve_raw = breath_df['ve_raw'].values[idx_start:idx_end+1]
    breath_times = breath_df['breath_time'].values[idx_start:idx_end+1]
    
    # Relative time within interval (in seconds)
    rel_times = breath_times - breath_times[0]
    # Convert to minutes for drift calculations
    rel_times_min = rel_times / 60.0
    
    # Apply filtering
    ve_filtered = apply_filtering(ve_raw, params)
    
    # Determine domain-specific parameters
    if run_type == RunType.VT1_STEADY:
        h_mult = params.h_multiplier_vt1
        expected_drift = params.expected_drift_vt1  # L/min per minute
    else:
        h_mult = params.h_multiplier_vt2
        expected_drift = params.expected_drift_vt2  # L/min per minute
    
    # Find calibration window (90-150s into interval)
    cal_mask = (rel_times >= params.blanking_period) & (rel_times <= params.calibration_end)
    n_cal_points = np.sum(cal_mask)
    
    if n_cal_points < 10:
        # Extend calibration window if needed
        cal_mask = rel_times >= params.blanking_period
        n_cal_points = np.sum(cal_mask)
        
        if n_cal_points < 5:
            cal_mask = np.arange(len(rel_times)) >= len(rel_times) // 3
    
    cal_ve = ve_filtered[cal_mask]
    cal_times_min = rel_times_min[cal_mask]
    
    # Baseline model: VE_expected = alpha + expected_drift * t
    # Self-calibrate alpha (intercept) based on calibration window mean
    # Adjusted for the expected drift that occurred up to calibration midpoint
    cal_midpoint_min = np.mean(cal_times_min)
    cal_ve_mean = np.mean(cal_ve)
    
    # alpha is the VE at t=0 that would give cal_ve_mean at cal_midpoint given expected_drift
    alpha = cal_ve_mean - expected_drift * cal_midpoint_min
    
    # Calculate expected VE for all timepoints
    ve_expected = alpha + expected_drift * rel_times_min
    
    # Calculate residuals in calibration window to estimate noise (sigma)
    ve_expected_cal = alpha + expected_drift * cal_times_min
    residuals_cal = cal_ve - ve_expected_cal
    sigma_ref = np.std(residuals_cal)
    
    # Minimum noise floor
    if sigma_ref < 2.0:
        sigma_ref = 2.0
    
    # CUSUM parameters
    k = params.slack_multiplier * sigma_ref  # Slack
    h = h_mult * sigma_ref  # Threshold
    
    # CUSUM calculation - detect VE rising faster than expected
    cusum = np.zeros(len(rel_times))
    s = 0.0
    alarm_time = None
    alarm_triggered = False
    
    for i in range(len(rel_times)):
        if rel_times[i] >= params.calibration_end:
            # Residual: actual - expected (positive means VE higher than expected)
            residual = ve_filtered[i] - ve_expected[i]
            # One-sided upper CUSUM (detects increases beyond expected drift)
            s = max(0, s + residual - k)
            
            if s > h and not alarm_triggered:
                alarm_time = breath_times[i]  # Absolute time for display
                alarm_triggered = True
        
        cusum[i] = s
    
    # Determine status
    peak_cusum = np.max(cusum)
    final_cusum = cusum[-1] if len(cusum) > 0 else 0
    
    # Recovered threshold: half the alarm threshold
    recovered_threshold = h / 2
    
    if peak_cusum < h:
        status = IntervalStatus.PASS
    elif final_cusum <= recovered_threshold:
        status = IntervalStatus.RECOVERED
    else:
        status = IntervalStatus.FAIL
    
    # Calculate actual observed drift rate over the analysis window
    # (for display purposes - this is what actually happened, not the baseline model)
    analysis_mask = rel_times >= params.blanking_period
    if np.sum(analysis_mask) >= 2:
        analysis_times = rel_times_min[analysis_mask]
        analysis_ve = ve_filtered[analysis_mask]
        if np.std(analysis_times) > 0:
            observed_coeffs = np.polyfit(analysis_times, analysis_ve, 1)
            observed_drift = observed_coeffs[0]  # L/min per minute
        else:
            observed_drift = 0
    else:
        observed_drift = 0
    
    return IntervalResult(
        interval=interval,
        status=status,
        ve_drift_rate=observed_drift,  # Actual observed drift, not expected
        peak_ve=np.max(ve_filtered) if len(ve_filtered) > 0 else 0,
        peak_cusum=peak_cusum,
        final_cusum=final_cusum,
        alarm_time=alarm_time,
        cusum_values=cusum,
        time_values=breath_times,  # Absolute times for plotting
        ve_filtered=ve_filtered,
        expected_ve=ve_expected
    )

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_main_chart(breath_df: pd.DataFrame, results: List[IntervalResult],
                      intervals: List[Interval], params: AnalysisParams,
                      selected_interval: Optional[int] = None) -> go.Figure:
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
        # VE trace
        fig.add_trace(
            go.Scatter(
                x=result.time_values,
                y=result.ve_filtered,
                mode='lines',
                name=f'VE (Int {result.interval.interval_num})',
                line=dict(color='#2E86AB', width=1.5),
                showlegend=(result.interval.interval_num == 1)
            ),
            row=1, col=1
        )
        
        # Expected VE (baseline model)
        fig.add_trace(
            go.Scatter(
                x=result.time_values,
                y=result.expected_ve,
                mode='lines',
                name='Expected VE',
                line=dict(color='#90EE90', width=1, dash='dash'),
                showlegend=(result.interval.interval_num == 1)
            ),
            row=1, col=1
        )
        
        # CUSUM trace
        fig.add_trace(
            go.Scatter(
                x=result.time_values,
                y=result.cusum_values,
                mode='lines',
                name=f'CUSUM (Int {result.interval.interval_num})',
                line=dict(color='#E94F37', width=1.5),
                showlegend=(result.interval.interval_num == 1)
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
        
        .status-pass {
            color: #28a745;
            font-weight: 600;
        }
        
        .status-fail {
            color: #dc3545;
            font-weight: 600;
        }
        
        .status-recovered {
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Upload Data")
        
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
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 🏃 Run Format")
        
        # Run type selection
        run_type_option = st.selectbox(
            "Run Type",
            options=["Auto-detect", "VT1 (Steady State)", "VT2 (Intervals)"],
            help="Auto-detect analyzes power data to determine run format"
        )
        
        # Manual interval settings
        use_manual = st.checkbox("Manual interval settings", value=False)
        
        col1, col2 = st.columns(2)
        with col1:
            num_intervals = st.number_input(
                "# Intervals",
                min_value=1, max_value=30, value=12,
                disabled=not use_manual
            )
            interval_duration = st.number_input(
                "Interval (min)",
                min_value=1.0, max_value=30.0, value=4.0, step=1.0,
                disabled=not use_manual
            )
        with col2:
            recovery_duration = st.number_input(
                "Recovery (min)",
                min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                disabled=not use_manual
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
                min_value=30.0, max_value=180.0, value=90.0, step=10.0,
                help="Kinetic blanking period to ignore on-transient"
            )
        with col2:
            params.calibration_end = st.number_input(
                "Cal. End (s)",
                min_value=60.0, max_value=240.0, value=150.0, step=10.0,
                help="End of calibration window"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            params.h_multiplier_vt1 = st.number_input(
                "H (VT1)",
                min_value=3.0, max_value=10.0, value=5.0, step=0.5,
                help="H threshold multiplier for VT1"
            )
        with col2:
            params.h_multiplier_vt2 = st.number_input(
                "H (VT2)",
                min_value=3.0, max_value=12.0, value=7.0, step=0.5,
                help="H threshold multiplier for VT2"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            params.expected_drift_vt1 = st.number_input(
                "Drift VT1",
                min_value=0.0, max_value=2.0, value=0.25, step=0.1,
                help="Expected VE drift for VT1 (L/min per min)"
            )
        with col2:
            params.expected_drift_vt2 = st.number_input(
                "Drift VT2",
                min_value=0.5, max_value=5.0, value=2.0, step=0.25,
                help="Expected VE drift for VT2 (L/min per min)"
            )
        
        st.markdown("---")
        
        # Analyze button
        analyze_btn = st.button("🔬 Analyze Run", type="primary", use_container_width=True)
        
        if analyze_btn and st.session_state.breath_df is not None:
            breath_df = st.session_state.breath_df
            power_df = st.session_state.power_df
            
            # Determine run type and intervals
            if use_manual:
                if run_type_option == "VT1 (Steady State)":
                    run_type = RunType.VT1_STEADY
                    intervals = [Interval(
                        start_time=0,
                        end_time=breath_df['breath_time'].max(),
                        start_idx=0,
                        end_idx=len(breath_df) - 1,
                        interval_num=1
                    )]
                else:
                    run_type = RunType.VT2_INTERVAL
                    intervals = create_manual_intervals(
                        breath_df, num_intervals,
                        interval_duration, recovery_duration
                    )
            else:
                run_type, intervals = detect_intervals(
                    power_df, breath_df,
                    interval_duration, recovery_duration, num_intervals
                )
                
                # Override if user selected specific type
                if run_type_option == "VT1 (Steady State)":
                    run_type = RunType.VT1_STEADY
                elif run_type_option == "VT2 (Intervals)":
                    run_type = RunType.VT2_INTERVAL
            
            # Run analysis
            results = []
            for interval in intervals:
                result = analyze_interval(breath_df, interval, params, run_type)
                results.append(result)
            
            st.session_state.results = results
            st.session_state.intervals = intervals
            st.session_state.run_type = run_type
            st.session_state.selected_interval = None
    
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
        
        # Summary row
        col1, col2, col3, col4 = st.columns(4)
        
        pass_count = sum(1 for r in results if r.status == IntervalStatus.PASS)
        fail_count = sum(1 for r in results if r.status == IntervalStatus.FAIL)
        recovered_count = sum(1 for r in results if r.status == IntervalStatus.RECOVERED)
        
        with col1:
            st.metric("Run Type", run_type.value)
        with col2:
            st.metric("Intervals", len(intervals))
        with col3:
            st.metric("Pass Rate", f"{pass_count}/{len(results)}")
        with col4:
            overall_status = "✅ PASS" if fail_count == 0 else "❌ FAIL"
            st.metric("Overall", overall_status)
        
        # Chart
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
        
        fig = create_main_chart(
            breath_df, results, intervals, params,
            st.session_state.selected_interval
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("### 📋 Interval Results")
        st.markdown("*Click an interval row to zoom the chart*")
        
        # Create results dataframe
        results_data = []
        for r in results:
            status_class = {
                IntervalStatus.PASS: "status-pass",
                IntervalStatus.FAIL: "status-fail",
                IntervalStatus.RECOVERED: "status-recovered"
            }[r.status]
            
            results_data.append({
                "Interval": r.interval.interval_num,
                "Status": r.status.value,
                "VE Drift (L/min/min)": f"{r.ve_drift_rate:.2f}",
                "Peak VE (L/min)": f"{r.peak_ve:.1f}",
                "Peak CUSUM": f"{r.peak_cusum:.1f}",
                "Final CUSUM": f"{r.final_cusum:.1f}",
                "Alarm Time": f"{r.alarm_time:.1f}s" if r.alarm_time else "—"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Display as interactive table
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="medium"
                )
            }
        )
        
        # Detailed cards for failed/recovered intervals
        failed_recovered = [r for r in results if r.status != IntervalStatus.PASS]
        if failed_recovered:
            st.markdown("### ⚠️ Intervals Requiring Attention")
            
            for r in failed_recovered:
                status_color = "#dc3545" if r.status == IntervalStatus.FAIL else "#ffc107"
                st.markdown(f"""
                    <div class="result-card" style="border-left-color: {status_color};">
                        <h4 style="margin:0 0 0.5rem 0;">Interval {r.interval.interval_num} - {r.status.value}</h4>
                        <div class="metric-row">
                            <span class="metric-label">Duration</span>
                            <span class="metric-value">{(r.interval.end_time - r.interval.start_time)/60:.1f} min</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">VE Drift Rate</span>
                            <span class="metric-value">{r.ve_drift_rate:.2f} L/min per min</span>
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
            - Applies rolling median + Savitzky-Golay filtering
            - Uses breath timestamps (not row index) for accurate timing
            
            **2. Interval Detection**
            - Auto-detects work/recovery intervals from power data
            - Or manually specify interval structure
            
            **3. CUSUM Analysis (per interval)**
            - 90s kinetic blanking period
            - Self-calibrates baseline from 90-150s window
            - Detects sustained drift exceeding threshold
            
            **4. Classification**
            - **PASS**: CUSUM never exceeded threshold
            - **RECOVERED**: Alarm triggered but CUSUM returned to acceptable range
            - **FAIL**: Sustained drift above threshold
            """)

if __name__ == "__main__":
    main()
