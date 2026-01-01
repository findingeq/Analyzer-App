"""
CSV Parsing Service for VT Threshold Analyzer

Supports two CSV formats:
1. VitalPro format - Official Tymewear device export
2. iOS App format - Custom iOS app CSV with metadata headers
"""

from typing import Tuple, Optional, Dict, Any
from io import StringIO
import pandas as pd

from ..models.enums import RunType


def detect_csv_format(csv_content: str) -> str:
    """
    Detect whether the CSV is iOS app format or VitalPro format.

    Args:
        csv_content: Raw CSV content as string

    Returns:
        'ios', 'vitalpro', or 'unknown'
    """
    # iOS format starts with # comments containing metadata
    if csv_content.startswith('# Date:') or '# Run Type:' in csv_content[:500]:
        return 'ios'
    # VitalPro format has 'Breath by breath time' column
    elif 'Breath by breath time' in csv_content:
        return 'vitalpro'
    else:
        return 'unknown'


def parse_vitalpro_csv(csv_content: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Parse VitalPro CSV and extract breath-by-breath data with metadata.

    Args:
        csv_content: Raw CSV content as string

    Returns:
        Tuple of (breath_df, metadata, power_df)
        - breath_df: DataFrame with breath-by-breath data
        - metadata: Dict with file metadata
        - power_df: DataFrame with time-aligned power data
    """
    lines = csv_content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # Find header row (contains 'Time' and 'Breath by breath time')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Time' in line and 'Breath by breath time' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find data header row in VitalPro CSV")

    # Extract metadata from rows before header
    metadata = {}
    for i in range(header_idx):
        parts = lines[i].split(',')
        if len(parts) >= 2 and parts[0].strip():
            metadata[parts[0].strip()] = parts[1].strip()

    # Parse data starting from header
    df = pd.read_csv(StringIO(csv_content), skiprows=header_idx, skip_blank_lines=True)

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


def parse_ios_csv(csv_content: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Parse iOS app CSV format and extract breath-by-breath data with metadata.

    iOS format has:
    - Header comments starting with # containing metadata
    - Data columns: timestamp, elapsed_sec, VE, HR, [phase, speed]

    Args:
        csv_content: Raw CSV content as string

    Returns:
        Tuple of (breath_df, metadata, power_df, run_params)
        - breath_df: DataFrame with breath-by-breath data
        - metadata: Dict with parsed metadata
        - power_df: Empty DataFrame (no power data in iOS format)
        - run_params: Dict with run parameters (run_type, intervals, durations, speeds)
    """
    lines = csv_content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # Parse header metadata
    metadata = {}
    run_params: Dict[str, Any] = {}

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
        elif line.strip() and not line.startswith('#'):
            # Found data header row
            break

    # Parse data using pandas
    df = pd.read_csv(StringIO(csv_content), comment='#', skip_blank_lines=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Map iOS columns to internal format
    result = pd.DataFrame()

    if 'elapsed_sec' in df.columns:
        result['breath_time'] = pd.to_numeric(df['elapsed_sec'], errors='coerce')
    elif 'elapsed_s' in df.columns:
        result['breath_time'] = pd.to_numeric(df['elapsed_s'], errors='coerce')
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


def parse_csv_auto(csv_content: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Auto-detect CSV format and parse accordingly.

    Args:
        csv_content: Raw CSV content as string

    Returns:
        Tuple of (breath_df, metadata, power_df, run_params)
        - breath_df: DataFrame with breath-by-breath data
        - metadata: Dict with file metadata
        - power_df: DataFrame with power data (empty for iOS format)
        - run_params: Dict with run parameters (only for iOS format, None for VitalPro)
    """
    csv_format = detect_csv_format(csv_content)

    if csv_format == 'ios':
        breath_df, metadata, power_df, run_params = parse_ios_csv(csv_content)
        return breath_df, metadata, power_df, run_params
    elif csv_format == 'vitalpro':
        breath_df, metadata, power_df = parse_vitalpro_csv(csv_content)
        return breath_df, metadata, power_df, None
    else:
        raise ValueError(f"Unknown CSV format. Expected 'ios' or 'vitalpro' format.")
