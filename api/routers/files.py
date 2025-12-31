"""
File Parsing Router for VT Threshold Analyzer API

Handles CSV file parsing and format detection.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    ParseCSVRequest,
    ParseCSVResponse,
    DetectIntervalsRequest,
    DetectIntervalsResponse,
)
from ..models.enums import RunType
from ..services.csv_parser import (
    detect_csv_format,
    parse_csv_auto,
)
from ..services.interval_detector import detect_intervals

router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/parse", response_model=ParseCSVResponse)
def parse_csv(request: ParseCSVRequest):
    """
    Parse a CSV file and return metadata about its contents.

    Auto-detects the CSV format (iOS app or VitalPro) and extracts
    relevant metadata for configuring the analysis.
    """
    try:
        # Detect format if not specified
        csv_format = request.format or detect_csv_format(request.csv_content)

        if csv_format == "unknown":
            raise HTTPException(
                status_code=400,
                detail="Unknown CSV format. Expected iOS app or VitalPro format."
            )

        # Parse the CSV
        breath_df, metadata, power_df, run_params = parse_csv_auto(request.csv_content)

        # Calculate basic stats
        total_breaths = len(breath_df)
        duration_seconds = float(breath_df['breath_time'].max()) if len(breath_df) > 0 else 0
        has_power_data = len(power_df) > 0 if power_df is not None else False

        # Extract detected parameters (from iOS metadata)
        detected_run_type = None
        detected_intervals = None
        detected_interval_duration = None
        detected_recovery_duration = None
        detected_vt1_threshold = None
        detected_vt2_threshold = None
        detected_speeds = None

        if run_params:
            if 'run_type' in run_params:
                detected_run_type = run_params['run_type'].value
            if 'num_intervals' in run_params:
                detected_intervals = run_params['num_intervals']
            if 'interval_duration' in run_params:
                detected_interval_duration = run_params['interval_duration']
            if 'recovery_duration' in run_params:
                detected_recovery_duration = run_params['recovery_duration']
            if 'vt1_threshold' in run_params:
                detected_vt1_threshold = run_params['vt1_threshold']
            if 'vt2_threshold' in run_params:
                detected_vt2_threshold = run_params['vt2_threshold']
            if 'speeds' in run_params:
                detected_speeds = run_params['speeds']

        return ParseCSVResponse(
            success=True,
            format=csv_format,
            total_breaths=total_breaths,
            duration_seconds=duration_seconds,
            has_power_data=has_power_data,
            detected_run_type=detected_run_type,
            detected_intervals=detected_intervals,
            detected_interval_duration=detected_interval_duration,
            detected_recovery_duration=detected_recovery_duration,
            detected_vt1_threshold=detected_vt1_threshold,
            detected_vt2_threshold=detected_vt2_threshold,
            detected_speeds=detected_speeds,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")


@router.post("/detect-intervals", response_model=DetectIntervalsResponse)
def detect_intervals_endpoint(request: DetectIntervalsRequest):
    """
    Detect intervals from power data in a CSV file.

    Uses K-Means clustering on power data to identify work and recovery periods.
    Returns the detected run type, number of intervals, and durations.
    """
    try:
        # Parse the CSV first
        breath_df, metadata, power_df, run_params = parse_csv_auto(request.csv_content)

        # If we have run params from iOS metadata, use those
        if run_params and 'run_type' in run_params:
            run_type = run_params['run_type']
            num_intervals = run_params.get('num_intervals', 1)
            interval_duration = run_params.get('interval_duration', breath_df['breath_time'].max() / 60.0)
            recovery_duration = run_params.get('recovery_duration', 0.0)
            detection_method = "metadata"
        elif power_df is not None and len(power_df) > 0:
            # Detect from power data
            run_type, num_intervals, interval_duration, recovery_duration = detect_intervals(
                power_df, breath_df
            )
            detection_method = "power_kmeans"
        else:
            # No power data, assume VT1 steady state
            run_type = RunType.VT1_STEADY
            num_intervals = 1
            interval_duration = breath_df['breath_time'].max() / 60.0
            recovery_duration = 0.0
            detection_method = "default"

        total_duration_min = breath_df['breath_time'].max() / 60.0

        return DetectIntervalsResponse(
            run_type=run_type,
            num_intervals=num_intervals,
            interval_duration_min=interval_duration,
            recovery_duration_min=recovery_duration,
            total_duration_min=total_duration_min,
            detection_method=detection_method,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect intervals: {str(e)}")
