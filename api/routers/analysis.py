"""
Analysis Router for VT Threshold Analyzer API

Handles CUSUM analysis of respiratory data.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BreathData,
    Interval,
    IntervalResult,
)
from ..models.params import AnalysisParams
from ..models.enums import RunType
from ..services.csv_parser import parse_csv_auto
from ..services.interval_detector import create_intervals_from_params
from ..services.signal_filter import apply_hybrid_filtering
from ..services.cusum_analyzer import analyze_interval_segmented, analyze_interval_ceiling
from ..services.cumulative_drift import compute_cumulative_drift

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/run", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest):
    """
    Run full CUSUM analysis on respiratory data.

    Parses the CSV, creates intervals based on the specified parameters,
    and performs CUSUM analysis on each interval.

    Returns analysis results including classification, drift metrics,
    and chart data for visualization.
    """
    try:
        # Parse CSV
        breath_df, metadata, power_df, run_params = parse_csv_auto(request.csv_content)

        if len(breath_df) == 0:
            raise HTTPException(status_code=400, detail="No breath data found in CSV")

        # Get analysis parameters (use defaults if not provided)
        params = request.params or AnalysisParams()

        # Create intervals from parameters
        intervals = create_intervals_from_params(
            breath_df,
            request.run_type,
            request.num_intervals,
            request.interval_duration_min,
            request.recovery_duration_min
        )

        if len(intervals) == 0:
            raise HTTPException(status_code=400, detail="No valid intervals created")

        # Determine analysis method for each interval
        results: List[IntervalResult] = []
        detected_phase3_onset = None

        for i, interval in enumerate(intervals):
            # Get speed for this interval
            speed = None

            # First try: from iOS metadata (comma-separated speeds)
            if run_params and 'speeds' in run_params:
                speeds = run_params['speeds']
                if i < len(speeds):
                    speed = speeds[i]

            # Second try: compute from per-breath speed column if available
            if speed is None and 'speed' in breath_df.columns:
                interval_mask = (breath_df['breath_time'] >= interval.start_time) & \
                               (breath_df['breath_time'] <= interval.end_time)
                interval_speeds = breath_df.loc[interval_mask, 'speed'].dropna()
                if len(interval_speeds) > 0:
                    speed = float(interval_speeds.mean())

            # Determine whether to use segmented or ceiling-based analysis
            interval_duration_min = (interval.end_time - interval.start_time) / 60.0

            use_ceiling = False
            if params.use_thresholds_for_all:
                # User explicitly requested ceiling-based for all
                use_ceiling = True
            elif interval_duration_min < 6.0:
                # Short intervals use ceiling-based by default
                use_ceiling = True

            # Run appropriate analysis
            if use_ceiling:
                result = analyze_interval_ceiling(
                    breath_df, interval, params, request.run_type, speed
                )
            else:
                result = analyze_interval_segmented(
                    breath_df, interval, params, request.run_type, speed
                )

                # Capture detected Phase III onset from first segmented interval
                if detected_phase3_onset is None and result.phase3_onset_rel is not None:
                    detected_phase3_onset = result.phase3_onset_rel

            results.append(result)

        # Compute cumulative drift for multi-interval VT2/SEVERE runs
        cumulative_drift = None
        if request.run_type in (RunType.VT2_INTERVAL, RunType.SEVERE) and len(results) >= 2:
            cumulative_drift = compute_cumulative_drift(results)

        # Prepare breath data for full chart
        ve_raw = breath_df['ve_raw'].values
        breath_times = breath_df['breath_time'].values
        ve_median, bin_times, ve_binned = apply_hybrid_filtering(ve_raw, breath_times, params)

        # Get HR data if available
        hr_values = None
        if 'hr' in breath_df.columns:
            hr_values = breath_df['hr'].tolist()

        breath_data = BreathData(
            times=breath_times.tolist(),
            ve_median=ve_median.tolist(),
            bin_times=bin_times.tolist(),
            ve_binned=ve_binned.tolist(),
            hr=hr_values
        )

        # Convert Interval objects to dicts for response
        interval_dicts = [
            Interval(
                start_time=intv.start_time,
                end_time=intv.end_time,
                start_idx=intv.start_idx,
                end_idx=intv.end_idx,
                interval_num=intv.interval_num
            )
            for intv in intervals
        ]

        return AnalysisResponse(
            success=True,
            run_type=request.run_type,
            intervals=interval_dicts,
            results=results,
            cumulative_drift=cumulative_drift,
            breath_data=breath_data,
            detected_phase3_onset=detected_phase3_onset,
            error=None
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
