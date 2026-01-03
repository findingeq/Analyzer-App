"""
Calibration Router for VT Threshold Analyzer API

Handles ML calibration state management and parameter sync.
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..models.schemas import (
    CalibrationStateSchema,
    CalibrationParamsResponse,
    CalibrationUpdateRequest,
    CalibrationUpdateResponse,
    VEApprovalRequest,
    AdvancedParamsRequest,
)
from ..models.enums import RunType, IntervalStatus
from ..services.calibration import (
    CalibrationState,
    update_calibration_from_run,
    apply_ve_threshold_approval,
    get_blended_params,
    enforce_ordinal_constraints,
    DEFAULT_VT1_VE,
    DEFAULT_VT2_VE,
    DEFAULT_PARAMS,
)

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# In-memory storage for calibration states (per user)
# In production, this would be backed by Firebase Storage
_calibration_cache: dict[str, CalibrationState] = {}

# Firebase bucket reference (set by main.py)
_firebase_bucket = None
_firebase_enabled = False


def set_firebase_bucket(bucket, enabled: bool):
    """Set the Firebase bucket reference from main.py."""
    global _firebase_bucket, _firebase_enabled
    _firebase_bucket = bucket
    _firebase_enabled = enabled


def _get_calibration_path(user_id: str) -> str:
    """Get Firebase Storage path for user's calibration data."""
    return f"calibration/{user_id}.json"


def _load_calibration_state(user_id: str) -> CalibrationState:
    """Load calibration state from Firebase or cache, or create new."""
    # Check cache first
    if user_id in _calibration_cache:
        return _calibration_cache[user_id]

    # Try loading from Firebase
    if _firebase_enabled and _firebase_bucket:
        try:
            blob = _firebase_bucket.blob(_get_calibration_path(user_id))
            if blob.exists():
                data = json.loads(blob.download_as_text())
                state = CalibrationState.from_dict(data)
                _calibration_cache[user_id] = state
                return state
        except Exception as e:
            print(f"Warning: Could not load calibration for {user_id}: {e}")

    # Create new default state
    state = CalibrationState()
    _calibration_cache[user_id] = state
    return state


def _save_calibration_state(user_id: str, state: CalibrationState) -> None:
    """Save calibration state to Firebase and cache."""
    # Update cache
    _calibration_cache[user_id] = state

    # Save to Firebase
    if _firebase_enabled and _firebase_bucket:
        try:
            blob = _firebase_bucket.blob(_get_calibration_path(user_id))
            blob.upload_from_string(
                json.dumps(state.to_dict()),
                content_type="application/json"
            )
        except Exception as e:
            print(f"Warning: Could not save calibration for {user_id}: {e}")


@router.get("/params", response_model=CalibrationParamsResponse)
def get_calibration_params(user_id: str = Query(..., description="User/device identifier")):
    """
    Get calibrated parameters for iOS app sync.

    Returns VE thresholds and sigma values for all three domains.
    This is the main endpoint called by the mobile app on startup.
    """
    state = _load_calibration_state(user_id)

    # Get blended sigma values for each domain
    moderate_params = get_blended_params(state, RunType.MODERATE)
    heavy_params = get_blended_params(state, RunType.HEAVY)
    severe_params = get_blended_params(state, RunType.SEVERE)

    return CalibrationParamsResponse(
        vt1_ve=state.vt1_ve.current_value,
        vt2_ve=state.vt2_ve.current_value,
        sigma_pct_moderate=moderate_params['sigma_pct'],
        sigma_pct_heavy=heavy_params['sigma_pct'],
        sigma_pct_severe=severe_params['sigma_pct'],
        expected_drift_moderate=moderate_params['expected_drift_pct'],
        expected_drift_heavy=heavy_params['expected_drift_pct'],
        expected_drift_severe=severe_params['expected_drift_pct'],
        max_drift_moderate=moderate_params['max_drift_pct'],
        max_drift_heavy=heavy_params['max_drift_pct'],
        max_drift_severe=severe_params['max_drift_pct'],
        enabled=state.enabled,
        last_updated=state.last_updated.isoformat() if state.last_updated else None
    )


@router.get("/state", response_model=CalibrationStateSchema)
def get_calibration_state(user_id: str = Query(..., description="User/device identifier")):
    """
    Get full calibration state for a user.

    Returns complete NIG posteriors and all metadata.
    Used by the web app for detailed calibration view.
    """
    state = _load_calibration_state(user_id)
    return CalibrationStateSchema(**state.to_dict())


@router.post("/update", response_model=CalibrationUpdateResponse)
def update_calibration(request: CalibrationUpdateRequest):
    """
    Update calibration from analysis results using majority-based logic.

    A run only counts toward calibration if:
    1. More than 50% of qualifying intervals (≥6 min) share the same classification
    2. The majority classification matches domain expectations

    Each run counts as ONE calibration sample regardless of number of intervals.
    Returns a VE prompt if threshold change >= 1 L/min.
    """
    state = _load_calibration_state(request.user_id)

    # Use majority-based calibration logic
    state, ve_prompt = update_calibration_from_run(
        state=state,
        run_type=request.run_type,
        interval_results=request.interval_results
    )

    # Enforce ordinal constraints
    state = enforce_ordinal_constraints(state)

    # Save updated state
    _save_calibration_state(request.user_id, state)

    return CalibrationUpdateResponse(
        success=True,
        run_count=state.get_run_count(request.run_type),
        ve_prompt=ve_prompt
    )


@router.post("/approve-ve")
def approve_ve_threshold(request: VEApprovalRequest):
    """
    Apply user's approval/rejection of VE threshold change.

    Called when user responds to the VE threshold prompt.
    After approval or rejection, the anchor is reset to start fresh.
    """
    if request.threshold not in ('vt1', 'vt2'):
        raise HTTPException(status_code=400, detail="threshold must be 'vt1' or 'vt2'")

    state = _load_calibration_state(request.user_id)
    state = apply_ve_threshold_approval(
        state,
        request.threshold,
        request.approved,
        request.proposed_value
    )

    # Save updated state
    _save_calibration_state(request.user_id, state)

    return {
        "success": True,
        "approved": request.approved,
        "new_value": (
            state.vt1_ve.current_value if request.threshold == 'vt1'
            else state.vt2_ve.current_value
        )
    }


@router.post("/reset")
def reset_calibration(user_id: str = Query(..., description="User/device identifier")):
    """
    Reset calibration to defaults.

    Creates a fresh calibration state with default parameters.
    """
    state = CalibrationState()
    _save_calibration_state(user_id, state)

    return {
        "success": True,
        "message": "Calibration reset to defaults"
    }


@router.get("/blended-params")
def get_blended_params_endpoint(
    user_id: str = Query(..., description="User/device identifier"),
    run_type: RunType = Query(..., description="Run type to get params for")
):
    """
    Get blended parameters for a specific run type.

    Returns calibrated values blended with defaults based on run count.
    Used by analysis to get the current parameters to use.
    """
    state = _load_calibration_state(user_id)
    params = get_blended_params(state, run_type)

    return {
        "run_type": run_type.value,
        **params
    }


@router.post("/set-ve-threshold")
def set_ve_threshold_manual(
    user_id: str = Query(..., description="User/device identifier"),
    threshold: str = Query(..., description="'vt1' or 'vt2'"),
    value: float = Query(..., description="New threshold value in L/min")
):
    """
    Manually set a VE threshold value.

    Called when user manually changes threshold in the UI.
    This becomes the new anchor for calibration, resetting the posterior.
    """
    if threshold not in ('vt1', 'vt2'):
        raise HTTPException(status_code=400, detail="threshold must be 'vt1' or 'vt2'")

    state = _load_calibration_state(user_id)

    if threshold == 'vt1':
        state.vt1_ve.current_value = value
        state.vt1_ve.reset_to_anchor()  # Reset posterior to new anchor
    else:
        state.vt2_ve.current_value = value
        state.vt2_ve.reset_to_anchor()  # Reset posterior to new anchor

    # Enforce VT1 < VT2 constraint
    state = enforce_ordinal_constraints(state)

    _save_calibration_state(user_id, state)

    return {
        "success": True,
        "threshold": threshold,
        "value": state.vt1_ve.current_value if threshold == 'vt1' else state.vt2_ve.current_value
    }


@router.post("/toggle")
def toggle_calibration(
    user_id: str = Query(..., description="User/device identifier"),
    enabled: bool = Query(..., description="Enable or disable calibration")
):
    """
    Toggle calibration on or off.

    When disabled:
    - Learned data is preserved in the cloud
    - System returns default parameters instead of calibrated values
    - No new calibration updates occur during analysis

    When re-enabled:
    - Previously learned values are restored as the new baseline
    - Calibration updates resume
    """
    state = _load_calibration_state(user_id)
    state.enabled = enabled
    _save_calibration_state(user_id, state)

    return {
        "success": True,
        "enabled": enabled,
        "message": f"Calibration {'enabled' if enabled else 'disabled'}"
    }


@router.post("/set-advanced-params")
def set_advanced_params(request: AdvancedParamsRequest):
    """
    Manually set advanced calibration parameters.

    Called when user manually changes advanced params in the UI.
    This directly sets the NIG posterior mu values for each domain.
    """
    state = _load_calibration_state(request.user_id)

    # Update Moderate domain (VT1)
    state.moderate.expected_drift.mu = request.expected_drift_vt1
    state.moderate.sigma.mu = request.sigma_pct_vt1

    # Update Heavy domain (VT2)
    state.heavy.expected_drift.mu = request.expected_drift_vt2
    state.heavy.max_drift.mu = request.max_drift_vt2
    state.heavy.sigma.mu = request.sigma_pct_vt2

    # Update Severe domain (uses same values as Heavy for VT2)
    state.severe.expected_drift.mu = request.expected_drift_vt2
    state.severe.max_drift.mu = request.max_drift_vt2
    state.severe.sigma.mu = request.sigma_pct_vt2

    # Note: H multipliers are not stored in calibration state
    # They are passed directly in AnalysisParams from the frontend

    # Enforce ordinal constraints
    state = enforce_ordinal_constraints(state)

    _save_calibration_state(request.user_id, state)

    return {
        "success": True,
        "message": "Advanced parameters updated"
    }
