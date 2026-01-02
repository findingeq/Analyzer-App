"""
Regression Service for VT Threshold Analyzer

Implements robust regression models for:
1. Single slope fitting (Huber regression)
2. Phase III onset detection (hinge model)
3. Second hinge detection (slope change after Phase III)
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize

from ..models.params import AnalysisParams


def _huber_loss_linear(params, t_arr, ve_arr, delta=5.0):
    """Huber loss for robust linear regression."""
    intercept, slope = params
    pred = intercept + slope * t_arr
    residuals = ve_arr - pred
    abs_res = np.abs(residuals)
    loss = np.where(
        abs_res <= delta,
        0.5 * residuals**2,
        delta * (abs_res - 0.5 * delta)
    )
    return np.sum(loss)


def _huber_loss_hinge(params_opt, t_arr, ve_arr, delta=5.0):
    """Huber loss for piecewise linear hinge model."""
    tau, b0, b1, b2 = params_opt
    pred = b0 + b1 * t_arr + b2 * np.maximum(0, t_arr - tau)
    residuals = ve_arr - pred
    abs_res = np.abs(residuals)
    loss = np.where(
        abs_res <= delta,
        0.5 * residuals**2,
        delta * (abs_res - 0.5 * delta)
    )
    return np.sum(loss)


def fit_single_slope(t: np.ndarray, ve: np.ndarray) -> Tuple[float, float]:
    """
    Fit a single linear slope using robust Huber regression.

    Used for:
    - Phase III slope estimation after calibration window
    - Cumulative drift across intervals

    Args:
        t: Time values (can be seconds or minutes)
        ve: VE values

    Returns:
        Tuple of (slope, intercept)
    """
    if len(t) < 2:
        return 0.0, np.mean(ve) if len(ve) > 0 else 0.0

    # Initial guess from simple linear fit
    slope_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
    intercept_init = ve[0] - slope_init * t[0]

    try:
        result = minimize(
            _huber_loss_linear,
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


def fit_robust_hinge(
    t: np.ndarray,
    ve: np.ndarray,
    params: AnalysisParams,
    tau_min_override: float = None,
    tau_max_override: float = None,
    tau_default_override: float = None
) -> Tuple[float, float, float, float, float, bool]:
    """
    Fit a robust hinge model to detect Phase III onset.

    Model: VE(t) = beta0 + beta1*t + beta2*max(0, t - tau)

    Where:
    - tau: The Phase III onset breakpoint
    - beta1: Slope of Phase II (initial ramp)
    - beta1 + beta2: Slope of Phase III (drift)

    Uses Huber loss for robustness to outliers.
    Constrains breakpoint detection to specified bounds.
    If detection fails or breakpoint is at bounds, returns default Phase III onset.

    Args:
        t: Time values (seconds relative to interval start)
        ve: VE values (L/min)
        params: Analysis parameters
        tau_min_override: Optional minimum tau bound (overrides params)
        tau_max_override: Optional maximum tau bound (overrides params)
        tau_default_override: Optional default tau if detection fails (overrides params)

    Returns:
        Tuple of (tau, beta0, beta1, beta2, loss, detection_succeeded)
    """
    # Determine tau bounds and default
    tau_min_param = tau_min_override if tau_min_override is not None else params.phase3_min_time
    tau_max_param = tau_max_override if tau_max_override is not None else params.phase3_max_time
    tau_default = tau_default_override if tau_default_override is not None else params.phase3_default

    # Use user override if provided
    if params.phase3_onset_override is not None:
        tau = params.phase3_onset_override
        mask_before = t < tau
        mask_after = t >= tau

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            t_before = t[mask_before]
            ve_before = ve[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            t_after = t[mask_after]
            ve_after = ve[mask_after]
            phase3_slope = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = phase3_slope - b1
        else:
            b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
            b0 = ve[0] - b1 * t[0]
            b2 = 0

        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, True

    # Set tau bounds
    tau_min = tau_min_param
    tau_max = tau_max_param

    # Ensure bounds are within data range
    t_min = t[0]
    t_max = t[-1]
    tau_min = max(tau_min, t_min + 10)
    tau_max = min(tau_max, t_max - 30)

    # Check if we have enough data for constrained detection
    if tau_max <= tau_min:
        tau = tau_default
        b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0 = ve[0] - b1 * t[0]
        b2 = 0
        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False

    # Initial guess: tau at midpoint of allowed range
    tau_init = (tau_min + tau_max) / 2

    if len(t) >= 2:
        b1_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0_init = ve[0] - b1_init * t[0]
    else:
        b0_init = np.mean(ve)
        b1_init = 0
    b2_init = 0

    initial_params = [tau_init, b0_init, b1_init, b2_init]

    bounds = [
        (tau_min, tau_max),  # tau constrained
        (None, None),  # b0
        (None, None),  # b1
        (None, None),  # b2
    ]

    detection_succeeded = True

    try:
        result = minimize(
            _huber_loss_hinge,
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
            tau = tau_default
    except Exception:
        detection_succeeded = False
        tau = tau_default
        b1 = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0 = ve[0] - b1 * t[0]
        b2 = 0
        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)

    return tau, b0, b1, b2, final_loss, detection_succeeded


def fit_second_hinge(
    t: np.ndarray,
    ve: np.ndarray,
    phase3_onset: float,
    interval_end: float
) -> Tuple[float, float, float, float, float, bool]:
    """
    Fit a second robust hinge model to detect slope change after Phase III onset.

    Model: VE(t) = beta0 + beta1*t + beta2*max(0, t - tau)

    Constraint window: (Phase III onset + 120s) to (interval end - 120s)
    If window is invalid, places hinge at midpoint between Phase III onset and interval end.
    If detection fails, falls back to midpoint of constraint window.

    Args:
        t: Time values (seconds relative to interval start)
        ve: VE values (L/min)
        phase3_onset: Detected Phase III onset time (relative to interval)
        interval_end: End time of interval (relative to interval start)

    Returns:
        Tuple of (tau2, beta0, beta1, beta2, loss, detection_succeeded)
    """
    # Define constraint window: 2 min after Phase III onset to 2 min before interval end
    tau_min = phase3_onset + 120.0
    tau_max = interval_end - 120.0

    # Calculate midpoint between Phase III onset and interval end (fallback position)
    midpoint_fallback = (phase3_onset + interval_end) / 2.0

    def _fit_coefficients_for_tau(tau_val, t_arr, ve_arr):
        """Helper to compute coefficients for a fixed tau."""
        mask_before = t_arr < tau_val
        mask_after = t_arr >= tau_val

        if np.sum(mask_before) >= 2 and np.sum(mask_after) >= 2:
            t_before = t_arr[mask_before]
            ve_before = ve_arr[mask_before]
            b1 = (ve_before[-1] - ve_before[0]) / (t_before[-1] - t_before[0]) if (t_before[-1] - t_before[0]) > 0 else 0
            b0 = ve_before[0] - b1 * t_before[0]

            t_after = t_arr[mask_after]
            ve_after = ve_arr[mask_after]
            slope_after = (ve_after[-1] - ve_after[0]) / (t_after[-1] - t_after[0]) if (t_after[-1] - t_after[0]) > 0 else 0
            b2 = slope_after - b1
        else:
            b1 = (ve_arr[-1] - ve_arr[0]) / (t_arr[-1] - t_arr[0]) if (t_arr[-1] - t_arr[0]) > 0 else 0
            b0 = ve_arr[0] - b1 * t_arr[0]
            b2 = 0

        return b0, b1, b2

    # Check if constraint window is valid
    if tau_max <= tau_min:
        tau = midpoint_fallback
        b0, b1, b2 = _fit_coefficients_for_tau(tau, t, ve)
        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False

    # Constraint window midpoint (for fallback if detection fails)
    window_midpoint = (tau_min + tau_max) / 2.0

    # Ensure bounds are within data range
    t_min = t[0]
    t_max = t[-1]
    tau_min = max(tau_min, t_min + 10)
    tau_max = min(tau_max, t_max - 10)

    if tau_max <= tau_min:
        tau = window_midpoint
        b0, b1, b2 = _fit_coefficients_for_tau(tau, t, ve)
        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)
        return tau, b0, b1, b2, final_loss, False

    # Initial guess
    tau_init = (tau_min + tau_max) / 2

    if len(t) >= 2:
        b1_init = (ve[-1] - ve[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        b0_init = ve[0] - b1_init * t[0]
    else:
        b0_init = np.mean(ve)
        b1_init = 0
    b2_init = 0

    initial_params = [tau_init, b0_init, b1_init, b2_init]

    bounds = [
        (tau_min, tau_max),
        (None, None),
        (None, None),
        (None, None),
    ]

    detection_succeeded = True

    try:
        result = minimize(
            _huber_loss_hinge,
            initial_params,
            args=(t, ve),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        tau, b0, b1, b2 = result.x
        final_loss = result.fun

        if abs(tau - tau_min) < 1.0 or abs(tau - tau_max) < 1.0:
            detection_succeeded = False
            tau = window_midpoint
            b0, b1, b2 = _fit_coefficients_for_tau(tau, t, ve)
            final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)

    except Exception:
        detection_succeeded = False
        tau = window_midpoint
        b0, b1, b2 = _fit_coefficients_for_tau(tau, t, ve)
        final_loss = _huber_loss_hinge([tau, b0, b1, b2], t, ve)

    return tau, b0, b1, b2, final_loss, detection_succeeded
