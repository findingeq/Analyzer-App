"""
ML Calibration Service for VT Threshold Analyzer

Implements Normal-Inverse-Gamma (NIG) with Forgetting Factor for
automatically adjusting VT parameters based on accumulated run data.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import math

from ..models.enums import RunType, IntervalStatus


# ============================================================================
# NIG Posterior Data Structures
# ============================================================================

@dataclass
class NIGPosterior:
    """
    Normal-Inverse-Gamma posterior for a single parameter.

    The NIG conjugate prior models uncertainty about both the mean (mu)
    and variance (sigma^2) of a normally distributed parameter.
    """
    mu: float = 0.0       # Posterior mean estimate
    kappa: float = 1.0    # Precision of mean (higher = more confident)
    alpha: float = 2.0    # Shape parameter for variance
    beta: float = 1.0     # Scale parameter for variance
    n_obs: int = 0        # Total observation count

    def get_point_estimate(self) -> float:
        """Return the posterior mean as point estimate."""
        return self.mu

    def get_variance(self) -> float:
        """Return the posterior predictive variance."""
        if self.alpha <= 1:
            return float('inf')
        return self.beta / (self.alpha - 1) * (1 + 1/self.kappa)

    def get_std(self) -> float:
        """Return the posterior predictive standard deviation."""
        var = self.get_variance()
        return math.sqrt(var) if var != float('inf') else float('inf')

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'alpha': self.alpha,
            'beta': self.beta,
            'n_obs': self.n_obs
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NIGPosterior':
        """Deserialize from dictionary."""
        return cls(
            mu=d.get('mu', 0.0),
            kappa=d.get('kappa', 1.0),
            alpha=d.get('alpha', 2.0),
            beta=d.get('beta', 1.0),
            n_obs=d.get('n_obs', 0)
        )


@dataclass
class AnchoredNIGPosterior:
    """
    NIG posterior with decaying anchor for stability.

    The anchor provides early stability while gradually fading to let
    observations fully determine the parameter value over time.

    - anchor_value: Starting point (default or user-set value)
    - anchor_kappa: Decaying weight of anchor (starts at 4.0, decays with λ)
    - posterior: Observation-derived NIG posterior
    """
    anchor_value: float = 0.0      # The anchor point (default or user-set)
    anchor_kappa: float = 4.0      # Decaying anchor weight
    posterior: NIGPosterior = field(default_factory=NIGPosterior)

    def get_anchored_mean(self) -> float:
        """
        Get the effective mean considering decaying anchor.

        Formula: (anchor_kappa × anchor_value + obs_kappa × obs_mu) / (anchor_kappa + obs_kappa)
        """
        obs_kappa = self.posterior.kappa
        obs_mu = self.posterior.mu

        if self.anchor_kappa <= 0 and obs_kappa <= 0:
            return self.anchor_value

        total_kappa = self.anchor_kappa + obs_kappa
        return (self.anchor_kappa * self.anchor_value + obs_kappa * obs_mu) / total_kappa

    def get_point_estimate(self) -> float:
        """Return the anchored mean as point estimate."""
        return self.get_anchored_mean()

    def reset_anchor(self, new_anchor_value: float, anchor_kappa: float = 4.0) -> None:
        """
        Reset to a new anchor (e.g., when user manually sets a value).

        Clears observation posterior and starts fresh from the new anchor.
        """
        self.anchor_value = new_anchor_value
        self.anchor_kappa = anchor_kappa
        self.posterior = NIGPosterior(mu=new_anchor_value, kappa=0.0, alpha=2.0, beta=1.0, n_obs=0)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'anchor_value': self.anchor_value,
            'anchor_kappa': self.anchor_kappa,
            'posterior': self.posterior.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AnchoredNIGPosterior':
        """Deserialize from dictionary."""
        return cls(
            anchor_value=d.get('anchor_value', 0.0),
            anchor_kappa=d.get('anchor_kappa', 4.0),
            posterior=NIGPosterior.from_dict(d.get('posterior', {}))
        )


@dataclass
class DomainPosterior:
    """
    All posteriors for a single intensity domain.

    Each domain (Moderate/Heavy/Severe) has its own set of parameters.
    Uses AnchoredNIGPosterior for stability with eventual full adaptation.
    """
    expected_drift: AnchoredNIGPosterior = field(default_factory=AnchoredNIGPosterior)
    max_drift: AnchoredNIGPosterior = field(default_factory=AnchoredNIGPosterior)
    sigma: AnchoredNIGPosterior = field(default_factory=AnchoredNIGPosterior)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'expected_drift': self.expected_drift.to_dict(),
            'max_drift': self.max_drift.to_dict(),
            'sigma': self.sigma.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DomainPosterior':
        """Deserialize from dictionary."""
        return cls(
            expected_drift=AnchoredNIGPosterior.from_dict(d.get('expected_drift', {})),
            max_drift=AnchoredNIGPosterior.from_dict(d.get('max_drift', {})),
            sigma=AnchoredNIGPosterior.from_dict(d.get('sigma', {}))
        )


@dataclass
class VEThresholdState:
    """
    State for a VE threshold (VT1 or VT2).

    Uses decaying anchor approach:
    - current_value auto-updates to anchored posterior mean
    - anchor_kappa decays over time, eventually letting observations dominate
    - Manual override resets anchor with fresh anchor_kappa
    """
    current_value: float = 60.0  # Current threshold (auto-updated)
    anchor_value: float = 60.0   # Anchor point (default or user-set)
    anchor_kappa: float = 4.0    # Decaying anchor weight
    posterior: NIGPosterior = field(default_factory=NIGPosterior)

    def get_anchored_mean(self) -> float:
        """Get the effective mean considering decaying anchor."""
        obs_kappa = self.posterior.kappa
        obs_mu = self.posterior.mu

        if self.anchor_kappa <= 0 and obs_kappa <= 0:
            return self.anchor_value

        total_kappa = self.anchor_kappa + obs_kappa
        return (self.anchor_kappa * self.anchor_value + obs_kappa * obs_mu) / total_kappa

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'current_value': self.current_value,
            'anchor_value': self.anchor_value,
            'anchor_kappa': self.anchor_kappa,
            'posterior': self.posterior.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'VEThresholdState':
        """Deserialize from dictionary."""
        # Handle migration from old format (no anchor_value field)
        anchor_value = d.get('anchor_value', d.get('current_value', 60.0))
        return cls(
            current_value=d.get('current_value', 60.0),
            anchor_value=anchor_value,
            anchor_kappa=d.get('anchor_kappa', 4.0),
            posterior=NIGPosterior.from_dict(d.get('posterior', {}))
        )

    def reset_anchor(self, new_value: float, anchor_kappa: float = 4.0) -> None:
        """
        Reset to a new anchor (e.g., when user manually sets a value).

        Sets both current_value and anchor_value, clears posterior.
        """
        self.current_value = new_value
        self.anchor_value = new_value
        self.anchor_kappa = anchor_kappa
        self.posterior = NIGPosterior(mu=new_value, kappa=0.0, alpha=2.0, beta=1.0, n_obs=0)


@dataclass
class CalibrationState:
    """
    Complete calibration state for a user.

    Contains per-domain NIG posteriors and global VE thresholds.
    """
    # Per-domain posteriors
    moderate: DomainPosterior = field(default_factory=DomainPosterior)
    heavy: DomainPosterior = field(default_factory=DomainPosterior)
    severe: DomainPosterior = field(default_factory=DomainPosterior)

    # Global VE thresholds
    vt1_ve: VEThresholdState = field(default_factory=lambda: VEThresholdState(
        current_value=60.0
    ))
    vt2_ve: VEThresholdState = field(default_factory=lambda: VEThresholdState(
        current_value=80.0
    ))

    # Calibration enabled flag
    enabled: bool = True

    # Metadata
    last_updated: Optional[datetime] = None
    run_counts: Dict[str, int] = field(default_factory=lambda: {
        'moderate': 0, 'heavy': 0, 'severe': 0
    })

    def get_domain_posterior(self, run_type: RunType) -> DomainPosterior:
        """Get the posterior for a specific run type."""
        if run_type == RunType.MODERATE:
            return self.moderate
        elif run_type == RunType.HEAVY:
            return self.heavy
        else:
            return self.severe

    def get_run_count(self, run_type: RunType) -> int:
        """Get the qualifying run count for a domain."""
        key = run_type.value.lower()
        return self.run_counts.get(key, 0)

    def increment_run_count(self, run_type: RunType) -> None:
        """Increment the qualifying run count for a domain."""
        key = run_type.value.lower()
        self.run_counts[key] = self.run_counts.get(key, 0) + 1

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            'moderate': self.moderate.to_dict(),
            'heavy': self.heavy.to_dict(),
            'severe': self.severe.to_dict(),
            'vt1_ve': self.vt1_ve.to_dict(),
            'vt2_ve': self.vt2_ve.to_dict(),
            'enabled': self.enabled,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'run_counts': self.run_counts
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationState':
        """Deserialize from dictionary."""
        last_updated = None
        if d.get('last_updated'):
            last_updated = datetime.fromisoformat(d['last_updated'])

        return cls(
            moderate=DomainPosterior.from_dict(d.get('moderate', {})),
            heavy=DomainPosterior.from_dict(d.get('heavy', {})),
            severe=DomainPosterior.from_dict(d.get('severe', {})),
            vt1_ve=VEThresholdState.from_dict(d.get('vt1_ve', {'current_value': 60.0})),
            vt2_ve=VEThresholdState.from_dict(d.get('vt2_ve', {'current_value': 80.0})),
            enabled=d.get('enabled', True),
            last_updated=last_updated,
            run_counts=d.get('run_counts', {'moderate': 0, 'heavy': 0, 'severe': 0})
        )


# ============================================================================
# Default Parameter Values
# ============================================================================

# These MUST match AnalysisParams defaults in api/models/params.py
DEFAULT_PARAMS = {
    'moderate': {
        'expected_drift_pct': 0.3,  # expected_drift_pct_vt1
        'max_drift_pct': 1.0,       # max_drift_pct_vt1
        'sigma_pct': 7.0            # sigma_pct_vt1
    },
    'heavy': {
        'expected_drift_pct': 1.0,  # expected_drift_pct_vt2
        'max_drift_pct': 3.0,       # max_drift_pct_vt2
        'sigma_pct': 4.0            # sigma_pct_vt2
    },
    'severe': {
        'expected_drift_pct': 1.0,  # uses vt2 params
        'max_drift_pct': 3.0,       # uses vt2 params
        'sigma_pct': 4.0            # uses vt2 params
    }
}

DEFAULT_VT1_VE = 60.0
DEFAULT_VT2_VE = 80.0


# ============================================================================
# NIG Bayesian Update Functions
# ============================================================================

def apply_forgetting_factor(posterior: NIGPosterior, lambda_: float = 0.95) -> NIGPosterior:
    """
    Apply forgetting factor to posterior sufficient statistics.

    This decays older observations, giving more weight to recent data.
    Should be called BEFORE each update.

    Args:
        posterior: Current NIG posterior
        lambda_: Forgetting factor (0.95 = 5% decay per update, preserves baseline)

    Returns:
        New posterior with decayed statistics
    """
    return NIGPosterior(
        mu=posterior.mu,  # Mean preserved
        kappa=lambda_ * posterior.kappa,
        alpha=lambda_ * posterior.alpha,
        beta=lambda_ * posterior.beta,
        n_obs=posterior.n_obs  # Keep original count for blending logic
    )


def update_nig_posterior(
    posterior: NIGPosterior,
    observation: float,
    lambda_: float = 0.95
) -> NIGPosterior:
    """
    Perform Bayesian update of NIG posterior with new observation.

    Steps:
    1. Apply forgetting factor to decay old data
    2. Perform standard NIG conjugate update

    Args:
        posterior: Current NIG posterior
        observation: New observed value
        lambda_: Forgetting factor

    Returns:
        Updated NIG posterior
    """
    # Step 1: Apply forgetting factor
    decayed = apply_forgetting_factor(posterior, lambda_)

    # Step 2: Standard NIG update equations
    kappa_n = decayed.kappa + 1
    mu_n = (decayed.kappa * decayed.mu + observation) / kappa_n
    alpha_n = decayed.alpha + 0.5
    beta_n = decayed.beta + 0.5 * decayed.kappa * (observation - decayed.mu)**2 / kappa_n

    return NIGPosterior(
        mu=mu_n,
        kappa=kappa_n,
        alpha=alpha_n,
        beta=beta_n,
        n_obs=posterior.n_obs + 1
    )


def update_anchored_posterior(
    anchored: AnchoredNIGPosterior,
    observation: float,
    lambda_: float = 0.95
) -> AnchoredNIGPosterior:
    """
    Update AnchoredNIGPosterior with new observation using decaying anchor.

    Both the anchor_kappa and observation posterior decay, then a new
    observation is incorporated. Over time, observations dominate and
    the anchor fades toward zero influence.

    Args:
        anchored: Current anchored posterior state
        observation: New observed value
        lambda_: Forgetting factor (applied to both anchor and observations)

    Returns:
        Updated AnchoredNIGPosterior with decayed anchor and new observation
    """
    # Step 1: Decay anchor_kappa
    decayed_anchor_kappa = lambda_ * anchored.anchor_kappa

    # Step 2: Apply forgetting factor to observation posterior
    decayed_posterior = apply_forgetting_factor(anchored.posterior, lambda_)

    # Step 3: Standard NIG update with the new observation
    kappa_n = decayed_posterior.kappa + 1
    mu_n = (decayed_posterior.kappa * decayed_posterior.mu + observation) / kappa_n if kappa_n > 0 else observation
    alpha_n = decayed_posterior.alpha + 0.5
    beta_n = decayed_posterior.beta + 0.5 * decayed_posterior.kappa * (observation - decayed_posterior.mu)**2 / kappa_n if kappa_n > 0 else decayed_posterior.beta

    new_posterior = NIGPosterior(
        mu=mu_n,
        kappa=kappa_n,
        alpha=alpha_n,
        beta=beta_n,
        n_obs=anchored.posterior.n_obs + 1
    )

    return AnchoredNIGPosterior(
        anchor_value=anchored.anchor_value,
        anchor_kappa=decayed_anchor_kappa,
        posterior=new_posterior
    )


def update_ve_threshold_state(
    ve_state: VEThresholdState,
    observation: float,
    lambda_: float = 0.95
) -> VEThresholdState:
    """
    Update VE threshold state with new observation using decaying anchor.

    The anchor_kappa decays over time, eventually letting observations
    fully determine the threshold. current_value auto-updates to the
    anchored posterior mean.

    Args:
        ve_state: Current VE threshold state
        observation: New observed avg_ve value
        lambda_: Forgetting factor (applied to both anchor and observations)

    Returns:
        Updated VEThresholdState with new current_value
    """
    # Step 1: Decay anchor_kappa
    decayed_anchor_kappa = lambda_ * ve_state.anchor_kappa

    # Step 2: Apply forgetting factor to observation posterior
    decayed_posterior = apply_forgetting_factor(ve_state.posterior, lambda_)

    # Step 3: Standard NIG update with the new observation
    kappa_n = decayed_posterior.kappa + 1
    mu_n = (decayed_posterior.kappa * decayed_posterior.mu + observation) / kappa_n if kappa_n > 0 else observation
    alpha_n = decayed_posterior.alpha + 0.5
    beta_n = decayed_posterior.beta + 0.5 * decayed_posterior.kappa * (observation - decayed_posterior.mu)**2 / kappa_n if kappa_n > 0 else decayed_posterior.beta

    new_posterior = NIGPosterior(
        mu=mu_n,
        kappa=kappa_n,
        alpha=alpha_n,
        beta=beta_n,
        n_obs=ve_state.posterior.n_obs + 1
    )

    # Calculate new anchored mean
    total_kappa = decayed_anchor_kappa + new_posterior.kappa
    if total_kappa > 0:
        new_current_value = (decayed_anchor_kappa * ve_state.anchor_value + new_posterior.kappa * new_posterior.mu) / total_kappa
    else:
        new_current_value = ve_state.anchor_value

    return VEThresholdState(
        current_value=round(new_current_value, 1),  # Round to 0.1 L/min
        anchor_value=ve_state.anchor_value,
        anchor_kappa=decayed_anchor_kappa,
        posterior=new_posterior
    )


def blend_with_default(
    calibrated_value: float,
    default_value: float,
    run_count: int,
    min_runs: int = 3,
    full_runs: int = 5
) -> float:
    """
    Blend calibrated value with default based on run count.

    - < min_runs: Use default entirely
    - min_runs to full_runs: Linear blend
    - >= full_runs: Use calibrated entirely

    Args:
        calibrated_value: Value from NIG posterior
        default_value: Default parameter value
        run_count: Number of qualifying runs
        min_runs: Minimum runs before calibration applies
        full_runs: Runs needed for full calibration

    Returns:
        Blended parameter value
    """
    if run_count < min_runs:
        return default_value
    elif run_count >= full_runs:
        return calibrated_value
    else:
        # Linear blend: weight goes from 0 to 1 as run_count goes from min_runs to full_runs
        weight = (run_count - min_runs) / (full_runs - min_runs)
        return default_value * (1 - weight) + calibrated_value * weight


# ============================================================================
# Calibration Logic
# ============================================================================

def check_drift_calibration_eligible(
    run_type: RunType,
    interval_status: IntervalStatus,
    interval_duration_min: float
) -> bool:
    """
    Check if an interval is eligible for drift/sigma/split calibration.

    Eligibility requires:
    1. Interval duration >= 6 minutes
    2. Outcome matches domain expectation (not borderline)

    Args:
        run_type: The intensity domain
        interval_status: Classification result
        interval_duration_min: Interval duration in minutes

    Returns:
        True if eligible for calibration
    """
    # Minimum duration check
    if interval_duration_min < 6.0:
        return False

    # Borderline never contributes
    if interval_status == IntervalStatus.BORDERLINE:
        return False

    # Check domain-specific expectations
    if run_type == RunType.MODERATE:
        return interval_status == IntervalStatus.BELOW_THRESHOLD
    elif run_type == RunType.HEAVY:
        return interval_status == IntervalStatus.BELOW_THRESHOLD
    else:  # SEVERE
        return interval_status == IntervalStatus.ABOVE_THRESHOLD


def check_ve_threshold_update_needed(
    run_type: RunType,
    interval_status: IntervalStatus,
    avg_ve: float,
    vt1_ve: float,
    vt2_ve: float
) -> Tuple[Optional[str], bool]:
    """
    Check if VE threshold update is needed based on unexpected outcome.

    Implements the unexpected outcome matrix from the spec.

    Args:
        run_type: The intensity domain
        interval_status: Classification result
        avg_ve: Average VE during post-calibration period
        vt1_ve: Current VT1 threshold
        vt2_ve: Current VT2 threshold

    Returns:
        Tuple of (threshold_to_update: 'vt1'|'vt2'|None, increase: bool)
    """
    if interval_status == IntervalStatus.BORDERLINE:
        return None, False

    if run_type == RunType.MODERATE:
        if interval_status == IntervalStatus.ABOVE_THRESHOLD and avg_ve <= vt1_ve:
            # Unexpected: above threshold but VE is low → decrease VT1
            return 'vt1', False
        elif interval_status == IntervalStatus.BELOW_THRESHOLD and avg_ve > vt1_ve:
            # Unexpected: below threshold but VE is high → increase VT1
            return 'vt1', True

    elif run_type == RunType.HEAVY:
        if interval_status == IntervalStatus.ABOVE_THRESHOLD and avg_ve <= vt2_ve:
            # Unexpected: above threshold but VE is low → decrease VT2
            return 'vt2', False
        elif interval_status == IntervalStatus.BELOW_THRESHOLD and avg_ve > vt2_ve:
            # Unexpected: below threshold but VE is high → increase VT2
            return 'vt2', True

    elif run_type == RunType.SEVERE:
        if interval_status == IntervalStatus.ABOVE_THRESHOLD and avg_ve <= vt2_ve:
            # Unexpected: above threshold but VE <= VT2 → decrease VT2
            return 'vt2', False
        elif interval_status == IntervalStatus.BELOW_THRESHOLD and avg_ve > vt2_ve:
            # Unexpected: below threshold but VE > VT2 → increase VT2
            return 'vt2', True

    return None, False


def update_calibration_from_run(
    state: CalibrationState,
    run_type: RunType,
    interval_results: List[dict],
    lambda_: float = 0.95
) -> Tuple[CalibrationState, Optional[dict]]:
    """
    Update calibration state from a complete run using majority-based logic.

    A run only counts toward calibration if:
    1. More than 50% of qualifying intervals (≥6 min) share the same classification
       (either ABOVE_THRESHOLD or BELOW_THRESHOLD)
    2. The majority classification matches domain expectations

    When eligible, averaged values from majority intervals are used for calibration.
    Each run counts as ONE calibration sample regardless of number of intervals.

    All parameters use decaying anchor approach:
    - Early runs: anchor provides stability
    - Later runs: observations dominate as anchor fades
    - VE thresholds auto-update (no user prompt)

    Args:
        state: Current calibration state
        run_type: Intensity domain (MODERATE, HEAVY, or SEVERE)
        interval_results: List of interval result dicts, each containing:
            - status: IntervalStatus or string
            - start_time, end_time: timestamps in seconds
            - ve_drift_pct: observed drift rate
            - sigma_pct: observed sigma
            - avg_ve: average VE during post-calibration period
        lambda_: Forgetting factor for Bayesian updates

    Returns:
        Tuple of (updated_state, None) - ve_prompt is always None (auto-update)
    """
    # Skip calibration updates if calibration is disabled
    if not state.enabled:
        return state, None

    # Step 1: Filter to intervals ≥ 6 minutes
    qualifying_intervals = []
    for interval in interval_results:
        duration_min = (interval.get('end_time', 0) - interval.get('start_time', 0)) / 60.0
        if duration_min >= 6.0:
            # Parse status
            status_str = interval.get('status', 'BORDERLINE')
            if isinstance(status_str, str):
                status = IntervalStatus(status_str)
            else:
                status = status_str
            qualifying_intervals.append({
                'status': status,
                'drift_pct': interval.get('ve_drift_pct', 0.0),
                'sigma_pct': interval.get('sigma_pct', 5.0),
                'avg_ve': interval.get('avg_ve', 60.0)
            })

    # If no qualifying intervals, run doesn't count
    if not qualifying_intervals:
        return state, None

    # Step 2: Count classifications
    above_count = sum(1 for i in qualifying_intervals if i['status'] == IntervalStatus.ABOVE_THRESHOLD)
    below_count = sum(1 for i in qualifying_intervals if i['status'] == IntervalStatus.BELOW_THRESHOLD)
    total_count = len(qualifying_intervals)

    # Step 3: Check for majority (> 50% of total)
    majority_status = None
    majority_intervals = []

    if above_count > total_count / 2:
        majority_status = IntervalStatus.ABOVE_THRESHOLD
        majority_intervals = [i for i in qualifying_intervals if i['status'] == IntervalStatus.ABOVE_THRESHOLD]
    elif below_count > total_count / 2:
        majority_status = IntervalStatus.BELOW_THRESHOLD
        majority_intervals = [i for i in qualifying_intervals if i['status'] == IntervalStatus.BELOW_THRESHOLD]

    # No majority means run doesn't count for calibration
    if majority_status is None:
        return state, None

    # Step 4: Calculate averages from majority intervals
    avg_drift_pct = sum(i['drift_pct'] for i in majority_intervals) / len(majority_intervals)
    avg_sigma_pct = sum(i['sigma_pct'] for i in majority_intervals) / len(majority_intervals)
    avg_avg_ve = sum(i['avg_ve'] for i in majority_intervals) / len(majority_intervals)

    # Step 5: Check domain eligibility and update drift/sigma calibration
    # Moderate and Heavy expect BELOW_THRESHOLD, Severe expects ABOVE_THRESHOLD
    domain_expects_below = run_type in (RunType.MODERATE, RunType.HEAVY)
    domain_expects_above = run_type == RunType.SEVERE

    if (domain_expects_below and majority_status == IntervalStatus.BELOW_THRESHOLD) or \
       (domain_expects_above and majority_status == IntervalStatus.ABOVE_THRESHOLD):
        # Eligible for drift/sigma calibration
        domain = state.get_domain_posterior(run_type)

        # Update expected_drift and sigma using decaying anchor approach
        domain.expected_drift = update_anchored_posterior(domain.expected_drift, avg_drift_pct, lambda_)
        domain.sigma = update_anchored_posterior(domain.sigma, avg_sigma_pct, lambda_)

        # Note: max_drift values are derived from expected_drift in enforce_ordinal_constraints
        # - moderate.max_drift = heavy.expected_drift
        # - heavy.max_drift = severe.expected_drift

        # Increment run count by 1 (not per interval)
        state.increment_run_count(run_type)

    # Step 6: Check VE threshold update using averaged avg_ve from majority intervals
    threshold_key, increase = check_ve_threshold_update_needed(
        run_type, majority_status, avg_avg_ve,
        state.vt1_ve.current_value, state.vt2_ve.current_value
    )

    if threshold_key:
        # Update VE threshold using decaying anchor (auto-updates current_value)
        if threshold_key == 'vt1':
            state.vt1_ve = update_ve_threshold_state(state.vt1_ve, avg_avg_ve, lambda_)
        else:
            state.vt2_ve = update_ve_threshold_state(state.vt2_ve, avg_avg_ve, lambda_)

    state.last_updated = datetime.utcnow()
    return state, None  # No prompt - VE thresholds auto-update


def apply_manual_threshold_override(
    state: CalibrationState,
    threshold_key: str,
    new_value: float
) -> CalibrationState:
    """
    Apply user's manual override of a VE threshold.

    When user manually sets a threshold value, it becomes a new anchor.
    This resets the observation posterior and starts fresh calibration
    from the user-specified value.

    Args:
        state: Current calibration state
        threshold_key: 'vt1' or 'vt2'
        new_value: The new threshold value set by user

    Returns:
        Updated calibration state
    """
    if threshold_key == 'vt1':
        state.vt1_ve.reset_anchor(new_value)
    else:
        state.vt2_ve.reset_anchor(new_value)

    return state


def get_blended_params(
    state: CalibrationState,
    run_type: RunType
) -> dict:
    """
    Get blended parameters for a run type, mixing calibrated with defaults.

    If calibration is disabled, returns system defaults.

    Args:
        state: Current calibration state
        run_type: Intensity domain

    Returns:
        Dict with blended parameter values
    """
    defaults = DEFAULT_PARAMS[run_type.value.lower()]

    # If calibration is disabled, return system defaults
    if not state.enabled:
        return {
            'expected_drift_pct': defaults['expected_drift_pct'],
            'max_drift_pct': defaults['max_drift_pct'],
            'sigma_pct': defaults['sigma_pct'],
            'vt1_ve': DEFAULT_VT1_VE,
            'vt2_ve': DEFAULT_VT2_VE,
            'run_count': 0,
            'calibration_enabled': False
        }

    domain = state.get_domain_posterior(run_type)
    run_count = state.get_run_count(run_type)

    return {
        'expected_drift_pct': blend_with_default(
            domain.expected_drift.get_point_estimate(),
            defaults['expected_drift_pct'],
            run_count
        ),
        'max_drift_pct': blend_with_default(
            domain.max_drift.get_point_estimate(),
            defaults['max_drift_pct'],
            run_count
        ),
        'sigma_pct': blend_with_default(
            domain.sigma.get_point_estimate(),
            defaults['sigma_pct'],
            run_count
        ),
        'vt1_ve': state.vt1_ve.current_value,
        'vt2_ve': state.vt2_ve.current_value,
        'run_count': run_count,
        'calibration_enabled': True
    }


def enforce_ordinal_constraints(state: CalibrationState) -> CalibrationState:
    """
    Enforce ordinal constraints: Moderate drift < Heavy drift < Severe drift.

    Only applies to drift parameters (expected and max).
    Sigma has NO ordinal constraint.

    Args:
        state: Current calibration state

    Returns:
        State with constraints enforced
    """
    # Get point estimates (uses anchored mean from AnchoredNIGPosterior)
    mod_expected = state.moderate.expected_drift.get_point_estimate()
    heavy_expected = state.heavy.expected_drift.get_point_estimate()
    severe_expected = state.severe.expected_drift.get_point_estimate()

    # Enforce: Moderate < Heavy
    if mod_expected >= heavy_expected:
        avg = (mod_expected + heavy_expected) / 2
        state.moderate.expected_drift.posterior.mu = avg - 0.1
        state.heavy.expected_drift.posterior.mu = avg + 0.1

    # Enforce: Heavy < Severe
    if heavy_expected >= severe_expected:
        avg = (heavy_expected + severe_expected) / 2
        state.heavy.expected_drift.posterior.mu = avg - 0.1
        state.severe.expected_drift.posterior.mu = avg + 0.1

    # Derive max_drift values from the next domain's expected_drift
    # Moderate's ceiling = Heavy's expected (where heavy domain starts)
    # Heavy's ceiling = Severe's expected (where severe domain starts)
    state.moderate.max_drift.posterior.mu = state.heavy.expected_drift.get_point_estimate()
    state.heavy.max_drift.posterior.mu = state.severe.expected_drift.get_point_estimate()

    # Enforce VT1 < VT2
    if state.vt1_ve.current_value >= state.vt2_ve.current_value:
        gap = 5.0  # Minimum 5 L/min gap
        mid = (state.vt1_ve.current_value + state.vt2_ve.current_value) / 2
        state.vt1_ve.current_value = mid - gap / 2
        state.vt2_ve.current_value = mid + gap / 2

    return state
