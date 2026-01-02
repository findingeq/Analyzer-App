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
class DomainPosterior:
    """
    All NIG posteriors for a single intensity domain.

    Each domain (Moderate/Heavy/Severe) has its own set of parameters.
    """
    expected_drift: NIGPosterior = field(default_factory=NIGPosterior)
    max_drift: NIGPosterior = field(default_factory=NIGPosterior)
    sigma: NIGPosterior = field(default_factory=NIGPosterior)
    split_ratio: NIGPosterior = field(default_factory=NIGPosterior)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'expected_drift': self.expected_drift.to_dict(),
            'max_drift': self.max_drift.to_dict(),
            'sigma': self.sigma.to_dict(),
            'split_ratio': self.split_ratio.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DomainPosterior':
        """Deserialize from dictionary."""
        return cls(
            expected_drift=NIGPosterior.from_dict(d.get('expected_drift', {})),
            max_drift=NIGPosterior.from_dict(d.get('max_drift', {})),
            sigma=NIGPosterior.from_dict(d.get('sigma', {})),
            split_ratio=NIGPosterior.from_dict(d.get('split_ratio', {}))
        )


@dataclass
class VEThresholdState:
    """
    State for a VE threshold (VT1 or VT2).

    Uses "Anchor & Pull" Bayesian approach:
    - current_value is the user-approved anchor
    - posterior tracks observations, but anchored to current_value
    - When posterior mean diverges enough from anchor, user is prompted
    """
    current_value: float = 60.0  # Current user-approved threshold (anchor)
    posterior: NIGPosterior = field(default_factory=NIGPosterior)
    anchor_kappa: float = 4.0    # Virtual sample size for anchoring (stability)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'current_value': self.current_value,
            'posterior': self.posterior.to_dict(),
            'anchor_kappa': self.anchor_kappa
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'VEThresholdState':
        """Deserialize from dictionary."""
        return cls(
            current_value=d.get('current_value', 60.0),
            posterior=NIGPosterior.from_dict(d.get('posterior', {})),
            anchor_kappa=d.get('anchor_kappa', 4.0)
        )

    def reset_to_anchor(self) -> None:
        """
        Reset posterior to be anchored at current_value.
        Called after user approves a change or manually sets threshold.
        """
        self.posterior = NIGPosterior(
            mu=self.current_value,
            kappa=self.anchor_kappa,
            alpha=2.0,
            beta=1.0,
            n_obs=0
        )


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

DEFAULT_PARAMS = {
    'moderate': {
        'expected_drift_pct': 0.5,
        'max_drift_pct': 2.0,
        'sigma_pct': 10.0,
        'split_ratio': 1.0
    },
    'heavy': {
        'expected_drift_pct': 1.0,
        'max_drift_pct': 3.0,
        'sigma_pct': 5.0,
        'split_ratio': 1.0
    },
    'severe': {
        'expected_drift_pct': 2.0,
        'max_drift_pct': 5.0,
        'sigma_pct': 5.0,
        'split_ratio': 1.2
    }
}

DEFAULT_VT1_VE = 60.0
DEFAULT_VT2_VE = 80.0


# ============================================================================
# NIG Bayesian Update Functions
# ============================================================================

def apply_forgetting_factor(posterior: NIGPosterior, lambda_: float = 0.9) -> NIGPosterior:
    """
    Apply forgetting factor to posterior sufficient statistics.

    This decays older observations, giving more weight to recent data.
    Should be called BEFORE each update.

    Args:
        posterior: Current NIG posterior
        lambda_: Forgetting factor (0.9 = 10% decay per update)

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
    lambda_: float = 0.9
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


def update_anchored_ve_posterior(
    ve_state: VEThresholdState,
    observation: float,
    lambda_: float = 0.9
) -> NIGPosterior:
    """
    Update VE threshold posterior using Anchor & Pull method.

    The current_value acts as a strong prior (anchor) with weight anchor_kappa.
    This prevents single observations from causing large jumps while still
    allowing the posterior to converge toward true values over time.

    Args:
        ve_state: Current VE threshold state (contains anchor and posterior)
        observation: New observed avg_ve value
        lambda_: Forgetting factor for decaying old observations

    Returns:
        Updated NIG posterior
    """
    # Step 1: Apply forgetting factor to existing posterior
    decayed = apply_forgetting_factor(ve_state.posterior, lambda_)

    # Step 2: Blend decayed posterior with anchor
    # The anchor always contributes anchor_kappa worth of "virtual observations"
    # at the current_value (user-approved threshold)
    anchor_kappa = ve_state.anchor_kappa
    anchor_value = ve_state.current_value

    # Combined prior: anchor + decayed observations
    combined_kappa = anchor_kappa + decayed.kappa
    combined_mu = (anchor_kappa * anchor_value + decayed.kappa * decayed.mu) / combined_kappa

    # Step 3: Standard NIG update with the new observation
    kappa_n = combined_kappa + 1
    mu_n = (combined_kappa * combined_mu + observation) / kappa_n
    alpha_n = decayed.alpha + 0.5
    beta_n = decayed.beta + 0.5 * combined_kappa * (observation - combined_mu)**2 / kappa_n

    return NIGPosterior(
        mu=mu_n,
        kappa=kappa_n - anchor_kappa,  # Store only observation-derived kappa
        alpha=alpha_n,
        beta=beta_n,
        n_obs=ve_state.posterior.n_obs + 1
    )


def get_anchored_posterior_mean(ve_state: VEThresholdState) -> float:
    """
    Get the effective posterior mean considering the anchor.

    This combines the anchor (current_value with anchor_kappa weight)
    with the observation-derived posterior.

    Args:
        ve_state: Current VE threshold state

    Returns:
        Anchored posterior mean
    """
    anchor_kappa = ve_state.anchor_kappa
    anchor_value = ve_state.current_value
    obs_kappa = ve_state.posterior.kappa
    obs_mu = ve_state.posterior.mu

    if obs_kappa <= 0:
        return anchor_value

    combined_kappa = anchor_kappa + obs_kappa
    return (anchor_kappa * anchor_value + obs_kappa * obs_mu) / combined_kappa


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


def update_calibration_from_interval(
    state: CalibrationState,
    run_type: RunType,
    interval_status: IntervalStatus,
    interval_duration_min: float,
    drift_pct: float,
    sigma_pct: float,
    split_ratio: float,
    avg_ve: float,
    lambda_: float = 0.9
) -> Tuple[CalibrationState, Optional[dict]]:
    """
    Update calibration state from a single interval result.

    Args:
        state: Current calibration state
        run_type: Intensity domain
        interval_status: Classification result
        interval_duration_min: Interval duration in minutes
        drift_pct: Observed drift rate (%/min)
        sigma_pct: Observed sigma (% of baseline)
        split_ratio: Observed slope split ratio
        avg_ve: Average VE during post-calibration period
        lambda_: Forgetting factor

    Returns:
        Tuple of (updated_state, ve_prompt) where ve_prompt is dict if
        user approval needed for VE threshold change >= 1 L/min
    """
    # Skip calibration updates if calibration is disabled
    # Learned data is preserved but no new updates occur
    if not state.enabled:
        return state, None

    ve_prompt = None

    # Check if drift calibration is eligible
    if check_drift_calibration_eligible(run_type, interval_status, interval_duration_min):
        # Update domain-specific parameters (no cross-pollination)
        domain = state.get_domain_posterior(run_type)

        domain.expected_drift = update_nig_posterior(domain.expected_drift, drift_pct, lambda_)
        domain.sigma = update_nig_posterior(domain.sigma, sigma_pct, lambda_)
        if split_ratio is not None and split_ratio > 0:
            domain.split_ratio = update_nig_posterior(domain.split_ratio, split_ratio, lambda_)

        # Update max drift based on observed (use higher of expected + buffer)
        observed_max = drift_pct * 1.5  # Buffer above observed
        domain.max_drift = update_nig_posterior(domain.max_drift, observed_max, lambda_)

        # Increment qualifying run count
        state.increment_run_count(run_type)

    # Check if VE threshold update is needed (independent of drift eligibility)
    threshold_key, increase = check_ve_threshold_update_needed(
        run_type, interval_status, avg_ve,
        state.vt1_ve.current_value, state.vt2_ve.current_value
    )

    if threshold_key:
        ve_state = state.vt1_ve if threshold_key == 'vt1' else state.vt2_ve

        # Update posterior using Anchor & Pull method
        ve_state.posterior = update_anchored_ve_posterior(ve_state, avg_ve, lambda_)

        # Get anchored posterior mean (blends anchor with observations)
        posterior_mean = get_anchored_posterior_mean(ve_state)

        # Check if we need to prompt user (posterior diverged >= 1 L/min from anchor)
        divergence = posterior_mean - ve_state.current_value
        if abs(divergence) >= 1.0:
            ve_prompt = {
                'threshold': threshold_key,
                'current_value': ve_state.current_value,
                'proposed_value': round(posterior_mean, 1),
                'divergence': round(divergence, 1)
            }

    state.last_updated = datetime.utcnow()
    return state, ve_prompt


def apply_ve_threshold_approval(
    state: CalibrationState,
    threshold_key: str,
    approved: bool,
    proposed_value: float
) -> CalibrationState:
    """
    Apply user's approval/rejection of VE threshold change.

    After approval or rejection, the anchor is reset to the new current_value.
    This means future observations start fresh relative to the new baseline.

    Args:
        state: Current calibration state
        threshold_key: 'vt1' or 'vt2'
        approved: Whether user approved the change
        proposed_value: The value that was proposed to the user

    Returns:
        Updated calibration state
    """
    ve_state = state.vt1_ve if threshold_key == 'vt1' else state.vt2_ve

    if approved:
        # Apply the proposed change
        ve_state.current_value = proposed_value

    # Reset posterior to anchor at new/current value (either way)
    # This starts fresh observation tracking from the approved baseline
    ve_state.reset_to_anchor()

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
            'split_ratio': defaults['split_ratio'],
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
        'split_ratio': blend_with_default(
            domain.split_ratio.get_point_estimate(),
            defaults['split_ratio'],
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
    Sigma and split_ratio have NO ordinal constraints.

    Args:
        state: Current calibration state

    Returns:
        State with constraints enforced
    """
    # Get point estimates
    mod_expected = state.moderate.expected_drift.get_point_estimate()
    heavy_expected = state.heavy.expected_drift.get_point_estimate()
    severe_expected = state.severe.expected_drift.get_point_estimate()

    mod_max = state.moderate.max_drift.get_point_estimate()
    heavy_max = state.heavy.max_drift.get_point_estimate()
    severe_max = state.severe.max_drift.get_point_estimate()

    # Enforce: Moderate < Heavy
    if mod_expected >= heavy_expected:
        avg = (mod_expected + heavy_expected) / 2
        state.moderate.expected_drift.mu = avg - 0.1
        state.heavy.expected_drift.mu = avg + 0.1

    # Enforce: Heavy < Severe
    if heavy_expected >= severe_expected:
        avg = (heavy_expected + severe_expected) / 2
        state.heavy.expected_drift.mu = avg - 0.1
        state.severe.expected_drift.mu = avg + 0.1

    # Same for max drift
    if mod_max >= heavy_max:
        avg = (mod_max + heavy_max) / 2
        state.moderate.max_drift.mu = avg - 0.1
        state.heavy.max_drift.mu = avg + 0.1

    if heavy_max >= severe_max:
        avg = (heavy_max + severe_max) / 2
        state.heavy.max_drift.mu = avg - 0.1
        state.severe.max_drift.mu = avg + 0.1

    # Enforce VT1 < VT2
    if state.vt1_ve.current_value >= state.vt2_ve.current_value:
        gap = 5.0  # Minimum 5 L/min gap
        mid = (state.vt1_ve.current_value + state.vt2_ve.current_value) / 2
        state.vt1_ve.current_value = mid - gap / 2
        state.vt2_ve.current_value = mid + gap / 2

    return state
