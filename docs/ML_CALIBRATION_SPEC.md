# ML Calibration System - Implementation Specification

> **Status**: Design Complete - Pending Implementation
> **Last Updated**: 2026-01-02

---

## Overview

This document specifies the machine learning calibration system for automatically adjusting VT1/VT2 parameters based on accumulated user run data.

---

## Algorithm: NIG with Forgetting Factor

**Core approach**: Normal-Inverse-Gamma (NIG) with Forgetting Factor

Before each update, decay the sufficient statistics:
```
κ' = λ × κ
α' = λ × α
β' = λ × β
```

Then perform standard Bayesian update with new observation.

- **λ** = forgetting factor (≈ 0.9), weights recent runs more heavily
- Damping is achieved through the Bayesian framework itself
- Preserves valid confidence intervals
- No separate EMA formula needed (would cause "double damping")

---

## Minimum Duration Requirement

**Only calibrate for runs/intervals ≥ 6 minutes in length.**

Intervals < 6 min use ceiling-based analysis and do not contribute to calibration.

---

## Three Intensity Domains

| Enum Value | Domain | Description |
|------------|--------|-------------|
| `MODERATE` | Moderate | Below VT1 threshold |
| `HEAVY` | Heavy | Between VT1 and VT2 |
| `SEVERE` | Severe | Above VT2 threshold |

**Note**: All three domains can have intervals. Enum values to be renamed from `VT1_STEADY`/`VT2_INTERVAL`.

---

## Parameters to Calibrate

### Per-Domain Parameters:

| Parameter | Description | Ordinal Constraint |
|-----------|-------------|-------------------|
| `expected_drift_pct` | Expected drift rate (%/min) | Moderate < Heavy < Severe |
| `max_drift_pct` | Max acceptable drift (%/min) | Moderate < Heavy < Severe |
| `sigma_pct` | Noise/variability (% of baseline) | **NONE** |
| `split_ratio` | Slope acceleration ratio | **NONE** |

### Global VE Thresholds:

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `vt1_ve_ceiling` | VT1 VE threshold (L/min) | VT1 < VT2 |
| `vt2_ve_ceiling` | VT2 VE threshold (L/min) | VT1 < VT2 |

---

## Drift/Sigma/Split Calibration Logic

### For Interval Runs (≥ 6 min intervals only):
1. **Majority rule**: >50% of qualifying intervals must meet domain expectation
2. **Selective calculation**: Only intervals meeting expectation contribute to parameter updates
3. Borderline intervals excluded from calculations
4. Intervals < 6 min excluded entirely

### Domain-Specific Update Rules:

| Run Domain | Expected Outcome | Parameters Updated |
|------------|------------------|-------------------|
| **Moderate** | BELOW_THRESHOLD | Moderate: drift, sigma, split |
| **Heavy** | BELOW_THRESHOLD | Heavy: drift, sigma, split |
| **Severe** | ABOVE_THRESHOLD | Severe: drift, sigma, split |

### Domain Isolation (No Cross-Pollination):
- **Moderate runs** → only update Moderate parameters
- **Heavy runs** → only update Heavy parameters
- **Severe runs** → only update Severe parameters

Domain models are kept pure. If a Severe run unexpectedly shows low drift, this triggers VT2 threshold adjustment, not Heavy parameter updates.

**Rationale**: Heavy domain behavior is concave (drift stabilizes), while Severe is convex (drift accelerates). Cross-pollination would corrupt domain-specific drift shapes.

---

## VE Threshold Calibration Logic

### Unexpected Outcome Matrix:

| Domain | Classification | AVG VE of Post-Cal Period | Unexpected? | VE Update Action |
|--------|---------------|---------------------------|-------------|------------------|
| Moderate | ABOVE_THRESHOLD | ≤ VT1 VE | YES | Decrease VT1 ceiling |
| Moderate | ABOVE_THRESHOLD | > VT1 VE | NO | No update |
| Moderate | BORDERLINE | — | NO | No update |
| Moderate | BELOW_THRESHOLD | ≤ VT1 VE | NO | No update |
| Moderate | BELOW_THRESHOLD | > VT1 VE | YES | Increase VT1 ceiling |
| Heavy | ABOVE_THRESHOLD | ≤ VT2 VE | YES | Decrease VT2 ceiling |
| Heavy | BORDERLINE | — | NO | No update |
| Heavy | BELOW_THRESHOLD | ≤ VT2 VE | NO | No update |
| Heavy | BELOW_THRESHOLD | > VT2 VE | YES | Increase VT2 ceiling |
| Severe | ABOVE_THRESHOLD | > VT2 VE | NO | No update (expected) |
| Severe | ABOVE_THRESHOLD | ≤ VT2 VE | YES | Decrease VT2 ceiling |
| Severe | BORDERLINE | — | NO | No update |
| Severe | BELOW_THRESHOLD | > VT2 VE | YES | Increase VT2 ceiling |
| Severe | BELOW_THRESHOLD | ≤ VT2 VE | NO | No update |

### Update Magnitude Scaling:
- Greater deviation from threshold → greater incremental change
- Formula: `update_magnitude ∝ |avg_ve - threshold|`

### Internal vs User-Facing Updates:
- **Internal**: Updates occur for any fractional change (accumulated in background)
- **User prompt**: Only shown when cumulative change reaches ≥ ±1 L/min

### User Rejection Handling:
1. If user rejects proposed update → calibration continues accumulating in background
2. Background tracks cumulative delta from last user prompt
3. When cumulative delta reaches another ≥ ±1 L/min → prompt user again
4. This prevents nagging while still tracking the physiological trend

---

## Minimum Data Requirements

- **Minimum 3 runs** per domain before calibration applies
- **Soft blending** for runs 3-5 (gradual transition from defaults to calibrated)
- After 5+ runs: fully calibrated values used

---

## User Controls

1. **Calibration exclusion checkbox** - per run in cloud run table
   - Allows user to exclude specific runs from calibration
   - Useful for unusual/outlier sessions

2. **VE threshold approval popup** - when cumulative change ≥ ±1 L/min
   - Shows proposed new threshold
   - User can accept or reject
   - Rejection doesn't stop background tracking

---

## Storage Schema (per user)

```python
class CalibrationState:
    # Per-domain NIG posteriors (all domains have full parameter sets)
    moderate: DomainPosterior
    heavy: DomainPosterior
    severe: DomainPosterior

    # Global VE thresholds
    vt1_ve: VEThresholdState
    vt2_ve: VEThresholdState

    # Metadata
    last_updated: datetime
    run_counts: Dict[str, int]  # qualifying runs per domain

class DomainPosterior:
    expected_drift: NIGPosterior
    max_drift: NIGPosterior
    sigma: NIGPosterior
    split_ratio: NIGPosterior

class VEThresholdState:
    current_value: float
    posterior: NIGPosterior
    pending_delta: float  # Accumulated change since last user prompt
    last_prompted_value: float

class NIGPosterior:
    mu: float      # mean
    kappa: float   # precision of mean
    alpha: float   # shape
    beta: float    # scale
    n_obs: int     # observation count
```

---

## API Endpoints (proposed)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calibration/state` | GET | Get user's calibration state |
| `/api/calibration/update` | POST | Update calibration from run result |
| `/api/calibration/reset` | POST | Reset to defaults |
| `/api/calibration/exclude-run` | POST | Toggle run exclusion |
| `/api/calibration/approve-ve` | POST | User approves/rejects VE threshold change |

---

## Integration with Analysis

When running analysis:
1. Load user's calibration state
2. Check if run/intervals meet 6 min minimum
3. Blend calibrated values with defaults based on run count
4. Use blended parameters for CUSUM analysis
5. After classification:
   - Check if outcome triggers drift/sigma/split calibration (domain-specific only)
   - Check if outcome triggers VE threshold calibration
   - Apply forgetting factor to posteriors, then update
   - Check if VE threshold change ≥ ±1 L/min for user prompt

---

## Mobile App Sync

**iPhone repo**: https://github.com/findingeq/vt-threshold-app

To investigate:
- Existing sync infrastructure between web and mobile
- API endpoints for pushing updated thresholds
- Storage mechanism on mobile side

---

## Code Changes Required

1. **Rename enums**: `VT1_STEADY` → `MODERATE`, `VT2_INTERVAL` → `HEAVY`
2. **Allow intervals for all domains** (already partially done)
3. **Backend calibration service** (new) - NIG math with forgetting factor
4. **Storage/database schema** (new)
5. **API endpoints** (new)
6. **Frontend UI**: exclusion checkboxes, approval popup (new)
7. **Mobile sync** (TBD based on repo investigation)

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-02 | Initial specification |
