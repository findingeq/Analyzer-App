# Classification Specification

> **Status**: Implemented
> **Last Updated**: 2026-01-02

---

## Overview

This document specifies the classification system used to determine whether an interval is BELOW_THRESHOLD, BORDERLINE, or ABOVE_THRESHOLD based on VE drift analysis.

---

## Analysis Modes

Two analysis modes are available, selected based on interval duration:

| Interval Duration | Analysis Mode | Description |
|-------------------|---------------|-------------|
| ≥ 6 minutes | **Segmented** | Uses calibration period + drift detection |
| < 6 minutes | **Ceiling-based** | Uses absolute VE threshold |

**Override**: If `use_thresholds_for_all = true`, ceiling-based analysis is used regardless of duration.

---

## Segmented Analysis (≥ 6 min intervals)

### Phase Structure

```
┌──────────────┬─────────────┬───────────────────────────────┐
│   Phase I    │  Phase II   │          Phase III            │
│   (Warm-up)  │   (Ramp)    │     (Steady State/Drift)      │
├──────────────┴─────────────┼───────────────────────────────┤
│      Blanking Period       │      Analysis Period          │
│                            │                               │
│   ← Phase III Onset →      │ ← Calibration → ← CUSUM →    │
└────────────────────────────┴───────────────────────────────┘
```

### Phase III Onset Detection

All domains use a piecewise linear "hinge" model with Huber loss:

```
VE(t) = β₀ + β₁·t + β₂·max(0, t - τ)
```

Where:
- `τ` = Phase III onset breakpoint (the hinge point)
- `β₁` = Slope of Phase II (initial ramp)
- `β₁ + β₂` = Slope of Phase III (drift)

**Detection constraints** (same for all domains):

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ Min | 90 sec | Minimum allowed Phase III onset |
| τ Max | 210 sec (3:30) | Maximum allowed Phase III onset |
| τ Default | 150 sec (2.5 min) | Fallback if detection fails |

**Detection failure conditions**:
- τ converges to constraint boundary (within 1 second)
- Optimization fails to converge
- Insufficient data in constrained window

### Calibration Window

**Purpose**: Establish baseline VE for drift calculation

**Same for all domains**:
- **Start**: Phase III onset
- **Duration**: 60 seconds

**Outputs**:
- `cal_ve_mean`: Mean VE during calibration (baseline)
- `cal_midpoint_min`: Time midpoint of calibration window (minutes)

---

## CUSUM Calculation

### Segmented CUSUM (Drift-based)

**Expected VE line** (drift model):

```
expected_drift = (expected_drift_pct / 100) × cal_ve_mean
α = cal_ve_mean - expected_drift × cal_midpoint_min
VE_expected(t) = α + expected_drift × t
```

**CUSUM parameters**:

```
σ_ref = (sigma_pct / 100) × cal_ve_mean
k = 0.5 × σ_ref                    (slack parameter)
h = h_multiplier × σ_ref           (threshold)
```

**CUSUM accumulation** (starts after calibration window ends):

```
S₀ = 0
Sᵢ = max(0, Sᵢ₋₁ + (VE_observed[i] - VE_expected[i]) - k)
```

**Alarm condition**: `Sᵢ > h`

**Recovery condition**: Alarm triggered AND `S_final ≤ h/2`

### Ceiling-based CUSUM (Absolute threshold)

**For short intervals (< 6 min)**:

```
σ_ref = (sigma_pct / 100) × ceiling_ve
k = 0.5 × σ_ref
h = h_multiplier × σ_ref
VE_expected = ceiling_ve (constant)
```

**CUSUM accumulation** (starts after 20-second warmup):

```
S₀ = 0
Sᵢ = max(0, Sᵢ₋₁ + (VE_observed[i] - ceiling_ve) - k)
```

### Domain-Specific Parameters

| Parameter | Moderate (VT1) | Heavy/Severe (VT2) |
|-----------|----------------|-------------------|
| `h_multiplier` | 5.0 | 5.0 |
| `sigma_pct` | 7.0% | 4.0% |
| `expected_drift_pct` | 0.3%/min | 1.0%/min |
| `max_drift_pct` | N/A | 3.0%/min |
| `split_ratio_threshold` | N/A | 1.2 |

**Note**: Moderate domain uses only `expected_drift_pct` for classification. The `max_drift_pct` and `split_ratio_threshold` are not used because:
1. True moderate exercise (below VT1) has virtually no drift
2. If drift exceeds expected (0.3%/min), runner is likely in heavy domain
3. In heavy domain, the VO2 slow component plateaus after 6-10 minutes, so split slope ratio adds no diagnostic value

---

## Slope Calculation

### Single Slope (Overall Drift)

Fitted using **Huber regression** (robust to outliers) on post-calibration data.

**Model**:
```
VE(t) = intercept + slope × t
```

**Huber loss function**:
```
L(r) = {
  0.5 × r²           if |r| ≤ δ
  δ × (|r| - 0.5δ)   if |r| > δ
}
```

Where `δ = 5.0` (Huber delta parameter)

**Output**:
- `slope`: VE change per minute (L/min per minute)
- `slope_pct`: `(slope / cal_ve_mean) × 100` (% per minute)

**Used by**: All domains (Moderate, Heavy, Severe)

### Split Slope (Two-Segment) — Heavy/Severe Only

Detects slope change within Phase III using a **second hinge model**.

**Not used for Moderate domain** because:
- Moderate exercise has minimal drift (steady state)
- Heavy domain VO2 slow component plateaus after 6-10 minutes
- Split slope ratio on small numbers is noisy and unreliable

**Constraint window**:
- Start: Phase III onset + 120 seconds
- End: Interval end - 120 seconds
- Fallback: Midpoint between Phase III onset and interval end

**Model**:
```
VE(t) = β₀ + β₁·t + β₂·max(0, t - τ₂)
```

Where:
- `τ₂` = Second hinge point (slope change time)
- `β₁` = Slope of segment 1 (early Phase III)
- `β₁ + β₂` = Slope of segment 2 (late Phase III)

**Outputs**:
- `slope1_pct`: First segment slope as % of baseline per minute
- `slope2_pct`: Second segment slope as % of baseline per minute
- `split_slope_ratio`: `slope2_pct / slope1_pct`

**Split ratio bounds**:
- If |slope1_pct| < 0.1%: Use minimum denominator of 0.1%
- Maximum ratio capped at 5.0

---

## Classification Tree

### Moderate Domain (VT1) Classification

**Inputs**:
- `cusum_alarm`: Did CUSUM exceed threshold h?
- `cusum_recovered`: Did CUSUM return below h/2?
- `overall_slope_pct`: Total drift rate (%/min)

**Threshold**:
- `drift_threshold` = expected_drift_pct (0.3%/min default)

**Decision Tree**:

```
IF (cusum_alarm = FALSE) OR (cusum_recovered = TRUE):
│
├── IF overall_slope_pct < drift_threshold:
│   └── BELOW_THRESHOLD
│
└── ELSE:
    └── BORDERLINE

ELSE (cusum_alarm = TRUE AND cusum_recovered = FALSE):
│
├── IF overall_slope_pct < drift_threshold:
│   └── BORDERLINE
│
└── ELSE:
    └── ABOVE_THRESHOLD
```

**Rationale**: Moderate domain classification is simplified because:
- True moderate exercise has virtually no VO2 drift
- Any drift exceeding 0.3%/min suggests runner is not in moderate domain
- No need for max_drift or split_ratio checks

---

### Heavy/Severe Domain (VT2) Classification

**Inputs**:
- `cusum_alarm`: Did CUSUM exceed threshold h?
- `cusum_recovered`: Did CUSUM return below h/2?
- `overall_slope_pct`: Total drift rate (%/min)
- `split_slope_ratio`: Ratio of segment 2 to segment 1 slopes

**Thresholds**:
- `low_threshold` = expected_drift_pct (1.0%/min default)
- `high_threshold` = max_drift_pct (3.0%/min default)
- `split_ratio_threshold` = 1.2

**Decision Tree**:

```
IF (cusum_alarm = FALSE) OR (cusum_recovered = TRUE):
│
├── IF overall_slope_pct < low_threshold:
│   └── BELOW_THRESHOLD
│
├── ELSE IF overall_slope_pct < high_threshold:
│   ├── IF split_slope_ratio < split_ratio_threshold:
│   │   └── BELOW_THRESHOLD
│   └── ELSE:
│       └── BORDERLINE
│
└── ELSE (slope ≥ high_threshold):
    └── BORDERLINE

ELSE (cusum_alarm = TRUE AND cusum_recovered = FALSE):
│
├── IF overall_slope_pct < low_threshold:
│   └── BORDERLINE
│
├── ELSE IF overall_slope_pct < high_threshold:
│   ├── IF split_slope_ratio < split_ratio_threshold:
│   │   └── BORDERLINE
│   └── ELSE:
│       └── ABOVE_THRESHOLD
│
└── ELSE (slope ≥ high_threshold):
    └── ABOVE_THRESHOLD
```

### Ceiling-based Analysis Classification

**Simpler logic** (no slope analysis):

```
IF (cusum_alarm = FALSE) OR (cusum_recovered = TRUE):
    └── BELOW_THRESHOLD

ELSE (cusum_alarm = TRUE AND cusum_recovered = FALSE):
    └── ABOVE_THRESHOLD
```

**Note**: Ceiling-based analysis does not produce BORDERLINE status.

---

## Classification Rationale

### Why Three-Way Classification?

| Status | Meaning | Calibration Impact |
|--------|---------|-------------------|
| BELOW_THRESHOLD | VE stable, within expected bounds | Updates domain parameters |
| BORDERLINE | Inconclusive - some indicators elevated | Excluded from calibration |
| ABOVE_THRESHOLD | Clear drift detected | May trigger VE threshold update |

### Why Split Slope Ratio? (Heavy/Severe Only)

Distinguishes between:
- **Stable drift** (ratio ≈ 1.0): Constant slow drift throughout
- **Accelerating drift** (ratio > 1.2): Slope increases in second half
- **Decelerating drift** (ratio < 1.0): Slope decreases (fatigue response)

Accelerating drift is more indicative of exceeding threshold.

**Not used for Moderate domain** because the VO2 slow component in heavy domain plateaus after 6-10 minutes, so an accelerating slope pattern in a "moderate" run would indicate severe domain, not heavy. The expected_drift threshold alone is sufficient for moderate classification.

### Why CUSUM Recovery Matters?

A recovered CUSUM indicates:
- Brief excursion above threshold that self-corrected
- May be physiological response to temporary stress
- Less indicative of true threshold violation

---

## Interval Results Output

| Field | Description |
|-------|-------------|
| `status` | BELOW_THRESHOLD, BORDERLINE, or ABOVE_THRESHOLD |
| `baseline_ve` | Calibration period mean VE (L/min) |
| `avg_ve` | Post-calibration average VE |
| `ve_drift_pct` | Overall slope as % of baseline per minute |
| `peak_cusum` | Maximum CUSUM value reached |
| `final_cusum` | CUSUM value at interval end |
| `cusum_threshold` | The h threshold used |
| `alarm_time` | Time when CUSUM first exceeded h (if any) |
| `cusum_recovered` | Whether CUSUM dropped below h/2 |
| `slope1_pct` | First segment slope (%/min) |
| `slope2_pct` | Second segment slope (%/min) |
| `split_slope_ratio` | slope2 / slope1 |
| `phase3_onset_rel` | Phase III onset time (seconds from interval start) |
| `hinge2_time_rel` | Second hinge time (seconds from interval start) |
| `is_ceiling_based` | Whether ceiling-based analysis was used |
| `is_segmented` | Whether segmented analysis was used |
| `observed_sigma_pct` | MADSD-calculated noise level |

---

## Visual Representation

### Chart Elements

| Element | Source | Description |
|---------|--------|-------------|
| VE binned line | Post-filtering | Filtered VE over time (with gradient) |
| Breath dots | Post-median only | Individual breath values |
| Segment 1 line | Hinge model | Phase I+II (ramp-up) slope |
| Segment 2 line | Overall slope (Moderate) or Second hinge (Heavy/Severe) | Phase III slope |
| Segment 3 line | Second hinge (Heavy/Severe only) | Late Phase III slope |
| CUSUM line | Accumulation | Green until alarm, red after |
| Interval shading | Classification | Green/yellow/red background |

**Note**: Moderate runs display only Segment 1 (ramp) and Segment 2 (overall drift). Heavy/Severe runs display Segment 1, Segment 2 (early Phase III), and Segment 3 (late Phase III).

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-02 | Initial specification |
| 2026-01-03 | Moderate domain uses same Phase III detection as Heavy/Severe (90s-180s window, 60s calibration); removed max_drift and split_ratio from classification; simplified to single expected_drift threshold |
