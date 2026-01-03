# Classification Specification

> **Status**: Implemented
> **Last Updated**: 2026-01-03

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

**Recovery condition**: `Sᵢ ≤ h/2` (after alarm was triggered)

### CUSUM State Transitions

The CUSUM can transition between states multiple times within an interval:

| State | Condition | Line Color |
|-------|-----------|------------|
| Normal | `S ≤ h` (never triggered or recovered) | Green |
| Alarm | `S > h` | Orange |
| Recovered | `S ≤ h/2` (after alarm) | Green |

**Transition tracking**: Each alarm trigger and recovery is recorded with a timestamp, allowing multiple green→orange→green transitions to be visualized.

**Final state determination**:
- `cusum_recovered = TRUE` if alarm was triggered AND ended in Normal/Recovered state
- `cusum_recovered = FALSE` if alarm was triggered AND ended in Alarm state

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
| `max_drift_pct` | 1.0%/min | 3.0%/min |

---

## Slope Calculation

### Single Slope Model (All Domains)

All domains use a **single slope** model fitted using **Huber regression** (robust to outliers) on post-calibration data.

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

Where `δ = 5.0` (Huber delta parameter, configurable in Advanced settings)

**Output**:
- `slope`: VE change per minute (L/min per minute)
- `slope_pct`: `(slope / cal_ve_mean) × 100` (% per minute)

**Visualization**: The slope line is anchored at Phase III onset (visual continuity) but the slope value is calculated from post-calibration data.

---

## Classification Tree

### Moderate Domain (VT1) Classification

**Inputs**:
- `sustained_alarm`: CUSUM triggered AND NOT recovered
- `overall_slope_pct`: Total drift rate (%/min)

**Thresholds** (from cloud calibration):
- `low_threshold` = expected_drift_pct (0.3%/min default)
- `high_threshold` = max_drift_pct (1.0%/min default)

**Decision Tree**:

```
IF (sustained_alarm = FALSE):
│   (No cusum alarm OR cusum recovered)
│
├── IF overall_slope_pct < low_threshold:
│   └── BELOW_THRESHOLD (green)
│
├── ELSE IF overall_slope_pct < high_threshold:
│   └── BORDERLINE (yellow)
│
└── ELSE (slope ≥ high_threshold):
    └── ABOVE_THRESHOLD (red)

ELSE (sustained_alarm = TRUE):
│   (Cusum triggered AND NOT recovered)
│
├── IF overall_slope_pct < low_threshold:
│   └── BORDERLINE (yellow)
│
└── ELSE (slope ≥ low_threshold):
    └── ABOVE_THRESHOLD (red)
```

---

### Heavy/Severe Domain (VT2) Classification

**Inputs**:
- `sustained_alarm`: CUSUM triggered AND NOT recovered
- `overall_slope_pct`: Total drift rate (%/min)

**Thresholds** (from cloud calibration):
- `low_threshold` = expected_drift_pct (1.0%/min default)
- `high_threshold` = max_drift_pct (3.0%/min default)

**Decision Tree**:

```
IF (sustained_alarm = FALSE):
│   (No cusum alarm OR cusum recovered)
│
├── IF overall_slope_pct < low_threshold:
│   └── BELOW_THRESHOLD (green)
│
├── ELSE IF overall_slope_pct < high_threshold:
│   └── BORDERLINE (yellow)
│
└── ELSE (slope ≥ high_threshold):
    └── ABOVE_THRESHOLD (red)

ELSE (sustained_alarm = TRUE):
│   (Cusum triggered AND NOT recovered)
│
├── IF overall_slope_pct < low_threshold:
│   └── BORDERLINE (yellow)
│
└── ELSE (slope ≥ low_threshold):
    └── ABOVE_THRESHOLD (red)
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

| Status | Color | Meaning | Calibration Impact |
|--------|-------|---------|-------------------|
| BELOW_THRESHOLD | Green | VE stable, within expected bounds | Updates domain parameters |
| BORDERLINE | Yellow | Inconclusive - some indicators elevated | Excluded from calibration |
| ABOVE_THRESHOLD | Red | Clear drift detected | May trigger VE threshold update |

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
| `cusum_recovered` | Whether CUSUM ended in recovered state |
| `cusum_transitions` | Array of {time, is_alarm} for all state changes |
| `phase3_onset_rel` | Phase III onset time (seconds from interval start) |
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
| Slope line (Segment 1) | Hinge model | Phase II (ramp-up) from start to Phase III onset |
| Slope line (Segment 2) | Single slope | Phase III drift from Phase III onset to interval end |
| CUSUM line | Accumulation | Multi-segment: green (normal) ↔ orange (alarm) |
| Interval shading | Classification | Green/yellow/red background |

### CUSUM Line Coloring

The CUSUM line changes color at each state transition:

| Segment | State | Color |
|---------|-------|-------|
| Before first alarm | Normal | Green |
| After alarm trigger | Alarm | Orange |
| After recovery | Recovered | Green |
| After re-trigger | Alarm | Orange |
| ... | ... | ... |

Multiple transitions are possible within a single interval.

### Bottom Box Font Colors

**CUSUM (Average VE)**:
| Final State | Font Color |
|-------------|------------|
| Never triggered | Green |
| Triggered + recovered | Yellow |
| Triggered + not recovered | Red |

**Slope (%/min)**:
| Slope Value | Font Color |
|-------------|------------|
| < low_threshold | Green |
| low_threshold to < high_threshold | Yellow |
| ≥ high_threshold | Red |

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-02 | Initial specification |
| 2026-01-03 | Moderate domain uses same Phase III detection as Heavy/Severe (90s-180s window, 60s calibration) |
| 2026-01-03 | Removed split_slope_ratio from all classification logic |
| 2026-01-03 | Simplified to single slope model for all domains |
| 2026-01-03 | Added max_drift_pct for Moderate domain (1.0%/min default) |
| 2026-01-03 | Updated classification tree: both domains use low_threshold and high_threshold |
| 2026-01-03 | Added cusum_transitions for multi-segment CUSUM line visualization |
| 2026-01-03 | CUSUM line now shows green→orange→green transitions at each alarm/recovery |
| 2026-01-03 | Added font color specifications for bottom box (CUSUM and Slope) |
| 2026-01-03 | BORDERLINE status color changed from orange/amber to yellow |
