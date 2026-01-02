# Signal Processing Specification

> **Status**: Implemented
> **Last Updated**: 2026-01-02

---

## Overview

This document specifies the three-stage hybrid filtering pipeline used to transform raw breath-by-breath VE (minute ventilation) data into a clean, uniformly-sampled time series suitable for CUSUM analysis.

---

## Input Data

**Source**: Breath-by-breath respiratory data from wearable device (COSMED/PNOE)

| Field | Description | Units |
|-------|-------------|-------|
| `breath_time` | Timestamp of each breath | seconds |
| `ve_raw` | Raw minute ventilation | L/min |

**Characteristics**:
- Irregular sampling (breath rate varies ~40-70 breaths/min)
- Contains physiological artifacts (coughs, swallows, talking)
- Contains sensor artifacts (connection drops, measurement errors)
- Accumulation rates vary with breathing frequency

---

## Three-Stage Filtering Pipeline

```
Raw VE (breath domain)
         │
         ▼
┌─────────────────────────┐
│   Stage 1: Median       │
│   (breath domain)       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Stage 2: Time Binning │
│   (→ time domain)       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Stage 3: Hampel       │
│   (time domain)         │
└─────────────────────────┘
         │
         ▼
Clean VE (uniform time series)
```

---

## Stage 1: Rolling Median Filter

**Domain**: Breath (irregular sampling)

**Purpose**: Remove single-breath outliers and non-physiological spikes (coughs, sensor errors)

### Algorithm

For each breath `i`, compute the median of VE values in a centered window:

```
ve_filtered[i] = median(ve_raw[i - w/2 : i + w/2])
```

Where `w` = window size in breaths

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `median_window` | 9 breaths | Filter window size |

### Rationale

- **9-breath window** at ~55 breaths/min covers approximately 10 seconds
- Provides robust outlier rejection while preserving physiological trends
- Median is robust to extreme values (50% breakdown point)
- Operates in breath domain before time standardization

### Edge Handling

Uses "nearest" mode - edge values are extended to fill the window.

---

## Stage 2: Time Binning

**Domain**: Breath → Time (converts irregular to uniform sampling)

**Purpose**: Standardize accumulation rate by averaging breaths within fixed time intervals

### Algorithm

1. **Create bins**: Divide the recording into fixed-duration bins starting at `t_min`
2. **Assign breaths**: Each breath is assigned to the bin containing its timestamp
3. **Compute means**: For each bin, compute mean VE of all assigned breaths
4. **Interpolate gaps**: Fill empty bins via linear interpolation

```
bin_edges: [t_min, t_min + Δt, t_min + 2Δt, ...]

For each bin [t_start, t_end):
    bin_value = mean(ve_filtered where t_start ≤ breath_time < t_end)

If bin is empty:
    bin_value = linear_interpolate(neighboring_bins)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_size` | 4.0 seconds | Duration of each time bin |

### Rationale

- **4-second bins** at ~55 breaths/min capture ~3-4 breaths per bin
- Provides reliable averaging while maintaining temporal resolution
- Standardizes the "accumulation rate" regardless of breathing frequency
- Converts from irregular breath-domain to uniform time-domain sampling

### Empty Bin Handling

| Scenario | Action |
|----------|--------|
| Empty bin with ≥2 valid neighbors | Linear interpolation |
| Empty bin with 1 valid neighbor | Fill with that neighbor's value |
| No valid bins | Fill all with overall mean |

---

## Stage 3: Hampel Filter

**Domain**: Time (uniform sampling)

**Purpose**: Remove consecutive outlier clusters that survive median filtering

### Algorithm

For each bin `i` with value `v` at time `t`:

1. **Define window**: All bins within `[t - W/2, t + W/2]`
2. **Compute statistics**:
   - `window_median` = median of window values
   - `window_MAD` = median(|values - window_median|)
   - `scaled_MAD` = 1.4826 × window_MAD
3. **Test for outlier**:
   - `deviation` = |v - window_median|
   - `sigma_dev` = deviation / scaled_MAD
4. **Replace if outlier**:
   - If `sigma_dev > n_sigma`: replace `v` with `window_median`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hampel_window_sec` | 30.0 seconds | Window size (centered on each point) |
| `hampel_n_sigma` | 3.0 | Outlier threshold in MAD-scaled units |

### Rationale

- **Handles consecutive outliers**: Median filter fails when >50% of window is outliers
- **30-second window**: Long enough for robust statistics, short enough to preserve trends
- **MAD scaling**: 1.4826 factor converts MAD to standard deviation for normal distributions
- **3-sigma threshold**: Balances outlier removal vs. preserving legitimate variation

### Edge Cases

| Condition | Action |
|-----------|--------|
| Window has < 3 points | Skip (keep original value) |
| scaled_MAD < 10⁻⁶ | Skip (all values identical) |

---

## Output

The pipeline produces three outputs:

| Output | Description | Used For |
|--------|-------------|----------|
| `ve_median` | After Stage 1 only | Visualization (breath dots on chart) |
| `bin_times` | Uniform time array | X-axis for analysis |
| `ve_binned` | After all 3 stages | CUSUM analysis, slope fitting |

---

## Mathematical Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| MAD scale factor | 1.4826 | Converts MAD to σ for normal distribution |

The MAD scale factor arises from:
```
For X ~ Normal(μ, σ):
MAD(X) = median(|X - median(X)|) ≈ 0.6745σ
Therefore: σ ≈ MAD / 0.6745 = 1.4826 × MAD
```

---

## Implementation Notes

### Why Three Stages?

| Stage | Handles | Fails When |
|-------|---------|------------|
| Median | Single-breath spikes | >50% of window is bad |
| Binning | Irregular sampling | N/A (transformation only) |
| Hampel | Consecutive outliers | Very long outlier sequences |

### Why This Order?

1. **Median first**: Most effective in breath domain before aggregation
2. **Binning second**: Must convert to time domain before Hampel (needs uniform sampling)
3. **Hampel third**: Catches clusters that slip through median filter

### Performance Considerations

- Stage 1: O(n × w) where n = breaths, w = window size
- Stage 2: O(n) single pass through breaths
- Stage 3: O(m × W/Δt) where m = bins, W = window size, Δt = bin size

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-02 | Initial specification |
