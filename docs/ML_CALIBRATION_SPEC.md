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

**Note**: All three domains can have intervals. ✅ Enum values renamed from legacy `VT1_STEADY`/`VT2_INTERVAL`.

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

### Current iOS App Architecture

The iOS app (Flutter/Dart) stores VT thresholds locally:
- **Storage**: SharedPreferences with keys `vt1_ve` and `vt2_ve`
- **Defaults**: VT1 = 60.0 L/min, VT2 = 80.0 L/min
- **State Management**: `AppState` class extends `ChangeNotifier` (Provider pattern)
- **Update Methods**: `setVt1Ve()` and `setVt2Ve()` persist and notify listeners

The iOS app also uses **sigma_pct** in CUSUM calculations (`cusum_processor.dart`):
- **Moderate**: 10.0% (hardcoded)
- **Heavy/Severe**: 5.0% (hardcoded)
- Used to calculate: `k = 0.5 × sigma`, `h = 5.0 × sigma`

### Current Cloud Integration

The app has **upload-only** capability:
- `uploadToCloud()` POSTs workout data to Railway endpoint
- **No authentication** (no API keys or tokens)
- **No download/sync** for thresholds or sigma values

### Required iOS App Changes

To enable automatic threshold sync, add the following to the iOS app:

#### 1. New API Service Method (in `workout_data_service.dart`)

```dart
/// Fetches calibrated parameters from the cloud
/// Returns null if no calibration data exists or on error
Future<Map<String, dynamic>?> fetchCalibratedParams(String userId) async {
  try {
    final response = await http.get(
      Uri.parse('$_baseUrl/api/calibration/params?user_id=$userId'),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return {
        'vt1_ve': data['vt1_ve']?.toDouble(),
        'vt2_ve': data['vt2_ve']?.toDouble(),
        'sigma_pct_moderate': data['sigma_pct_moderate']?.toDouble(),
        'sigma_pct_heavy': data['sigma_pct_heavy']?.toDouble(),
        'sigma_pct_severe': data['sigma_pct_severe']?.toDouble(),
      };
    }
    return null;
  } catch (e) {
    print('Error fetching calibrated params: $e');
    return null;
  }
}
```

#### 2. Add Sigma Storage to AppState (in `app_state.dart`)

```dart
// Add new keys and state variables
static const String _sigmaModeratePctKey = 'sigma_pct_moderate';
static const String _sigmaHeavyPctKey = 'sigma_pct_heavy';
static const String _sigmaSeverePctKey = 'sigma_pct_severe';

double _sigmaPctModerate = 10.0;  // Default
double _sigmaPctHeavy = 5.0;      // Default
double _sigmaPctSevere = 5.0;     // Default

double get sigmaPctModerate => _sigmaPctModerate;
double get sigmaPctHeavy => _sigmaPctHeavy;
double get sigmaPctSevere => _sigmaPctSevere;

Future<void> setSigmaPctModerate(double value) async {
  _sigmaPctModerate = value;
  final prefs = await SharedPreferences.getInstance();
  await prefs.setDouble(_sigmaModeratePctKey, value);
  notifyListeners();
}

// Similar setters for Heavy and Severe...
```

#### 3. Sync Method (in `app_state.dart`)

```dart
/// Syncs calibrated params from cloud
/// Shows confirmation dialog if VE change >= 1 L/min
/// Sigma updates are applied silently
Future<void> syncParamsFromCloud(BuildContext context) async {
  final workoutService = WorkoutDataService();
  final calibrated = await workoutService.fetchCalibratedParams(_userId);

  if (calibrated == null) return;

  // Check for significant VE threshold changes
  final vt1Change = (calibrated['vt1_ve']! - _vt1Ve).abs();
  final vt2Change = (calibrated['vt2_ve']! - _vt2Ve).abs();

  if (vt1Change >= 1.0 || vt2Change >= 1.0) {
    // Show confirmation dialog for VE thresholds only
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Update Thresholds?'),
        content: Text(
          'Calibration suggests:\n'
          'VT1: ${calibrated['vt1_ve']!.toStringAsFixed(1)} L/min\n'
          'VT2: ${calibrated['vt2_ve']!.toStringAsFixed(1)} L/min\n\n'
          'Apply these changes?'
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text('No')),
          TextButton(onPressed: () => Navigator.pop(ctx, true), child: Text('Yes')),
        ],
      ),
    );

    if (confirmed == true) {
      if (calibrated['vt1_ve'] != null) await setVt1Ve(calibrated['vt1_ve']!);
      if (calibrated['vt2_ve'] != null) await setVt2Ve(calibrated['vt2_ve']!);
    }
  }

  // Sigma values are updated silently (no user prompt needed)
  if (calibrated['sigma_pct_moderate'] != null) {
    await setSigmaPctModerate(calibrated['sigma_pct_moderate']!);
  }
  if (calibrated['sigma_pct_heavy'] != null) {
    await setSigmaPctHeavy(calibrated['sigma_pct_heavy']!);
  }
  if (calibrated['sigma_pct_severe'] != null) {
    await setSigmaPctSevere(calibrated['sigma_pct_severe']!);
  }
}
```

#### 4. Update CUSUM Processor (in `cusum_processor.dart`)

```dart
// Change from hardcoded values:
// final sigmaPct = runType == RunType.moderate ? 10.0 : 5.0;

// To using calibrated values from AppState:
double getSigmaPct(RunType runType, AppState appState) {
  switch (runType) {
    case RunType.moderate:
      return appState.sigmaPctModerate;
    case RunType.heavy:
      return appState.sigmaPctHeavy;
    case RunType.severe:
      return appState.sigmaPctSevere;
  }
}
```

#### 5. Trigger Sync on App Launch (in `main.dart`)

```dart
// After app initialization, attempt to sync calibrated params
WidgetsBinding.instance.addPostFrameCallback((_) {
  final appState = Provider.of<AppState>(context, listen: false);
  appState.syncParamsFromCloud(context);
});
```

#### 6. User ID Considerations

Currently the app has no user authentication. Options:
- **Device ID**: Use a unique device identifier (simple but not cross-device)
- **Anonymous ID**: Generate and persist a UUID on first launch
- **Future**: Add proper user authentication for cross-device sync

### Backend Endpoint Required

Add to web app API:

```
GET /api/calibration/params?user_id={user_id}

Response:
{
  "vt1_ve": 62.5,
  "vt2_ve": 85.0,
  "sigma_pct_moderate": 9.2,
  "sigma_pct_heavy": 4.8,
  "sigma_pct_severe": 5.1,
  "last_updated": "2026-01-02T12:00:00Z"
}
```

---

## Code Changes Required

1. ✅ **Rename enums**: `VT1_STEADY` → `MODERATE`, `VT2_INTERVAL` → `HEAVY` (complete)
2. ✅ **Allow intervals for all domains** (complete)
3. ✅ **Backend calibration service** - NIG math with forgetting factor (`api/services/calibration.py`)
4. ✅ **Storage/database schema** - Firebase Storage per-user JSON (`api/models/schemas.py`)
5. ✅ **API endpoints** - CRUD + iOS sync (`api/routers/calibration.py`)
6. ✅ **Frontend UI**: VE approval popup (complete), exclusion checkboxes (pending)
7. **Mobile sync** - Code examples provided, iOS app changes required

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-02 | Initial specification |
| 2026-01-02 | Added iOS app sync recommendations with code examples |
| 2026-01-02 | Expanded sync to include sigma_pct per domain (not just VE thresholds) |
| 2026-01-02 | Implemented backend calibration service with NIG algorithm |
| 2026-01-02 | Implemented frontend VE approval dialog and calibration integration |
