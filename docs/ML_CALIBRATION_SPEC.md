# ML Calibration System - Implementation Specification

> **Status**: Implemented
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

### VE Threshold Calibration: Anchor & Pull Method

For VE thresholds (VT1/VT2), we use an **Anchor & Pull** approach with **κ=4**:

- **Anchor**: The user-approved threshold acts as a strong prior worth 4 virtual observations
- **Pull**: Observed avg_ve values during unexpected outcomes pull the posterior mean toward physiological reality
- **Prompt**: User is prompted when posterior mean diverges ≥1 L/min from anchor

**Why Anchor & Pull vs Delta Accumulation:**

| Approach | VT1=60, observe 75 L/min | VT1=60, 3× observe 75 L/min |
|----------|--------------------------|------------------------------|
| Delta Accumulation | +0.5-1.0 nudge → 61 | ~62.5 (15+ runs to reach 75) |
| Anchor & Pull (κ=4) | Posterior ≈ 63 | Posterior ≈ 66-67 |

Benefits:
1. **Physiological Directness**: avg_ve IS the measurement, not just a directional signal
2. **Convergence Speed**: Bad ramp test corrects in 2-3 runs, not 15-20
3. **Mathematical Robustness**: κ scales step size based on certainty, no magic numbers

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

### Anchor & Pull Mechanics:

When an unexpected outcome occurs:
1. **Update posterior**: Observed avg_ve is fed into NIG posterior using anchored update
2. **Calculate posterior mean**: Blend anchor (κ=4) with observation-derived posterior
3. **Check divergence**: If `|posterior_mean - current_value| ≥ 1 L/min`, prompt user

```
anchored_mean = (anchor_κ × current_value + obs_κ × obs_mean) / (anchor_κ + obs_κ)
```

### User Approval/Rejection:
- **Approve**: New value becomes the anchor; posterior resets to fresh state
- **Reject**: Current value remains anchor; posterior resets (starts fresh tracking)

Both actions reset the posterior, meaning future observations start fresh relative to the (new or existing) anchor. This prevents stale observations from accumulating indefinitely.

---

## Minimum Data Requirements

- **Minimum 3 runs** per domain before calibration applies
- **Soft blending** for runs 3-5 (gradual transition from defaults to calibrated)
- After 5+ runs: fully calibrated values used

---

## User Controls

1. **Calibration On/Off toggle** - on startup screen
   - Toggle to enable/disable ML calibration globally
   - When disabled: Learned data is preserved, system defaults are used
   - When re-enabled: Previously learned values are restored as new baseline
   - Persisted in cloud per user
   - Endpoint: `POST /api/calibration/toggle`

2. **VT Threshold Sidebar** - auto-populates from calibration
   - On sidebar load, VT1/VT2 thresholds sync from cloud calibration
   - User can manually adjust values for specific run analysis
   - Changes are local until explicitly synced

3. **Sync to Calibration button** - in VT Thresholds card
   - Pushes current sidebar VT1/VT2 values to cloud calibration
   - Resets Bayesian posterior to new anchor values
   - Useful when user wants to "lock in" manual adjustments

4. **Restore to Last Calibration button** - in VT Thresholds card
   - Fetches current cloud-calibrated VT1/VT2 values
   - Restores sidebar to cloud values (discards local changes)
   - Does not affect the Bayesian posterior

5. **Calibration exclusion checkbox** - per run in cloud run table (pending)
   - Allows user to exclude specific runs from calibration
   - Useful for unusual/outlier sessions

6. **VE threshold approval popup** - when cumulative change ≥ ±1 L/min
   - Shows proposed new threshold
   - User can accept or reject
   - Both approval and rejection reset the Bayesian posterior (starts fresh tracking)

7. **Manual threshold override** - from web app or iOS app
   - User can directly edit VT1/VT2 threshold values at any time
   - Change syncs to cloud immediately (bidirectional)
   - Resets Bayesian posterior to new anchor value
   - Future calibration observations start fresh from this baseline
   - Endpoint: `POST /api/calibration/set-ve-threshold`

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

    # Calibration toggle
    enabled: bool = True  # When False, system defaults used; learned data preserved

    # Metadata
    last_updated: datetime
    run_counts: Dict[str, int]  # qualifying runs per domain

class DomainPosterior:
    expected_drift: NIGPosterior
    max_drift: NIGPosterior
    sigma: NIGPosterior
    split_ratio: NIGPosterior

class VEThresholdState:
    current_value: float     # User-approved threshold (anchor)
    posterior: NIGPosterior  # Observation-derived posterior
    anchor_kappa: float = 4.0  # Virtual sample size for anchoring

class NIGPosterior:
    mu: float      # mean
    kappa: float   # precision of mean
    alpha: float   # shape
    beta: float    # scale
    n_obs: int     # observation count
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calibration/params` | GET | Get calibrated params for iOS sync (includes `enabled` flag) |
| `/api/calibration/state` | GET | Get full user's calibration state |
| `/api/calibration/update` | POST | Update calibration from run result |
| `/api/calibration/reset` | POST | Reset to defaults |
| `/api/calibration/approve-ve` | POST | User approves/rejects VE threshold change |
| `/api/calibration/set-ve-threshold` | POST | Manual threshold override (resets anchor) |
| `/api/calibration/blended-params` | GET | Get blended params for a run type |
| `/api/calibration/toggle` | POST | Enable/disable calibration (preserves learned data) |

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

#### 7. Sync Manual Overrides to Cloud (in `workout_data_service.dart`)

When the user manually changes a VT threshold in the iOS app, sync it back to the cloud so the web app and calibration system stay in sync:

```dart
/// Syncs a manual threshold change to the cloud
/// This resets the Bayesian anchor to the new value
Future<bool> syncThresholdToCloud(String userId, String threshold, double value) async {
  try {
    final response = await http.post(
      Uri.parse('$_baseUrl/api/calibration/set-ve-threshold'
          '?user_id=$userId&threshold=$threshold&value=$value'),
    );
    return response.statusCode == 200;
  } catch (e) {
    print('Error syncing threshold to cloud: $e');
    return false;
  }
}
```

#### 8. Update Threshold Setters to Sync (in `app_state.dart`)

Modify the existing `setVt1Ve()` and `setVt2Ve()` methods to sync to cloud:

```dart
Future<void> setVt1Ve(double value) async {
  _vt1Ve = value;
  final prefs = await SharedPreferences.getInstance();
  await prefs.setDouble(_vt1VeKey, value);
  notifyListeners();

  // Sync to cloud (resets Bayesian anchor)
  final service = WorkoutDataService();
  await service.syncThresholdToCloud(userId, 'vt1', value);
}

Future<void> setVt2Ve(double value) async {
  _vt2Ve = value;
  final prefs = await SharedPreferences.getInstance();
  await prefs.setDouble(_vt2VeKey, value);
  notifyListeners();

  // Sync to cloud (resets Bayesian anchor)
  final service = WorkoutDataService();
  await service.syncThresholdToCloud(userId, 'vt2', value);
}
```

**Important**: When a manual override is synced to the cloud, the Bayesian posterior resets. This means future calibration observations start fresh from the new anchor value.

### Bidirectional Sync Summary

| Direction | Trigger | What Syncs | Posterior Reset? |
|-----------|---------|------------|------------------|
| Cloud → iOS | App launch | VT1, VT2, sigma×3 | No |
| iOS → Cloud | Manual threshold change | VT1 or VT2 | Yes |
| Web → Cloud | Analysis complete | Calibration update | Only on prompt |
| Web → Cloud | Manual threshold change | VT1 or VT2 | Yes |

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
| 2026-01-02 | Replaced delta accumulation with Anchor & Pull Bayesian approach (κ=4) |
| 2026-01-02 | Added iOS → cloud sync for manual threshold overrides (bidirectional sync) |
| 2026-01-02 | Added calibration On/Off toggle with enable/disable endpoint |
| 2026-01-02 | Added sidebar integration: VT thresholds sync from cloud calibration on load |
| 2026-01-02 | Added "Sync to Calibration" and "Restore to Last Calibration" buttons |
