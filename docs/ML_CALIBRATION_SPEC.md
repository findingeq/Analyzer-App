# ML Calibration System - Implementation Specification

> **Status**: Implemented
> **Last Updated**: 2026-01-03

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

### Decaying Anchor Approach (All Parameters)

All calibrated parameters use a **Decaying Anchor** approach:

- **Anchor**: Starting point (system default or user-set value) with initial κ=4
- **Decay**: Anchor κ decays with same forgetting factor (λ=0.95) as observation posterior
- **Convergence**: Early runs have anchor stability (~80% weight), later runs converge to observations

**Why Decaying Anchor:**

| Run # | Anchor κ | Obs κ | Anchor Weight | Observation Weight |
|-------|----------|-------|---------------|-------------------|
| 1     | 4.0      | 1.0   | 80%           | 20%               |
| 3     | 3.4      | 2.7   | 56%           | 44%               |
| 5     | 3.1      | 3.9   | 44%           | 56%               |
| 10    | 2.4      | 6.6   | 27%           | 73%               |
| 20    | 1.5      | 10.2  | 13%           | 87%               |

Benefits:
1. **Early Stability**: Anchor provides ~80% stability for first few runs
2. **Full Convergence**: Anchor fades to ~10-15% over time, letting observations dominate
3. **Manual Override**: User can set a new anchor at any time, resetting observation posterior

**Formula:**
```
effective_value = (anchor_κ × anchor_value + obs_κ × obs_mu) / (anchor_κ + obs_κ)
```

After each observation:
- `anchor_κ' = λ × anchor_κ` (decay anchor)
- `obs_κ' = λ × obs_κ + 1` (decay + add new observation)

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

| Parameter | Moderate | Heavy | Severe | Ordinal Constraint |
|-----------|----------|-------|--------|-------------------|
| `expected_drift_pct` | ✓ | ✓ | ✓ | Moderate < Heavy < Severe |
| `max_drift_pct` | ✓ (derived) | ✓ (derived) | — | Derived from next domain |
| `sigma_pct` | ✓ | ✓ | ✓ | **NONE** |

**Note**:
- Both Moderate and Heavy domains use expected_drift and max_drift for classification
- Severe calibrates expected_drift to inform Heavy's max_drift ceiling
- **max_drift is derived, not directly calibrated**: moderate.max_drift = heavy.expected_drift, heavy.max_drift = severe.expected_drift

### Global VE Thresholds:

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `vt1_ve_ceiling` | VT1 VE threshold (L/min) | VT1 < VT2 |
| `vt2_ve_ceiling` | VT2 VE threshold (L/min) | VT1 < VT2 |

### Default Parameter Values:

| Parameter | Moderate | Heavy | Severe |
|-----------|----------|-------|--------|
| `expected_drift_pct` | 0.3 | 1.0 | 2.0 |
| `max_drift_pct` | 1.0 | 3.0 | 5.0 |
| `sigma_pct` | 7.0 | 4.0 | 4.0 |

---

## Sigma Calculation: MADSD Method

Sigma (noise level) is calculated from actual data using **MAD of Successive Differences (MADSD)**:

```python
def calculate_madsd_sigma(ve_binned):
    diffs = np.diff(ve_binned)           # First differences
    median_diff = np.median(diffs)        # Median of differences
    mad = np.median(np.abs(diffs - median_diff))  # MAD
    sigma = mad * 1.4826 / np.sqrt(2)     # Scale to normal sigma
    return sigma
```

**Why MADSD?**
- **Robust to drift**: Differencing removes linear trends
- **Robust to outliers**: MAD has 50% breakdown point
- **Applied post-filtering**: Uses VE data after 3-stage hybrid filtering (median → time-bin → Hampel)

The observed sigma is calculated as a percentage of baseline VE and fed into calibration.

---

## Drift/Sigma Calibration Logic

### Run-Level Calibration with Majority Rule

Each run (session) counts as **ONE calibration sample**, regardless of number of intervals. Multi-interval runs use majority-based logic:

1. **Filter intervals**: Only intervals ≥ 6 minutes are considered
2. **Majority check**: >50% of filtered intervals must share the same classification (ABOVE_THRESHOLD or BELOW_THRESHOLD)
3. **If majority exists**:
   - Calculate **averaged values** (drift_pct, sigma_pct, avg_ve) from majority intervals only
   - Check if majority classification matches domain expectations
   - If eligible: update posteriors with averaged values, increment run_count by 1
4. **If no majority**: Run is excluded from calibration entirely (mixed results are too noisy)
5. BORDERLINE intervals count toward denominator but never toward majority

### Domain-Specific Update Rules:

| Run Domain | Expected Outcome | Parameters Updated |
|------------|------------------|-------------------|
| **Moderate** | BELOW_THRESHOLD | Moderate: expected_drift, max_drift, sigma |
| **Heavy** | BELOW_THRESHOLD | Heavy: expected_drift, max_drift, sigma |
| **Severe** | ABOVE_THRESHOLD | Severe: expected_drift, sigma; **Heavy: max_drift** |

### Cross-Domain max_drift Calibration:

Heavy domain's max_drift is informed by Severe's expected_drift:

- **max_drift_heavy** ← Severe's expected_drift (if you drift like Severe, you're above VT2)

### Severe Domain Specifics:

Severe runs only calibrate **expected_drift** and **sigma** (not max_drift):
- **expected_drift**: Used to set Heavy's max_drift ceiling
- **sigma**: Used for CUSUM sensitivity in Severe runs
- **max_drift**: Not needed (no domain above Severe)

### Domain Isolation (No Cross-Pollination):

Domain models are kept pure for expected_drift and sigma. If a Severe run unexpectedly shows low drift, this triggers VT2 threshold adjustment, not Heavy parameter updates.

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

### Auto-Update Mechanics:

VE thresholds use the same decaying anchor approach as drift/sigma:

1. **Update posterior**: Observed avg_ve is fed into NIG posterior
2. **Decay anchor**: anchor_κ decays with forgetting factor λ=0.95
3. **Auto-update current_value**: The threshold automatically updates to the anchored mean

```
current_value = (anchor_κ × anchor_value + obs_κ × obs_mu) / (anchor_κ + obs_κ)
```

**No user prompts**: Thresholds auto-populate from cloud calibration. Users can manually override at any time.

### Multi-Interval Runs:

For runs with multiple intervals, VE threshold calibration uses the same majority-based logic as drift/sigma calibration:

1. **Majority classification** determines the run's overall classification
2. **Averaged avg_ve** from majority intervals is used for unexpected outcome check
3. Example: Moderate run with 4/7 intervals ABOVE_THRESHOLD → majority is ABOVE, use avg of those 4 intervals' avg_ve values

### Manual Override:
- **Manual threshold change**: User sets new value → becomes new anchor, observation posterior resets
- This allows users to "lock in" a specific threshold value at any time
- Future observations start fresh relative to the new anchor

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

2. **Sidebar Auto-Sync from Calibration**
   - On sidebar load, all calibrated parameters sync from cloud:
     - VT1/VT2 thresholds
     - Sigma % (VT1=Moderate, VT2=Heavy)
     - Expected Drift % (VT1=Moderate, VT2=Heavy)
     - Max Drift % (VT1=Moderate, VT2=Heavy)
   - User can manually adjust values for specific run analysis
   - Changes are local until explicitly synced

3. **Sync to Calibration button** - in VT Thresholds card
   - Pushes current sidebar VT1/VT2 values to cloud calibration
   - Resets Bayesian posterior to new anchor values
   - Useful when user wants to "lock in" manual adjustments
   - Note: Advanced params (sigma, drift, etc.) are not synced back (read-only from calibration)

4. **Restore from Calibration button** - in VT Thresholds card
   - Fetches all cloud-calibrated values (VT thresholds + advanced params)
   - Restores sidebar to cloud values (discards local changes)
   - Does not affect the Bayesian posterior

5. **Calibration exclusion checkbox** - per run in cloud run table (pending)
   - Allows user to exclude specific runs from calibration
   - Useful for unusual/outlier sessions

6. **Manual threshold override** - from web app or iOS app
   - User can directly edit VT1/VT2 threshold values at any time
   - Change syncs to cloud immediately (bidirectional)
   - Resets both anchor and observation posterior to new value
   - Future calibration observations start fresh from this baseline
   - Endpoint: `POST /api/calibration/set-ve-threshold`

**Note**: VE thresholds auto-update without user prompts. The approval popup has been removed - calibrated values automatically populate from the cloud.

---

## Storage Schema (per user)

```python
class CalibrationState:
    # Per-domain posteriors (each using decaying anchor)
    moderate: DomainPosterior
    heavy: DomainPosterior
    severe: DomainPosterior

    # Global VE thresholds (using decaying anchor)
    vt1_ve: VEThresholdState
    vt2_ve: VEThresholdState

    # Calibration toggle
    enabled: bool = True  # When False, system defaults used; learned data preserved

    # Metadata
    last_updated: datetime
    run_counts: Dict[str, int]  # qualifying runs per domain (1 per run, not per interval)

class DomainPosterior:
    expected_drift: AnchoredNIGPosterior  # Calibrated from observations
    max_drift: AnchoredNIGPosterior       # Derived from next domain's expected_drift
    sigma: AnchoredNIGPosterior           # Calibrated from observations

class AnchoredNIGPosterior:
    anchor_value: float     # Starting point (default or user-set)
    anchor_kappa: float     # Decaying anchor weight (starts at 4.0)
    posterior: NIGPosterior # Observation-derived posterior

class VEThresholdState:
    current_value: float     # Current threshold (auto-updated from anchored mean)
    anchor_value: float      # Anchor point (default or user-set)
    anchor_kappa: float      # Decaying anchor weight (starts at 4.0)
    posterior: NIGPosterior  # Observation-derived posterior

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
| `/api/calibration/update` | POST | Update calibration from run result (auto-updates thresholds) |
| `/api/calibration/reset` | POST | Reset to defaults |
| `/api/calibration/approve-ve` | POST | DEPRECATED: Kept for backwards compatibility |
| `/api/calibration/set-ve-threshold` | POST | Manual threshold override (resets anchor) |
| `/api/calibration/set-advanced-params` | POST | Manual override for drift/sigma params (resets anchors) |
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
   - Check majority classification across intervals
   - If majority matches domain expectations: update drift/sigma posteriors
   - If unexpected outcome: update VE threshold posterior (auto-updates current_value)
   - Both anchor and observation posteriors decay with λ=0.95

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
  "sigma_pct_moderate": 7.0,
  "sigma_pct_heavy": 4.0,
  "sigma_pct_severe": 4.0,
  "expected_drift_moderate": 0.3,
  "expected_drift_heavy": 1.0,
  "expected_drift_severe": 2.0,
  "max_drift_moderate": 1.0,
  "max_drift_heavy": 3.0,
  "max_drift_severe": 5.0,
  "enabled": true,
  "last_updated": "2026-01-03T12:00:00Z"
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
| 2026-01-02 | Added MADSD (MAD of Successive Differences) for sigma calculation from actual data |
| 2026-01-02 | Updated sigma defaults: Moderate=7%, Heavy/Severe=4% (domain-specific) |
| 2026-01-02 | Implemented cross-domain max_drift calibration (Severe→Heavy) |
| 2026-01-03 | Removed split_ratio from all calibration and classification logic |
| 2026-01-03 | Added max_drift_pct for Moderate domain (1.0%/min default) |
| 2026-01-03 | Updated expected_drift_pct for Moderate domain to 0.3%/min |
| 2026-01-03 | Expanded sidebar sync: sigma, expected_drift, max_drift (all domains) |
| 2026-01-03 | Both Moderate and Heavy now use expected_drift and max_drift thresholds |
| 2026-01-03 | **Majority-based calibration**: Each run counts as ONE sample, not per-interval |
| 2026-01-03 | Multi-interval runs require >50% majority classification to count for calibration |
| 2026-01-03 | Averaged values from majority intervals used for calibration updates |
| 2026-01-03 | **Decaying Anchor**: All parameters now use decaying anchor approach (not just VE thresholds) |
| 2026-01-03 | VE thresholds auto-update without user prompts - values auto-populate from cloud |
| 2026-01-03 | anchor_kappa decays with λ=0.95, allowing observations to dominate over time |
| 2026-01-03 | Added AnchoredNIGPosterior structure for drift/sigma parameters |
| 2026-01-03 | Manual override creates new anchor and resets observation posterior |
