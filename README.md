# VT Threshold Analyzer

A Streamlit desktop application for analyzing Tymewear VitalPro respiratory data to assess whether runs were performed within VT1 or VT2 threshold zones.

## Features

- **CSV Upload**: Parse VitalPro breath-by-breath data
- **Automatic Interval Detection**: Detects work/recovery intervals from power data
- **Manual Override**: Specify interval structure manually
- **Signal Filtering**: Rolling median + Savitzky-Golay smoothing
- **CUSUM Analysis**: Statistical process control for drift detection
- **Interactive Visualization**: Plotly charts with interval highlighting
- **Adjustable Parameters**: Fine-tune blanking period, thresholds, and expected drift rates

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then:
1. Upload your VitalPro CSV file using the sidebar
2. Configure run format (auto-detect or manual)
3. Adjust analysis parameters if needed
4. Click "Analyze Run"
5. Review results in the visualization and table

## Analysis Algorithm

### Run Format Detection
- Analyzes power data to detect work/recovery intervals
- Uses hysteresis-based state machine for robust detection
- Falls back to manual specification if auto-detection fails

### Signal Processing
1. **Rolling Median Filter** (N=5): Removes outliers/spikes
2. **Savitzky-Golay Filter** (window=15, poly=2): Smooths while preserving inflection points

### CUSUM Threshold Analysis
1. **Kinetic Blanking** (90s): Ignores on-transient response
2. **Baseline Calibration** (90-150s): Establishes expected VE level
3. **Domain-Expected Drift**: Uses literature-based drift rates as baseline slope
   - VT1 (Moderate): 0.25 L/min per minute
   - VT2 (Heavy): 2.0 L/min per minute
4. **CUSUM Detection**: Accumulates deviations above expected drift
5. **Adaptive Threshold**:
   - VT1: H = 5σ (more sensitive)
   - VT2: H = 7σ (more tolerant)

### Status Classification
- **PASS**: CUSUM never exceeded threshold
- **RECOVERED**: Alarm triggered but CUSUM returned below H/2
- **FAIL**: Sustained drift above threshold

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Blanking Period | 90s | Time to ignore at interval start |
| Calibration End | 150s | End of baseline calibration window |
| H Multiplier (VT1) | 5.0 | Threshold sensitivity for moderate domain |
| H Multiplier (VT2) | 7.0 | Threshold sensitivity for heavy domain |
| Expected Drift (VT1) | 0.25 | Baseline drift rate L/min per min |
| Expected Drift (VT2) | 2.0 | Baseline drift rate L/min per min |

## Data Format

The app expects VitalPro CSV files with:
- Column Y: "Breath by breath time" (timestamp in seconds)
- Column AE: "VE breath by breath" (minute ventilation in L/min)
- Column F: "Power" (for interval detection)

## License

For personal use with Tymewear VitalPro data.
