# Project Rules

## Active Development Branch
**Branch:** `claude/modern-web-interface-7vbdH`
**Status:** In Progress - React frontend replacing Streamlit

Always checkout this branch at the start of a new conversation:
```bash
git checkout claude/modern-web-interface-7vbdH
git pull origin claude/modern-web-interface-7vbdH
```

## Feature Status (React Frontend vs Streamlit)

### Implemented in React:
- [x] Basic dashboard layout with sidebar and chart
- [x] CSV file upload (local only)
- [x] VE binned line with gradient
- [x] Breath scatter dots
- [x] Interval shading (colored by status)
- [x] Single slope line
- [x] CUSUM line (fixed color)
- [x] Results display in sidebar

### Missing in React (exists in Streamlit app.py):
- [ ] **2-Hinge Model Display:** Segment 2 + Segment 3 lines (purple)
- [ ] **Slope Annotations:** %/min labels on each segment
- [ ] **Grey Recovery Shading:** Between intervals
- [ ] **Yellow Ramp-up Shading:** Before Phase III onset
- [ ] **Chart Labels:** "Rest", "Int X", "Ramp" with durations
- [ ] **Below-Chart Metrics:** Clickable "Avg VE / Drift% / Split Slope" for each interval
- [ ] **Header Metrics:** "VT2 Intervals 4x10" format, cumulative drift display
- [ ] **Zoomed-in View:** Interval details with Reset button
- [ ] **Cloud Integration:** List/fetch sessions from Firebase
- [ ] **CUSUM Color Change:** Green→Red when in alarm

### Backend API Status:
- [x] CSV parsing (`/api/files/parse`)
- [x] Interval detection (`/api/files/detect-intervals`)
- [x] CUSUM analysis (`/api/analysis/run`)
- [x] Firebase cloud storage (`/api/sessions`, `/api/upload`)

## General Behavior
1. When user provides any instructions, carefully evaluate whether the instructions are logical and appropriate given the overall purpose of the app. If the instructions are not clear, ask clarifying questions.
2. Do not code anything automatically. Always ask for permission first before coding.
3. When user asks to change a specific part of the code, do not alter any other parts of the code without first confirming with user he wants those other parts to be altered.

## Interval Detection
- Single interval = VT1, Multiple intervals = VT2
- Always use grid-fitting to detect intervals that extend to end of recording
- All recoveries and intervals are the same duration across a single run (respectively)

## UI Preferences
- Zoom buttons must always be within the applicable row, never below the table
- Keep interface simple during testing phase
- Use st.columns for table layouts with CSS gap removal
