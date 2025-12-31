# Project Rules

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

## End of Session
- Always remind user to merge the branch to main via GitHub PR before ending the chat
