"""
Enums for VT Threshold Analyzer
"""

from enum import Enum


class RunType(str, Enum):
    """
    Intensity domain for training run.
    - MODERATE: Below VT1 threshold
    - HEAVY: Between VT1 and VT2
    - SEVERE: Above VT2 threshold
    All domains can have intervals.
    """
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"
    SEVERE = "SEVERE"

    @classmethod
    def from_string(cls, value: str) -> "RunType":
        """Convert string to RunType enum. Handles legacy values for backwards compatibility."""
        value_upper = value.upper().strip()
        # MODERATE domain (legacy: VT1, VT1_STEADY)
        if value_upper in ("MODERATE", "VT1", "VT1_STEADY", "VT1 (STEADY STATE)"):
            return cls.MODERATE
        # HEAVY domain (legacy: VT2, VT2_INTERVAL)
        elif value_upper in ("HEAVY", "VT2", "VT2_INTERVAL", "VT2 (INTERVALS)"):
            return cls.HEAVY
        # SEVERE domain
        elif value_upper in ("SEVERE", "VT2+", "VT2_PLUS"):
            return cls.SEVERE
        raise ValueError(f"Unknown run type: {value}")


class IntervalStatus(str, Enum):
    """Classification status for an interval."""
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    BORDERLINE = "BORDERLINE"
    ABOVE_THRESHOLD = "ABOVE_THRESHOLD"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return {
            self.BELOW_THRESHOLD: "Below Threshold",
            self.BORDERLINE: "Borderline",
            self.ABOVE_THRESHOLD: "Above Threshold",
        }[self]

    @property
    def color(self) -> str:
        """Color code for UI display."""
        return {
            self.BELOW_THRESHOLD: "green",
            self.BORDERLINE: "yellow",
            self.ABOVE_THRESHOLD: "red",
        }[self]
