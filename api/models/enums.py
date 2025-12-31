"""
Enums for VT Threshold Analyzer
"""

from enum import Enum


class RunType(str, Enum):
    """Type of training run."""
    VT1_STEADY = "VT1"
    VT2_INTERVAL = "VT2"

    @classmethod
    def from_string(cls, value: str) -> "RunType":
        """Convert string to RunType enum."""
        value_upper = value.upper().strip()
        if value_upper in ("VT1", "VT1_STEADY", "VT1 (STEADY STATE)"):
            return cls.VT1_STEADY
        elif value_upper in ("VT2", "VT2_INTERVAL", "VT2 (INTERVALS)"):
            return cls.VT2_INTERVAL
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
