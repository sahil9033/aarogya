"""Vital sign input validation.

Ranges are deliberately physiological rather than clinical-correct: the goal is
to reject impossible input, not to second-guess a real reading. Out-of-range
*physiological* values are the red-flag engine's job, not the validator's.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VitalsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(30, ge=0, le=130, description="Age in years.")
    heart_rate: float | None = Field(None, ge=20, le=300, description="Resting heart rate, bpm.")
    systolic_bp: float | None = Field(None, ge=40, le=300, description="Systolic blood pressure, mmHg.")
    spo2: float | None = Field(None, ge=40, le=100, description="Peripheral oxygen saturation, %.")
    temperature_c: float | None = Field(None, ge=25, le=45, description="Body temperature, Celsius.")
    glucose_mgdl: float | None = Field(None, ge=10, le=1500, description="Blood glucose, mg/dL.")

    def to_engine_vitals(self) -> dict[str, float]:
        """Map API field names onto the engine's short keys, filling defaults.

        The engine treats a missing reading as 'not measured' by substituting a
        normal value, which is why absent readings must not suppress genuine
        red flags from the values that *were* supplied.
        """
        return {
            "age": self.age,
            "hr": self.heart_rate if self.heart_rate is not None else 72,
            "bp": self.systolic_bp if self.systolic_bp is not None else 120,
            "spo2": self.spo2 if self.spo2 is not None else 98,
            "temp": self.temperature_c if self.temperature_c is not None else 37,
            "glucose": self.glucose_mgdl if self.glucose_mgdl is not None else 90,
        }

    def measured_keys(self) -> set[str]:
        """Which readings the caller actually supplied."""
        mapping = {
            "hr": self.heart_rate,
            "bp": self.systolic_bp,
            "spo2": self.spo2,
            "temp": self.temperature_c,
            "glucose": self.glucose_mgdl,
        }
        return {key for key, value in mapping.items() if value is not None}
