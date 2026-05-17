"""Configuration models for analysis components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveNetAnalyzerConfig:
    """Configuration shared by a WaveNetTraceAnalyzer instance."""

    sequence_length: int
    signal_normalization: str = "zscore"
    prediction_sampling_rate_hz: float = 125_000_000.0
