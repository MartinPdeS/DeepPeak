"""Compatibility import for the canonical detection configuration."""

from ...core.config import TraceConfig


class WaveNetAnalyzerConfig(TraceConfig):
    """Legacy-named view of :class:`DeepPeak.core.TraceConfig`."""

    def __init__(
        self,
        sequence_length: int,
        signal_normalization: str = "zscore",
        prediction_sampling_rate_hz: float = 125_000_000.0,
    ) -> None:
        super().__init__(
            sequence_length=sequence_length,
            normalization=signal_normalization,
            sampling_rate_hz=prediction_sampling_rate_hz,
        )

    @property
    def signal_normalization(self) -> str:
        return self.normalization

    @property
    def prediction_sampling_rate_hz(self) -> float:
        return float(self.sampling_rate_hz)
