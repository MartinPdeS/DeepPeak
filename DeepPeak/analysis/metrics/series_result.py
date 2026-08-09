"""Series-level aggregate result models."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .trace_record import TraceRecord


@dataclass(frozen=True)
class PeakCountSeriesResult:
    """Aggregated peak-count and particle-flow arrays over a dilution series."""

    dilution: np.ndarray
    concentration: np.ndarray
    standard_particle_count: np.ndarray
    standard_particle_flow: np.ndarray
    cnn_particle_count: np.ndarray
    cnn_particle_flow: np.ndarray
    water_record: Optional[TraceRecord]
    records: List[TraceRecord]

    def to_series_result(self):
        """Return this aggregate through the stable core series contract."""

        from ...core.types import SeriesResult

        return SeriesResult(
            records=list(self.records),
            metadata={
                "dilution": np.asarray(self.dilution).tolist(),
                "concentration": np.asarray(self.concentration).tolist(),
                "standard_particle_count": np.asarray(
                    self.standard_particle_count
                ).tolist(),
                "standard_particle_flow": np.asarray(
                    self.standard_particle_flow
                ).tolist(),
                "cnn_particle_count": np.asarray(self.cnn_particle_count).tolist(),
                "cnn_particle_flow": np.asarray(self.cnn_particle_flow).tolist(),
                "water_record": (
                    None if self.water_record is None else self.water_record.to_dict()
                ),
            },
        )
