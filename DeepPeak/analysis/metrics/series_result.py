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
