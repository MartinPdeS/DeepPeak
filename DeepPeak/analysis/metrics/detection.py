"""Detection result models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PeakDetectionResult:
    """Detected peaks and the parameters used to obtain them."""

    peaks: np.ndarray
    properties: Dict[str, Any]
    peak_count: int
    detection_kwargs: Dict[str, Any]
    threshold: Optional[float] = None
    amplitudes: Optional[np.ndarray] = None

    @property
    def std_kwargs(self) -> Dict[str, Any]:
        """Return the stored detection kwargs for the standard detector."""

        return self.detection_kwargs

    @property
    def cnn_kwargs(self) -> Dict[str, Any]:
        """Return the stored detection kwargs for the CNN detector."""

        return self.detection_kwargs
