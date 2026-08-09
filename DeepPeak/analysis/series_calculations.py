"""Pure calculations used by dilution-series analysis.

Keeping these calculations outside the workflow class makes them reusable for
serialized ``PeakCountSeriesResult`` objects and keeps file orchestration
separate from numerical fitting.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ..core.exceptions import AnalysisInputError
from .metrics import resolve_series_or_result


def estimate_expected_particle_flow(
    series_or_result: Any,
    index: int,
    base_index: int = 0,
    reference_indices: Optional[Sequence[int]] = None,
    use_water_baseline: bool = True,
    calibration_slope: float = 1.0,
    calibration_intercept: float = 0.0,
) -> float:
    """Estimate target particle flow from reference trace records.

    The fit uses ``throughput / dilution + background`` when multiple
    references are supplied. A water record supplies the background directly
    when requested.
    """

    result = resolve_series_or_result(series_or_result)
    if len(result.records) == 0:
        raise AnalysisInputError(
            "No trace records are available. Call run() first and make sure files were found."
        )

    base_record = result.records[base_index]
    current_record = result.records[index]
    current_dilution = float(current_record.dilution)
    if current_dilution == 0.0:
        raise AnalysisInputError(
            "Expected particle flow is undefined when the target dilution is zero."
        )

    water_record = getattr(result, "water_record", None)
    background_flow = (
        float(water_record.standard_particle_flow)
        if use_water_baseline and water_record is not None
        else 0.0
    )
    indices = (
        [base_index]
        if reference_indices is None
        else [int(value) for value in reference_indices]
    )
    if reference_indices is not None and len(indices) > 1:
        indices = [
            reference_index for reference_index in indices if reference_index != index
        ]
    if len(indices) == 0:
        raise AnalysisInputError(
            "reference_indices must contain at least one usable reference index. "
            "When multiple indices are provided, the target index is excluded from the fit."
        )

    if len(indices) == 1:
        expected_raw = (base_record.standard_particle_flow - background_flow) * float(
            base_record.dilution
        ) / current_dilution + background_flow
        return float(calibration_slope * expected_raw + calibration_intercept)

    reference_dilutions = []
    reference_flows = []
    for reference_index in indices:
        reference_record = result.records[reference_index]
        dilution = float(reference_record.dilution)
        if dilution == 0.0:
            raise AnalysisInputError(
                "Expected particle flow is undefined when a reference dilution is zero."
            )
        reference_dilutions.append(dilution)
        reference_flows.append(float(reference_record.standard_particle_flow))

    reciprocal_dilution = 1.0 / np.asarray(reference_dilutions, dtype=float)
    observed_flow = np.asarray(reference_flows, dtype=float)
    if use_water_baseline and water_record is not None:
        throughput = np.linalg.lstsq(
            reciprocal_dilution.reshape(-1, 1),
            observed_flow - background_flow,
            rcond=None,
        )[0][0]
    else:
        design_matrix = np.column_stack(
            (reciprocal_dilution, np.ones(len(reference_dilutions), dtype=float))
        )
        throughput, background_flow = np.linalg.lstsq(
            design_matrix, observed_flow, rcond=None
        )[0]

    expected_raw = throughput / current_dilution + background_flow
    return float(calibration_slope * expected_raw + calibration_intercept)


__all__ = ["estimate_expected_particle_flow"]
