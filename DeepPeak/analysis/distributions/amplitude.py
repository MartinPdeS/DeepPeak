"""Peak-amplitude distribution diagnostics.

These helpers mirror the event-arrival analysis workflow for peak amplitudes.
They operate on the canonical trace records produced by the analysis pipeline,
compute summary statistics for detected peak amplitudes on the processed trace,
and provide small, explicit plotting helpers for notebooks.
"""

from typing import Dict, Literal, Optional

import numpy as np
import scipy.stats as stats

from .. import metrics


DetectorLabel = Literal["standard", "cnn"]


def _resolve_single_label_metrics(
    metrics_by_label: Dict[str, metrics.PeakAmplitudeDistribution],
    label: Optional[str],
) -> metrics.PeakAmplitudeDistribution:
    if label is None:
        if len(metrics_by_label) != 1:
            raise ValueError(
                "label must be provided when multiple detector entries are present."
            )
        return next(iter(metrics_by_label.values()))

    try:
        return metrics_by_label[label]
    except KeyError as error:
        available_labels = ", ".join(sorted(metrics_by_label))
        raise ValueError(
            f"Unknown label {label!r}. Available labels: {available_labels}."
        ) from error


def _detector_labels(detector: DetectorLabel) -> tuple[str, ...]:
    if detector == "standard":
        return ("standard",)
    if detector == "cnn":
        return ("cnn",)
    raise ValueError('detector must be either "standard" or "cnn".')


def _extract_peak_amplitudes(
    values: np.ndarray,
    detection: object,
    *,
    use_recovered_amplitudes: bool,
) -> np.ndarray:
    if use_recovered_amplitudes:
        recovered_amplitudes = getattr(detection, "amplitudes", None)
        if recovered_amplitudes is not None:
            recovered_amplitudes = np.asarray(recovered_amplitudes, dtype=float).ravel()
            recovered_amplitudes = recovered_amplitudes[
                np.isfinite(recovered_amplitudes)
            ]
            if recovered_amplitudes.size > 0:
                return recovered_amplitudes

        raise ValueError(
            "cnn: recovered amplitudes are unavailable. "
            "Provide cnn_amplitude_sigma_samples when analyzing the trace so CNN amplitude plots "
            "use overlap-corrected amplitudes instead of raw signal samples."
        )

    peak_indices = np.asarray(getattr(detection, "peaks", np.asarray([])), dtype=int)
    peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < values.size)]

    amplitudes = np.asarray(values, dtype=float).ravel()[peak_indices]
    return amplitudes[np.isfinite(amplitudes)]


def compute_peak_amplitude_distribution_metrics(
    series_or_result,
    index: int,
    detector: DetectorLabel = "standard",
) -> Dict[str, metrics.PeakAmplitudeDistribution]:
    """Compute peak-amplitude diagnostics for one trace and one or more detectors.

    Parameters
    ----------
    series_or_result:
        A series instance or previously computed series result.
    index:
        Index of the trace record inside the sorted series result.
    detector:
        Which detector output to analyze: standard or CNN. CNN metrics require
        recovered amplitudes and therefore require analysis with
        ``cnn_amplitude_sigma_samples``.

    Returns
    -------
    dict
        Mapping from detector label to amplitude diagnostics.
    """

    resolved_result = metrics.resolve_series_or_result(series_or_result)
    record = resolved_result.records[index]

    metrics_by_label: Dict[str, metrics.PeakAmplitudeDistribution] = {}

    for label in _detector_labels(detector):
        values = np.asarray(record.signal, dtype=float).ravel()
        if label == "standard":
            detection = record.standard
        else:
            detection = record.cnn

        amplitudes = _extract_peak_amplitudes(
            values=values,
            detection=detection,
            use_recovered_amplitudes=(label == "cnn"),
        )
        if amplitudes.size == 0:
            raise ValueError(
                f"{label}: at least one detected peak amplitude is required."
            )

        mean_amplitude = float(np.mean(amplitudes))
        median_amplitude = float(np.median(amplitudes))
        minimum_amplitude = float(np.min(amplitudes))
        maximum_amplitude = float(np.max(amplitudes))
        standard_deviation_amplitude = (
            float(np.std(amplitudes, ddof=1)) if amplitudes.size > 1 else np.nan
        )
        coefficient_of_variation_amplitude = (
            float(standard_deviation_amplitude / mean_amplitude)
            if np.isfinite(standard_deviation_amplitude) and mean_amplitude != 0.0
            else np.nan
        )
        skewness_amplitude = (
            float(stats.skew(amplitudes, bias=False)) if amplitudes.size > 2 else np.nan
        )
        kurtosis_amplitude = (
            float(stats.kurtosis(amplitudes, fisher=True, bias=False))
            if amplitudes.size > 3
            else np.nan
        )

        fitted_normal_mean, fitted_normal_standard_deviation = stats.norm.fit(
            amplitudes
        )
        fitted_normal_mean = float(fitted_normal_mean)
        fitted_normal_standard_deviation = float(fitted_normal_standard_deviation)

        if amplitudes.size >= 5 and fitted_normal_standard_deviation > 0.0:
            ks_normal_statistic, ks_normal_p_value = stats.kstest(
                amplitudes,
                "norm",
                args=(fitted_normal_mean, fitted_normal_standard_deviation),
            )
            ks_normal_statistic = float(ks_normal_statistic)
            ks_normal_p_value = float(ks_normal_p_value)
        else:
            ks_normal_statistic = np.nan
            ks_normal_p_value = np.nan

        metrics_by_label[label] = metrics.PeakAmplitudeDistribution(
            label=label,
            number_of_peaks=int(amplitudes.size),
            mean_amplitude=mean_amplitude,
            median_amplitude=median_amplitude,
            minimum_amplitude=minimum_amplitude,
            maximum_amplitude=maximum_amplitude,
            standard_deviation_amplitude=standard_deviation_amplitude,
            coefficient_of_variation_amplitude=coefficient_of_variation_amplitude,
            skewness_amplitude=skewness_amplitude,
            kurtosis_amplitude=kurtosis_amplitude,
            fitted_normal_mean=fitted_normal_mean,
            fitted_normal_standard_deviation=fitted_normal_standard_deviation,
            ks_normal_statistic=ks_normal_statistic,
            ks_normal_p_value=ks_normal_p_value,
            amplitudes=amplitudes,
        )

    return metrics_by_label


__all__ = [
    "compute_peak_amplitude_distribution_metrics",
]
