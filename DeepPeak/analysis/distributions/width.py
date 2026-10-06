"""Peak-width distribution diagnostics."""

from typing import Dict, Literal, Optional

import numpy as np
import scipy.stats as stats

from .. import metrics


DetectorLabel = Literal["standard", "cnn"]
XAxisLabel = Literal["sample", "time"]


def _resolve_single_label_metrics(
    metrics_by_label: Dict[str, metrics.PeakWidthDistribution],
    label: Optional[str],
) -> metrics.PeakWidthDistribution:
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


def _extract_peak_widths(
    detection: object, dx: float, x_axis: XAxisLabel
) -> np.ndarray:
    properties = getattr(detection, "properties", {})
    widths = None
    if isinstance(properties, dict):
        widths = properties.get("widths_pixels")

    if widths is None:
        start_indices = np.asarray(
            properties.get("start_indices", np.asarray([])), dtype=float
        )
        end_indices = np.asarray(
            properties.get("end_indices", np.asarray([])), dtype=float
        )
        if (
            start_indices.size
            and end_indices.size
            and start_indices.size == end_indices.size
        ):
            widths = end_indices - start_indices + 1.0
        else:
            widths = np.asarray([])

    widths = np.asarray(widths, dtype=float).ravel()
    widths = widths[np.isfinite(widths) & (widths > 0.0)]

    if x_axis == "time":
        widths = widths * float(dx)
    elif x_axis != "sample":
        raise ValueError('x_axis must be either "sample" or "time".')

    return widths


def compute_peak_width_distribution_metrics(
    series_or_result,
    index: int,
    detector: DetectorLabel = "standard",
    x_axis: XAxisLabel = "sample",
) -> Dict[str, metrics.PeakWidthDistribution]:
    """Compute peak-width diagnostics for one trace and one or more detectors."""

    resolved_result = metrics.resolve_series_or_result(series_or_result)
    record = resolved_result.records[index]

    metrics_by_label: Dict[str, metrics.PeakWidthDistribution] = {}
    width_unit = "Time" if x_axis == "time" else "Samples"

    for label in _detector_labels(detector):
        detection = record.standard if label == "standard" else record.cnn
        widths = _extract_peak_widths(detection=detection, dx=record.dx, x_axis=x_axis)
        if widths.size == 0:
            raise ValueError(f"{label}: at least one detected peak width is required.")

        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))
        minimum_width = float(np.min(widths))
        maximum_width = float(np.max(widths))
        standard_deviation_width = (
            float(np.std(widths, ddof=1)) if widths.size > 1 else np.nan
        )
        coefficient_of_variation_width = (
            float(standard_deviation_width / mean_width)
            if np.isfinite(standard_deviation_width) and mean_width != 0.0
            else np.nan
        )
        skewness_width = (
            float(stats.skew(widths, bias=False)) if widths.size > 2 else np.nan
        )
        kurtosis_width = (
            float(stats.kurtosis(widths, fisher=True, bias=False))
            if widths.size > 3
            else np.nan
        )

        fitted_shape, fitted_loc, fitted_scale = stats.lognorm.fit(widths, floc=0.0)
        fitted_shape = float(fitted_shape)
        fitted_loc = float(fitted_loc)
        fitted_scale = float(fitted_scale)

        if widths.size >= 5 and fitted_shape > 0.0 and fitted_scale > 0.0:
            ks_lognormal_statistic, ks_lognormal_p_value = stats.kstest(
                widths,
                "lognorm",
                args=(fitted_shape, fitted_loc, fitted_scale),
            )
            ks_lognormal_statistic = float(ks_lognormal_statistic)
            ks_lognormal_p_value = float(ks_lognormal_p_value)
        else:
            ks_lognormal_statistic = np.nan
            ks_lognormal_p_value = np.nan

        metrics_by_label[label] = metrics.PeakWidthDistribution(
            label=label,
            x_axis=x_axis,
            width_unit_label=width_unit,
            number_of_peaks=int(widths.size),
            mean_width=mean_width,
            median_width=median_width,
            minimum_width=minimum_width,
            maximum_width=maximum_width,
            standard_deviation_width=standard_deviation_width,
            coefficient_of_variation_width=coefficient_of_variation_width,
            skewness_width=skewness_width,
            kurtosis_width=kurtosis_width,
            fitted_lognormal_shape=fitted_shape,
            fitted_lognormal_loc=fitted_loc,
            fitted_lognormal_scale=fitted_scale,
            ks_lognormal_statistic=ks_lognormal_statistic,
            ks_lognormal_p_value=ks_lognormal_p_value,
            widths=widths,
        )

    return metrics_by_label


__all__ = [
    "compute_peak_width_distribution_metrics",
]
