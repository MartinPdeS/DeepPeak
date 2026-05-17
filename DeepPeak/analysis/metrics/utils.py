"""Helpers for working with analysis result bundles."""

from typing import Any


def resolve_series_or_result(series_or_result: Any) -> Any:
    """Return a result object from either a series instance or a result bundle."""

    if hasattr(series_or_result, "records"):
        return series_or_result

    if hasattr(series_or_result, "get_last_result"):
        return series_or_result.get_last_result()

    raise TypeError(
        "series_or_result must be either a result object with `.records` "
        "or a PeakCountSeries-like object with `.get_last_result()`."
    )
