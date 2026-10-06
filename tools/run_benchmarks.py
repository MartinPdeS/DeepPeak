#!/usr/bin/env python3
"""Regenerate the documented DeepPeak scientific benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from DeepPeak.benchmarking import (
    classical_detector,
    evaluate_detector,
    make_benchmark_dataset,
)


def neural_detector(model_path: Path, threshold: float = 0.5):
    """Load a Keras model and turn its sequence output into peak indices."""

    from tensorflow import keras

    model = keras.models.load_model(model_path)

    def detect(signal):
        prediction = np.asarray(
            model.predict(signal[None, :, None], verbose=0)
        ).squeeze()
        peaks, _ = find_peaks(prediction, height=threshold, distance=7)
        return peaks

    return detect


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in {"DenseNet", "WaveNet", "U-Net"}:
        raise argparse.ArgumentTypeError(
            "use DenseNet=PATH, WaveNet=PATH, or U-Net=PATH"
        )
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/source/benchmarks"))
    parser.add_argument("--model", action="append", type=parse_model, default=[])
    parser.add_argument("--traces-per-level", type=int, default=24)
    args = parser.parse_args()

    cases = make_benchmark_dataset(traces_per_level=args.traces_per_level)
    detectors = {"Classical": classical_detector()}
    detectors.update({name: neural_detector(path) for name, path in args.model})
    results = [
        result
        for name, detector in detectors.items()
        for result in evaluate_detector(name, detector, cases)
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].to_dict())
    with (args.output / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {
        "Classical": "#006D77",
        "DenseNet": "#004854",
        "WaveNet": "#EF8354",
        "U-Net": "#B56B12",
    }
    for method in detectors:
        selected = [result for result in results if result.method == method]
        x = np.arange(len(selected))
        axes[0].plot(
            x,
            [item.precision for item in selected],
            "o-",
            label=method,
            color=colors[method],
        )
        axes[1].plot(
            x,
            [item.recall for item in selected],
            "o-",
            label=method,
            color=colors[method],
        )
    labels = [
        f"{item.scenario}\n{item.level:g}"
        for item in results
        if item.method == next(iter(detectors))
    ]
    for axis, title in zip(axes, ("Precision", "Recall")):
        axis.set(
            title=title,
            ylim=(0, 1.05),
            xticks=np.arange(len(labels)),
            xticklabels=labels,
        )
        axis.tick_params(axis="x", rotation=60)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(args.output / "benchmark-summary.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
