"""Create transparent DeepPeak webinar logos that visualize pulse overlap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
TEAL = "#5CE1D0"
CORAL = "#FF9A70"
LIGHT_INK = "#F7FBFA"
DARK_INK = "#10333A"


def pulse(time, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)


def build_logo(wordmark_color: str, output_stem: str) -> None:
    time = np.linspace(0, 8.2, 900)
    first = pulse(time, 1.45, 0.38, 1.0)
    second = pulse(time, 2.60, 0.42, 0.9)
    third = pulse(time, 3.65, 0.36, 1.04)

    figure, axis = plt.subplots(figsize=(11.5, 3.4), dpi=240)
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")

    for waveform, color, alpha in (
        (first, TEAL, 1.0),
        (second, CORAL, 0.92),
        (third, TEAL, 1.0),
    ):
        axis.plot(
            time,
            waveform,
            color=color,
            linewidth=9,
            solid_capstyle="round",
            alpha=alpha,
        )

    detector_signal = first + second + third
    axis.plot(time, detector_signal, color=wordmark_color, linewidth=1.4, alpha=0.35)

    event_times = np.array([1.45, 2.60, 3.65])
    event_heights = np.array([1.0, 0.9, 1.04])
    axis.scatter(
        event_times,
        event_heights,
        s=190,
        color=wordmark_color,
        edgecolor=TEAL,
        linewidth=2.6,
        zorder=4,
    )
    axis.plot([0.08, 4.45], [0, 0], color=wordmark_color, linewidth=1.6, alpha=0.5)

    axis.text(
        4.90,
        0.76,
        "DeepPeak",
        color=wordmark_color,
        fontsize=36,
        fontweight="bold",
        fontfamily="Avenir Next",
        va="center",
    )
    axis.text(
        4.93,
        0.35,
        "EVENT RESOLUTION",
        color=CORAL,
        fontsize=11,
        fontweight="bold",
        fontfamily="Avenir Next",
        va="center",
    )

    axis.set_xlim(-0.1, 8.2)
    axis.set_ylim(-0.12, 1.28)
    axis.axis("off")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for extension in ("png", "svg"):
        figure.savefig(
            ROOT / f"{output_stem}.{extension}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


if __name__ == "__main__":
    build_logo(LIGHT_INK, "deeppeak-overlap-logo-dark")
    build_logo(DARK_INK, "deeppeak-overlap-logo-light")
