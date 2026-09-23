# High-Throughput Extracellular Vesicle Detection

Materials for a webinar on using DeepPeak to detect extracellular vesicles at high acquisition rates. The central use case is neural-network-assisted separation of overlapping pulses, increasing the fraction of events that can be measured reliably at useful throughput.

## Scope

- Explain how pulse overlap limits conventional extracellular-vesicle detection.
- Demonstrate realistic trace generation and neural-network training with DeepPeak.
- Compare neural unmixing with conventional peak detection.
- Report event-level recovery, count bias, timing accuracy, and retained throughput.
- Define validation and quality-control criteria for experimental use.

## 20-Minute Run Sheet

| Time | Slides | Segment |
| --- | --- | --- |
| 00:00-05:00 | 1-4 | DeepPeak package: capabilities and reusable Python workflow |
| 05:00-10:00 | 5-8 | Problem and method: coincidence, FLASH, and realistic training |
| 10:00-17:00 | 9-13 | CYTO evidence: system, CNN localization, and three-fold validation |
| 17:00-20:00 | 14-16 | Quality controls, EV translation, and takeaways |

Slides 3-4 introduce DeepPeak as an open-source Python package, including its public workflow and main capability areas. The experimental result slides are adapted from the CYTO 2026 presentation in the related DeepPeak project. They show an internal experimental case study and should be presented with the stated validation limits.

## Materials

```text
presentation.key                         Primary polished Keynote webinar deck
build_presentation.py                    Rebuilds the primary figure-based deck
build_presentation.applescript           Optional native Keynote placeholder template
amsterdam-umc-logo.svg                   Official Amsterdam UMC vector mark
amsterdam-umc-logo-presentation.png      Presentation-ready logo asset
pulse-overlap-motif.svg                  Editable source of the pulse visual
pulse-overlap-motif.png                  Presentation-ready pulse visual
cyto-*.png                               CYTO 2026 experimental evidence figures
build_overlap_logo.py                    Generates transparent overlap logo variants
deeppeak-overlap-logo-*.png              First-slide logo assets for dark and light fields
deeppeak-overlap-logo-*.svg              Vector originals of the first-slide logo assets
```

Run `python showcase/webinar/build_presentation.py` from the repository root to rebuild the primary polished deck on macOS with Keynote installed. The generator requires `python-pptx`.

Run `python showcase/webinar/build_overlap_logo.py` to regenerate the transparent first-slide logo. Use the `-dark` asset on the deck's dark teal field.

Run `osascript showcase/webinar/build_presentation.applescript` to generate `presentation_native_placeholder.key`, a native Keynote template with named built-in `White` theme layouts, a shared title treatment, orange underline, presenter footer, and editable blank figure placeholders.

## Working Title

High-Throughput Extracellular Vesicle Detection with DeepPeak: Resolving Overlapping Pulses Using Neural Networks
