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
| 00:00-04:00 | 1-3 | Context: the coincidence bottleneck in time-domain cytometry |
| 04:00-10:00 | 4-6 | Method: FLASH and realistic DeepPeak training |
| 10:00-17:00 | 7-11 | CYTO evidence: system, CNN localization, and three-fold validation |
| 17:00-20:00 | 12-14 | Quality controls, EV translation, and takeaways |

The experimental result slides are adapted from the CYTO 2026 presentation in the related DeepPeak project. They show an internal experimental case study and should be presented with the stated validation limits.

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
```

Run `python showcase/webinar/build_presentation.py` from the repository root to rebuild the primary polished deck on macOS with Keynote installed. The generator requires `python-pptx`.

Run `osascript showcase/webinar/build_presentation.applescript` to generate `presentation_native_placeholder.key`, a native Keynote template with named built-in `White` theme layouts, a shared title treatment, orange underline, presenter footer, and editable blank figure placeholders.

## Working Title

High-Throughput Extracellular Vesicle Detection with DeepPeak: Resolving Overlapping Pulses Using Neural Networks
