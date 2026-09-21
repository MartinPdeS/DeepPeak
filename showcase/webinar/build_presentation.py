"""Build the Amsterdam UMC-branded DeepPeak webinar as a native Keynote file."""

from pathlib import Path
import subprocess

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).parent
PPTX_PATH = ROOT / "presentation.pptx"
KEYNOTE_PATH = ROOT / "presentation.key"
LOGO_PATH = ROOT / "amsterdam-umc-logo-presentation.png"
MOTIF_PATH = ROOT / "pulse-overlap-motif.png"
CNN_TRACE_PATH = ROOT / "cyto-cnn-trace.png"
ARRIVAL_TIME_PATH = ROOT / "cyto-arrival-time.png"
AMPLITUDES_PATH = ROOT / "cyto-amplitudes.png"
THROUGHPUT_PATH = ROOT / "cyto-throughput.png"
PRESENTER_NAME = "MartinPdeS"

TEAL = "006D77"
DARK_TEAL = "004854"
CORAL = "EF8354"
INK = "102A33"
MUTED = "56727A"
MINT = "F2F8F7"
PALE_TEAL = "DCEDEA"
PAPER = "F8FBFA"
SEAFOAM = "CFE5E1"
SUN = "F2B84B"
WHITE = "FFFFFF"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def rect(slide, x, y, width, height, color, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    shape.line.fill.background()
    return shape


def line(slide, x1, y1, x2, y2, color, width=1.2):
    shape = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    return shape


def image(slide, path, x, y, width, height=None):
    if height is None:
        return slide.shapes.add_picture(
            str(path), Inches(x), Inches(y), width=Inches(width)
        )
    return slide.shapes.add_picture(
        str(path), Inches(x), Inches(y), Inches(width), Inches(height)
    )


def text(
    slide, value, x, y, width, height, size, color=INK, bold=False, align=PP_ALIGN.LEFT
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = frame.paragraphs[0]
    paragraph.text = value
    paragraph.alignment = align
    paragraph.font.name = "Avenir Next"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    return box


def add_logo(slide, x=10.78, y=0.33, width=1.95):
    slide.shapes.add_picture(str(LOGO_PATH), Inches(x), Inches(y), width=Inches(width))


def add_footer(slide, number, dark=False):
    color = "BFD0D2" if dark else MUTED
    rule_color = "2A6670" if dark else SEAFOAM
    rect(slide, 0.82, 6.70, 11.70, 0.018, rule_color)
    text(slide, PRESENTER_NAME, 0.82, 6.94, 2.15, 0.18, 9, color, bold=True)
    text(slide, "Amsterdam UMC | DeepPeak", 3.05, 6.94, 3.0, 0.18, 9, color)
    text(
        slide, f"{number:02d}", 12.05, 6.94, 0.35, 0.18, 9, color, align=PP_ALIGN.RIGHT
    )


def add_content_background(slide):
    rect(slide, 0, 0, 13.333, 7.5, PAPER)
    rect(slide, 0, 0, 13.333, 0.12, DARK_TEAL)
    rect(slide, 12.94, 0.12, 0.39, 7.38, PALE_TEAL)
    rect(slide, 12.94, 0.12, 0.10, 1.22, CORAL)


def add_slide_heading(slide, eyebrow, title, number):
    add_content_background(slide)
    text(slide, eyebrow, 0.82, 0.47, 5.4, 0.24, 10, TEAL, bold=True)
    text(slide, title, 0.82, 0.86, 9.55, 0.7, 25, INK, bold=True)
    rect(slide, 0.82, 1.64, 0.92, 0.05, CORAL)
    rect(slide, 10.33, 0.32, 2.20, 0.57, WHITE, radius=True)
    add_logo(slide, 10.47, 0.40, 1.88)
    add_footer(slide, number)


def add_stage(slide, x, heading, body, accent):
    rect(slide, x, 3.16, 3.45, 2.16, WHITE, radius=True)
    rect(slide, x, 3.16, 3.45, 0.09, accent, radius=True)
    text(slide, heading, x + 0.3, 3.54, 2.78, 0.25, 11, accent, bold=True)
    text(slide, body, x + 0.3, 4.02, 2.75, 0.82, 19, INK, bold=True)


def add_figure_card(slide, path, x, y, width, height, label, accent=TEAL):
    rect(slide, x, y, width, height, WHITE, radius=True)
    rect(slide, x, y, 0.10, height, accent, radius=True)
    image(slide, path, x + 0.18, y + 0.18, width - 0.36, height - 0.58)
    text(
        slide,
        label,
        x + 0.26,
        y + height - 0.28,
        width - 0.45,
        0.16,
        9,
        accent,
        bold=True,
    )


def build_presentation() -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    blank = presentation.slide_layouts[6]

    # 1. Title
    slide = presentation.slides.add_slide(blank)
    rect(slide, 0, 0, 13.333, 7.5, PAPER)
    rect(slide, 0, 0, 0.17, 7.5, TEAL)
    rect(slide, 7.48, 0, 5.853, 7.5, DARK_TEAL)
    rect(slide, 0.84, 1.10, 0.13, 4.20, CORAL)
    rect(slide, 9.76, 0.56, 2.54, 0.67, WHITE, radius=True)
    add_logo(slide, 9.92, 0.67, 2.20)
    text(slide, "DEEPPeak WEBINAR", 1.22, 1.20, 4.8, 0.28, 11, TEAL, bold=True)
    text(
        slide,
        "High-Throughput\nExtracellular Vesicle Detection",
        1.22,
        1.75,
        5.95,
        1.34,
        31,
        INK,
        bold=True,
    )
    text(
        slide,
        "Resolving overlapping pulses using neural networks",
        1.22,
        3.42,
        5.55,
        0.34,
        16,
        CORAL,
        bold=True,
    )
    text(
        slide,
        "A measurement-focused workflow for increasing usable event throughput",
        1.22,
        4.16,
        5.20,
        0.58,
        15,
        MUTED,
    )
    rect(slide, 7.94, 2.12, 4.86, 2.95, WHITE, radius=True)
    slide.shapes.add_picture(
        str(MOTIF_PATH), Inches(8.13), Inches(2.39), width=Inches(4.46)
    )
    text(
        slide,
        "SIGNAL OVERLAP | EVENT RECOVERY",
        8.13,
        5.39,
        4.46,
        0.2,
        10,
        "89BDB9",
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    text(slide, PRESENTER_NAME, 1.22, 6.77, 2.20, 0.18, 10, MUTED, bold=True)
    text(slide, "Amsterdam UMC | DeepPeak", 3.54, 6.77, 2.70, 0.18, 10, MUTED)
    text(slide, "01", 12.05, 6.77, 0.35, 0.18, 10, "89BDB9", align=PP_ALIGN.RIGHT)

    # 2. Roadmap
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "20-MINUTE WEBINAR", "From coincidence loss to validated recovery", 2
    )
    roadmap = [
        (
            "01",
            "The bottleneck",
            "Why overlapping pulses erase usable information.",
            "00:00-04:00",
        ),
        (
            "02",
            "The method",
            "FLASH: CNN localization plus amplitude recovery.",
            "04:00-10:00",
        ),
        (
            "03",
            "The evidence",
            "Timing, amplitude, and throughput checks on data.",
            "10:00-17:00",
        ),
        (
            "04",
            "The translation",
            "Where DeepPeak supports EV workflows next.",
            "17:00-20:00",
        ),
    ]
    for index, (number, heading, body, timing) in enumerate(roadmap):
        y = 2.15 + (index * 0.95)
        rect(
            slide, 0.82, y, 0.52, 0.52, CORAL if index in (1, 2) else TEAL, radius=True
        )
        text(
            slide,
            number,
            0.82,
            y + 0.15,
            0.52,
            0.15,
            9,
            WHITE,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(slide, heading, 1.62, y + 0.02, 2.25, 0.24, 17, INK, bold=True)
        text(slide, body, 3.92, y + 0.04, 5.55, 0.28, 13, MUTED)
        text(
            slide,
            timing,
            10.08,
            y + 0.05,
            1.48,
            0.22,
            11,
            TEAL,
            bold=True,
            align=PP_ALIGN.RIGHT,
        )

    # 3. Bottleneck
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "THE MEASUREMENT PROBLEM",
        "When throughput rises, pulse overlap becomes the bottleneck",
        3,
    )
    text(
        slide,
        "Conventional peak detection assumes isolated, well-separated events.",
        0.78,
        2.16,
        9.7,
        0.34,
        17,
        MUTED,
    )
    add_stage(
        slide, 0.78, "LOW RATE", "Separated pulses\ncan be counted directly.", TEAL
    )
    add_stage(
        slide,
        4.94,
        "HIGHER RATE",
        "Partially overlapping pulses\nlose timing and count fidelity.",
        CORAL,
    )
    add_stage(
        slide,
        9.10,
        "SATURATION RISK",
        "Merged responses distort\nmeasured event populations.",
        DARK_TEAL,
    )
    line(slide, 4.3, 4.24, 4.72, 4.24, PALE_TEAL, 2.2)
    line(slide, 8.46, 4.24, 8.88, 4.24, PALE_TEAL, 2.2)

    # 4. FLASH pipeline
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "THE FLASH PIPELINE", "Localize events first, then recover amplitudes", 4
    )
    text(
        slide,
        "FLASH combines a CNN event-localization stage with an analytical amplitude solver.",
        0.78,
        2.17,
        10.2,
        0.5,
        18,
        MUTED,
    )
    rect(slide, 0.78, 3.25, 5.55, 2.05, PALE_TEAL, radius=True)
    rect(slide, 7.00, 3.25, 5.55, 2.05, "FBE8E1", radius=True)
    text(slide, "CNN LOCALIZATION", 1.14, 3.68, 2.8, 0.2, 11, TEAL, bold=True)
    text(
        slide,
        "Estimate arrival times\nfrom the noisy detector trace.",
        1.14,
        4.13,
        4.58,
        0.72,
        21,
        INK,
        bold=True,
    )
    text(slide, "ANALYTICAL RECOVERY", 7.36, 3.68, 3.2, 0.2, 11, CORAL, bold=True)
    text(
        slide,
        "Solve amplitudes from the\nknown response and event times.",
        7.36,
        4.13,
        4.6,
        0.72,
        21,
        INK,
        bold=True,
    )

    # 5. Workflow
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "THE MODELING WORKFLOW",
        "DeepPeak provides a realistic training workflow",
        5,
    )
    steps = [
        (
            "01",
            "Generate",
            "Pulse-shape variability, Poisson event counts, blanks, noise, and drift.",
        ),
        (
            "02",
            "Train",
            "WaveNet event-location learning with validation-aware controls.",
        ),
        (
            "03",
            "Evaluate",
            "Recovery, timing error, count bias, confidence, and retained throughput.",
        ),
    ]
    for index, (number, heading, body) in enumerate(steps):
        y = 2.26 + (index * 1.2)
        rect(slide, 0.80, y, 0.58, 0.58, CORAL if index == 1 else TEAL, radius=True)
        text(
            slide,
            number,
            0.80,
            y + 0.16,
            0.58,
            0.18,
            11,
            WHITE,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(slide, heading, 1.68, y + 0.02, 1.8, 0.28, 19, INK, bold=True)
        text(slide, body, 1.68, y + 0.42, 8.9, 0.3, 14, MUTED)
        if index < 2:
            line(slide, 1.09, y + 0.62, 1.09, y + 1.18, PALE_TEAL, 2.2)
    text(
        slide,
        "SignalGenerator + Gaussian + PoissonCount + WaveNet",
        1.68,
        6.10,
        7.5,
        0.24,
        13,
        TEAL,
        bold=True,
    )

    # 6. Training realism
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "REALISM MATTERS",
        "Design the training distribution around the acquisition",
        6,
    )
    columns = [
        (
            "PULSE LIBRARY",
            "Widths, amplitudes, positions, and asymmetric shapes reflect expected EV signals.",
        ),
        (
            "BACKGROUND",
            "Vary noise level, baseline level, and baseline drift across trace batches.",
        ),
        (
            "NEGATIVES",
            "Include blank traces so the detector learns when not to call an event.",
        ),
    ]
    for index, (heading, body) in enumerate(columns):
        x = 0.78 + (index * 4.16)
        rect(slide, x, 2.54, 3.48, 2.65, MINT, radius=True)
        rect(slide, x + 0.32, 2.90, 0.46, 0.08, TEAL if index != 1 else CORAL)
        text(
            slide,
            heading,
            x + 0.32,
            3.37,
            2.7,
            0.25,
            11,
            TEAL if index != 1 else CORAL,
            bold=True,
        )
        text(slide, body, x + 0.32, 3.92, 2.72, 0.94, 18, INK, bold=True)

    # 7. Experimental system
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "EXPERIMENTAL SYSTEM", "Test the method where coincidence is real", 7
    )
    text(
        slide,
        "The CYTO study used high-flow light-scattering measurements from 300 nm polystyrene beads.",
        0.82,
        2.10,
        10.8,
        0.34,
        16,
        MUTED,
    )
    system_cards = [
        (
            "ACQUISITION",
            "BD FACSCanto II flow cell and laser optics\nwith electronics bypassed.",
        ),
        (
            "DIGITIZATION",
            "Side-scatter PMT signal recorded with\na 125 MHz, 14-bit PicoScope.",
        ),
        (
            "REALITY CHECK",
            "A distorted response with a secondary lobe\ncreates a demanding overlap problem.",
        ),
    ]
    for index, (heading, body) in enumerate(system_cards):
        x = 0.82 + (index * 4.05)
        rect(slide, x, 3.12, 3.36, 2.25, WHITE, radius=True)
        rect(slide, x, 3.12, 3.36, 0.10, CORAL if index == 2 else TEAL, radius=True)
        text(
            slide,
            heading,
            x + 0.30,
            3.55,
            2.55,
            0.20,
            11,
            CORAL if index == 2 else TEAL,
            bold=True,
        )
        text(slide, body, x + 0.30, 4.02, 2.58, 0.80, 17, INK, bold=True)

    # 8. CNN evidence
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "CYTO EXPERIMENT", "CNN localization resolves dense event sequences", 8
    )
    text(
        slide,
        "At approximately 94k events/s, the model maps the detector trace to sharply localized event support.",
        0.82,
        2.08,
        10.8,
        0.34,
        16,
        MUTED,
    )
    add_figure_card(
        slide,
        CNN_TRACE_PATH,
        0.82,
        2.72,
        7.25,
        3.34,
        "CYTO 2026 | noisy detector trace and CNN event prediction",
        CORAL,
    )
    rect(slide, 8.48, 2.72, 3.90, 3.34, DARK_TEAL, radius=True)
    text(slide, "WHY THIS MATTERS", 8.84, 3.16, 2.9, 0.20, 11, "74D1D0", bold=True)
    text(
        slide,
        "The neural model supplies event timing.\n\nThat timing makes a deterministic amplitude recovery step possible.",
        8.84,
        3.66,
        2.95,
        1.30,
        19,
        WHITE,
        bold=True,
    )

    # 9. Validation
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "THREE-FOLD VALIDATION",
        "Check timing, amplitude, and throughput independently",
        9,
    )
    text(
        slide,
        "A high-throughput result is useful only when recovered measurements agree with a credible reference.",
        0.78,
        2.18,
        10.8,
        0.4,
        16,
        MUTED,
    )
    metrics = [
        ("RECOVERY", "Precision\nRecall"),
        ("COUNTING", "Count bias\nDistribution shift"),
        ("LOCALIZATION", "Timing error\nSeparation limit"),
        ("USABILITY", "Retained\nthroughput"),
    ]
    for index, (heading, body) in enumerate(metrics):
        x = 0.87 + (index * 3.14)
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x), Inches(3.32), Inches(2.12), Inches(2.12)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = rgb(PALE_TEAL if index % 2 == 0 else "FBE8E1")
        circle.line.color.rgb = rgb(TEAL if index % 2 == 0 else CORAL)
        circle.line.width = Pt(1.8)
        text(
            slide,
            heading,
            x + 0.30,
            3.83,
            1.52,
            0.2,
            10,
            TEAL if index % 2 == 0 else CORAL,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(
            slide,
            body,
            x + 0.22,
            4.28,
            1.68,
            0.55,
            16,
            INK,
            bold=True,
            align=PP_ALIGN.CENTER,
        )

    # 10. Timing and amplitude evidence
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "CYTO RESULTS", "Preserve physical timing and reduce amplitude bias", 10
    )
    add_figure_card(
        slide,
        ARRIVAL_TIME_PATH,
        0.82,
        2.22,
        5.45,
        3.94,
        "Arrival-time statistics | low-rate reference",
        TEAL,
    )
    add_figure_card(
        slide,
        AMPLITUDES_PATH,
        6.65,
        2.22,
        5.45,
        3.94,
        "Amplitude distribution | low-rate reference",
        CORAL,
    )
    text(
        slide,
        "At high event rate, the CYTO study found that FLASH preserved plausible arrival-time behavior while reducing coincidence-induced amplitude bias.",
        0.82,
        6.28,
        10.9,
        0.24,
        13,
        MUTED,
        bold=True,
    )

    # 11. Throughput evidence
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide, "CYTO RESULTS", "Extend the usable event-rate range by more than 10x", 11
    )
    add_figure_card(
        slide,
        THROUGHPUT_PATH,
        0.82,
        2.10,
        8.15,
        4.22,
        "Measured particle flow versus expected throughput",
        CORAL,
    )
    rect(slide, 9.43, 2.10, 2.90, 4.22, DARK_TEAL, radius=True)
    text(
        slide,
        ">10x",
        9.78,
        2.83,
        2.2,
        0.55,
        35,
        "74D1D0",
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    text(
        slide,
        "usable event-rate\nrange in the CYTO\nexperiment",
        9.78,
        3.72,
        2.2,
        0.82,
        17,
        WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    text(
        slide,
        "Interpret this as a validated operating-range extension, not unlimited deconvolution.",
        9.78,
        5.22,
        2.2,
        0.48,
        11,
        "BFD0D2",
        align=PP_ALIGN.CENTER,
    )

    # 12. Quality control
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "QUALITY CONTROL",
        "Operational controls keep neural estimates scientifically useful",
        12,
    )
    controls = [
        "Calibrate confidence thresholds against known mixtures or lower-throughput references.",
        "Track pulse-shape, noise, and baseline distributions for acquisition-to-training drift.",
        "Preserve flagged traces for review instead of forcing a count in ambiguous regions.",
        "Report the excluded region and the retained-throughput gain together.",
    ]
    for index, control in enumerate(controls):
        y = 2.22 + (index * 0.94)
        rect(slide, 0.80, y, 0.38, 0.38, TEAL if index % 2 == 0 else CORAL, radius=True)
        text(
            slide,
            f"0{index + 1}",
            0.80,
            y + 0.10,
            0.38,
            0.14,
            8,
            WHITE,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(slide, control, 1.46, y + 0.02, 9.98, 0.34, 16, INK, bold=True)

    # 13. EV translation
    slide = presentation.slides.add_slide(blank)
    add_slide_heading(
        slide,
        "TRANSLATING TO EXTRACELLULAR VESICLES",
        "Use DeepPeak to move the throughput boundary with evidence",
        13,
    )
    text(
        slide,
        "The CYTO result motivates an EV workflow, but every acquisition must establish its own validated operating range.",
        0.82,
        2.10,
        10.9,
        0.35,
        16,
        MUTED,
    )
    translation = [
        (
            "1",
            "Characterize",
            "Acquire low-rate EV traces and extract representative instrument responses.",
        ),
        (
            "2",
            "Train",
            "Generate realistic overlap, noise, drift, and blank examples with DeepPeak.",
        ),
        (
            "3",
            "Validate",
            "Compare timing, amplitude distributions, and counts against a lower-rate reference.",
        ),
    ]
    for index, (number, heading, body) in enumerate(translation):
        x = 0.82 + (index * 4.05)
        rect(slide, x, 3.20, 3.36, 2.38, MINT, radius=True)
        rect(
            slide,
            x + 0.30,
            3.54,
            0.48,
            0.48,
            CORAL if index == 1 else TEAL,
            radius=True,
        )
        text(
            slide,
            number,
            x + 0.30,
            3.68,
            0.48,
            0.15,
            10,
            WHITE,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(slide, heading, x + 1.04, 3.57, 1.98, 0.24, 18, INK, bold=True)
        text(slide, body, x + 0.30, 4.34, 2.72, 0.75, 14, MUTED)

    # 14. Close
    slide = presentation.slides.add_slide(blank)
    rect(slide, 0, 0, 13.333, 7.5, DARK_TEAL)
    rect(slide, 0, 0, 0.17, 7.5, CORAL)
    rect(slide, 9.73, 0.56, 2.54, 0.67, WHITE, radius=True)
    add_logo(slide, 9.89, 0.67, 2.20)
    text(slide, "TAKEAWAY", 0.90, 1.12, 4.2, 0.26, 11, "74D1D0", bold=True)
    text(
        slide,
        "Increase usable throughput\nby recovering resolvable overlap.",
        0.90,
        1.70,
        8.2,
        1.26,
        31,
        WHITE,
        bold=True,
    )
    rect(slide, 0.90, 3.48, 1.15, 0.05, SUN)
    text(
        slide,
        "DeepPeak supports a controlled workflow: realistic generation, neural event localization, and validation against a reference.",
        0.90,
        3.96,
        8.55,
        0.75,
        18,
        "D2E7E5",
    )
    text(
        slide,
        "Next: add acquisition-specific traces, reference data, and validated metrics.",
        0.90,
        5.45,
        8.7,
        0.32,
        15,
        "74D1D0",
        bold=True,
    )
    add_footer(slide, 14, dark=True)

    presentation.save(PPTX_PATH)


def convert_to_keynote() -> None:
    if KEYNOTE_PATH.exists():
        KEYNOTE_PATH.unlink()
    subprocess.run(["open", "-a", "Keynote"], check=True)
    script = """
on run argv
    set sourceFile to POSIX file (item 1 of argv)
    set outputFile to POSIX file (item 2 of argv)
    tell application "Keynote"
        activate
        set importedDocument to open sourceFile
        save importedDocument in outputFile
        close importedDocument saving no
    end tell
end run
"""
    subprocess.run(
        ["osascript", "-e", script, str(PPTX_PATH), str(KEYNOTE_PATH)], check=True
    )
    PPTX_PATH.unlink()


if __name__ == "__main__":
    build_presentation()
    convert_to_keynote()
