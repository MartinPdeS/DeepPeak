# Working in DeepPeak

DeepPeak is a Python package for generating, detecting, and analyzing peaks in
one-dimensional signals. It supports classical detection, optional neural
models, trace diagnostics, and dilution-series comparisons. Keep changes
focused on the requested behavior and grounded in the current implementation.

## Start here

- Read `README.rst`, `CONTRIBUTING.md`, and the relevant tests before changing an
  unfamiliar API. Build settings and dependencies live in `pyproject.toml`;
  `.github/workflows/` defines the CI checks.
- Inspect `git status --short` and the relevant diff. Preserve existing user
  edits, including notebooks, figures, and binary presentation files. Do not
  clean up unrelated files or overwrite a working artifact from Git history.
- Use an existing suitable virtual environment when available. Commands below
  use `python` to mean that environment's interpreter; avoid hard-coded paths
  to another developer's machine or environment.

## Repository map

| Path | Purpose |
| --- | --- |
| `DeepPeak/core/` | Typed configurations, result objects, protocols, and exceptions |
| `DeepPeak/generation/` | Signal generators, datasets, pulse kernels, noise, and peak counts |
| `DeepPeak/detection/`, `DeepPeak/processing.py` | Detection methods and signal processing |
| `DeepPeak/models/` | Optional ML models, training, evaluation, losses, and plotting |
| `DeepPeak/analysis/` | Trace diagnostics, dilution series, distributions, and comparisons |
| `DeepPeak/plotting/`, `DeepPeak/pipeline.py` | Reusable plots and workflow orchestration |
| `tests/` | Regression and integration checks |
| `docs/source/`, `docs/examples/` | Sphinx documentation and executable examples |
| `notebooks/` | Research and demonstration workflows |
| `showcase/webinar/` | Native Keynote deck, presenter notes, figures, and logo sources |
| `tools/`, `conda.recipe/` | Benchmark tooling and Conda packaging |

## Code and scientific behavior

- Support Python 3.10 and newer. Follow nearby conventions, use clear names and
  type annotations, and write NumPy-style docstrings for public callables.
- Preserve lazy imports in `DeepPeak/__init__.py` and subpackage APIs. A plain
  `import DeepPeak` must not import TensorFlow. Core generation and classical
  analysis must work without the `ml` extra.
- Prefer the existing typed configurations and result objects. Maintain public
  exports when adding APIs, and follow `docs/source/api_stability.rst` for
  deprecations and incompatible changes.
- Keep units, sampling rates, array shapes, normalization, and detector labels
  explicit. Distinguish event-location scores from recovered amplitudes and
  calibrated probabilities.
- Use fixed seeds for reproducible synthetic examples and regression tests.
  Validate scientific changes against known simulated events or appropriate
  experimental references; visual plausibility alone is insufficient.
- Return usable Matplotlib objects and support headless plotting. Avoid adding
  GUI windows or unconditional `plt.show()` calls to reusable library code.
- Add focused tests for behavioral changes, especially numerical edge cases,
  serialization, and public API compatibility. Avoid expensive training runs
  for changes that can be verified with small deterministic cases.

## Development and validation

Install only the extras needed for the task:

```sh
python -m pip install -e ".[testing,dev]"
# For model work:
python -m pip install -e ".[ml,testing,dev]"
# For documentation work:
python -m pip install -e ".[documentation]"
```

Run the relevant test files first, then broaden testing when the change affects
shared behavior. The core CI lane avoids these optional ML test files:

```sh
MPLBACKEND=Agg python -m pytest -q \
  --ignore=tests/test_classifiers.py \
  --ignore=tests/test_model_evaluation.py \
  --ignore=tests/test_model_plotting.py \
  --ignore=tests/test_training_config.py
```

With the ML extra installed, run the full suite for model or shared ML changes:

```sh
MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q
python -m ruff check DeepPeak tests
python -m ruff format --check DeepPeak tests
git diff --check
```

`pytest.ini` enables coverage reports by default. Keep generated reports and
caches out of commits. Report which checks ran and any dependency or environment
limitations; do not describe skipped ML checks as a full test-suite pass.

## Documentation, notebooks, and artifacts

- Update examples and documentation when public behavior changes. Check imports
  and signatures against the code rather than copying outdated examples.
- Build documentation with `MPLBACKEND=Agg python -m sphinx -b html docs/source
  /tmp/deeppeak-docs`. Sphinx-Gallery example execution is disabled by default;
  set `DEEPPEAK_DOCS_EXECUTE_EXAMPLES=1` only when an executable build is needed.
- Preserve notebook structure and meaningful results. Avoid unrelated output
  churn, checkpoint files, and large experimental inputs in commits.
- Keep editable figure/logo sources with their intended presentation assets.
  Regenerate the webinar overlap logos with
  `python showcase/webinar/build_overlap_logo.py` when changing that design.
- Do not commit virtual environments, build products, generated gallery pages,
  caches, or temporary previews. Preserve intentionally maintained result
  figures and benchmark artifacts when they are part of the requested work.

## Webinar and Keynote work

- `showcase/webinar/presentation.key` is the primary deck. Read its current
  contents before editing; the optional placeholder builder is not a substitute
  for the polished deck. Keep the run sheet in `showcase/webinar/README.md`
  consistent when adding, removing, or moving slides.
- Target the presentation by its explicit path or document name in Keynote
  automation. Never assume `document 1` is the webinar: multiple decks may be
  open. Save a temporary copy of the current file before binary edits.
- Preserve established layouts, section colors, typography, logos, and user
  changes. Verify slide numbering and transitions after structural edits.
- Presenter notes should follow the slide's visible cards, panels, or plot in
  order. Explain axes, curves, and colors where useful, then connect to the next
  slide. Keep introductions brief and natural to say aloud.
- Ground result claims in the evidence shown. Distinguish simulated ground
  truth from experimental distribution checks, study-specific throughput gains
  from general guarantees, and bead validation from EV validation.
- After saving, read back changed notes and export a temporary PDF to inspect
  affected slides. Check for clipping, retained image crops, overlaps, and
  incorrect labels. Do not restart Keynote without accounting for unsaved work
  in other open presentations.

## Releases

- Change versions and publish releases only within an explicit release task.
  `_version.py` is maintained by martinpdes-maintenance; use it for coordinated
  metadata updates.
- `make release-check` checks metadata consistency. In the current Makefile,
  `make release patch`, `make release minor`, and `make release major` create a
  release and push the commit and tag. `make tag VERSION=vX.Y.Z` creates a release
  commit and annotated tag. These are not routine validation commands.
- Inspect the Makefile and scripts before executing release commands; publishing
  workflows are triggered by `v*` tags.

## Shared Python style preferences

Apply these conventions to new and edited Python code. Leave unrelated existing code unchanged. Preserve each repository's existing public APIs and project-specific scientific or behavioral requirements.

- Use keyword-only parameters by default for new functions and methods, apart from implicit `self` and `cls`. Preserve existing public calling conventions unless an API change is part of the task.
- Use NumPy-style docstrings for classes, functions, methods, and properties. Document parameters, return values, units, array shapes, and limitations where they apply.
- Keep imports at module scope when practical. Use local imports only when needed to avoid circular dependencies or defer optional, expensive, or side-effectful imports.
- Keep all imports in one contiguous block, without blank lines between standard-library, third-party, and project imports.
- Avoid redundant self-alias imports such as `from package.module import Name as Name`; import the name directly unless the alias clarifies its meaning.
- Keep simple, single-purpose UI component calls on one line when practical, even when a text string is long. Use multiline formatting when several arguments or nested components make the call easier to read.
- Prefer descriptive variable and parameter names over abbreviations, including for local values and callback parameters.
- Avoid extracting a helper used only once unless it improves clarity or separates behavior that belongs in another layer.
- Keep callbacks focused on reading inputs, calling a service or workflow, and returning outputs. Move domain decisions and complicated data conversion into the appropriate service or workflow module.
- Write comments to explain non-obvious reasoning, scientific assumptions, or compatibility constraints; do not use them to narrate code that is already clear.
