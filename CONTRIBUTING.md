# Contributing to DeepPeak

Use Python 3.10 or newer. Install development dependencies with `python -m pip install -e ".[testing,documentation,dev]"`, then run `python -m pytest` and `python -m ruff check DeepPeak tests`.

Releases use semantic version tags. Run `make patch`, `make minor`, or `make major` from a clean `master` checkout; the command creates the release commit, tags it, and pushes both. Publishing workflows run only for `v*` tags.

The Conda recipe is at `conda.recipe/meta.yaml`. Do not commit generated documentation, virtual environments, build artefacts, or caches.

## Shared release versioning

Versions are explicit in `[project].version`; builds never derive versions from
Git. Install the pinned release tool in the interpreter used by Make:

```sh
python -m pip install "martinpdes-maintenance==0.2.2"
make release-check PYTHON=python
```

The `release` extra declares the same tool. The scripts in `tools/` delegate to
it, and `[tool.mps-maintenance]` declares the package's release files. Use
`python tools/release_tag.py vX.Y.Z --dry-run` to preview changes from a clean,
full checkout. The tool aligns configured metadata, commits, and creates an
annotated tag; it never pushes. Existing Makefile publishing shortcuts retain
their push behavior. Fetch remote tags first. With no release tags, `next` uses
the explicit project version as its starting point.
