"""DeepPeak documentation configuration."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from MPSPlots.styles import use_mpsplots_style
from sphinx_gallery.sorting import FileNameSortKey

import DeepPeak
from martinpdes_docs import configure

project = "DeepPeak"
package = "DeepPeak"
repository = "MartinPdeS/DeepPeak"
author = "Martin Poinsinet de Sivry-Houle"
version = os.getenv("tag", "latest")


def reset_mpl(gallery_conf, fname):
    """Apply the project plotting style before each gallery example."""
    use_mpsplots_style()


globals().update(
    configure(
        project=project,
        package=package,
        author=author,
        repository=repository,
        version=version,
        logo="_static/logo.svg",
        favicon="_static/thumbnail.png",
        sphinx_gallery_conf={
            "examples_dirs": "../examples",
            "gallery_dirs": "gallery",
            "image_scrapers": ("matplotlib",),
            "filename_pattern": r".*\.py",
            "plot_gallery": os.getenv("DEEPPEAK_DOCS_EXECUTE_EXAMPLES") == "1",
            "thumbnail_size": [600, 600],
            "download_all_examples": False,
            "reset_modules": reset_mpl,
            "capture_repr": ("_repr_html_", "__repr__"),
            "nested_sections": True,
            "within_subsection_order": FileNameSortKey,
        },
    )
)

# Sphinx-Gallery resolves ``default_thumb_file`` from the process directory.
# Use an absolute source-tree path so this remains valid in CI as well as when
# the documentation is built from the repository root or ``docs/source``.
sphinx_gallery_conf["default_thumb_file"] = str(
    Path(__file__).parent / "_static" / "thumbnail.png"
)

autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "numpydoc",
    "scipy",
    "tensorflow",
    "sklearn",
]

extensions += ["sphinx.ext.mathjax", "sphinx.ext.autosectionlabel"]
html_css_files += ["layout.css"]
html_context.update({"doc_path": "docs/source", "default_mode": "dark"})
