"""DeepPeak documentation configuration."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from MPSPlots.styles import use_mpsplots_style

from martinpdes_docs import configure

project = "DeepPeak"
package = "DeepPeak"
repository = "MartinPdeS/DeepPeak"
author = "Martin Poinsinet de Sivry-Houle"
docs_source = Path(__file__).resolve().parent


def reset_mpl(gallery_conf, fname):
    """Apply the project plotting style before each gallery example."""
    use_mpsplots_style()


globals().update(
    configure(
        project=project,
        package=package,
        author=author,
        repository=repository,
        logo="_static/logo.svg",
        favicon="_static/thumbnail.png",
        prefer_installed_package=True,
        gallery_default_thumb=str(docs_source / "_static" / "thumbnail.png"),
        gallery_options={
            "plot_gallery": os.getenv("DEEPPEAK_DOCS_EXECUTE_EXAMPLES") == "1",
            "reset_modules": reset_mpl,
        },
    )
)

autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "numpydoc",
    "scipy",
    "tensorflow",
    "sklearn",
]

html_css_files += ["layout.css"]
html_context.update({"doc_path": "docs/source", "default_mode": "dark"})
