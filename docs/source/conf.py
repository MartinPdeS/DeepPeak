"""DeepPeak documentation configuration."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from martinpdes_docs import configure

project = "DeepPeak"
package = "DeepPeak"
repository = "MartinPdeS/DeepPeak"
author = "Martin Poinsinet de Sivry-Houle"
docs_source = Path(__file__).resolve().parent


globals().update(
    configure(
        project=project,
        package=package,
        author=author,
        repository=repository,
        logo="../images/logo.png",
        logo_text="",
        favicon="_static/favicon.png",
        prefer_installed_package=True,
        gallery_default_thumb=str(docs_source / "_static" / "icon.png"),
        gallery_options={
            "plot_gallery": os.getenv("DEEPPEAK_DOCS_EXECUTE_EXAMPLES") == "1",
        },
        html_css_files=["branding.css"],
    )
)
