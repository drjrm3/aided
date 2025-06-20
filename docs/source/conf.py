# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import filecmp
import shutil
from pathlib import Path
from typing import List

DOCS_SRC = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SRC.parents[1]
APPS_ROOT = REPO_ROOT / "aided" / "apps"
APPS_DOCS = DOCS_SRC / "apps"
APPS_DOCS.mkdir(parents=True, exist_ok=True)

# LaTex
FRAG_SRC_DIR  = REPO_ROOT / "docs" / "latex" / "fragments"
PDF_SRC = REPO_ROOT / "docs" / "latex" / "aided.pdf"
FRAG_DEST_DIR = DOCS_SRC / "latex"

def _copy_latex_fragments(app):
    FRAG_DEST_DIR.mkdir(exist_ok=True)
    for src in FRAG_SRC_DIR.glob("*.tex"):
        dest = FRAG_DEST_DIR / src.name
        if (not dest.exists()
            or src.stat().st_mtime > dest.stat().st_mtime):
            shutil.copy2(src, dest)

def _copy_latex_pdf(app):
    FRAG_DEST_DIR.mkdir(exist_ok=True)
    dest = FRAG_DEST_DIR / PDF_SRC.name
    if (not dest.exists()
        or PDF_SRC.stat().st_mtime > dest.stat().st_mtime):
        shutil.copy2(PDF_SRC, dest)


def _copy_app_readmes(app):
    """
    Copy each aided/apps/*/README.md to docs/source/apps/<app>.md
    but *only* if the destination is missing or older/different,
    so sphinx-autobuild doesn't fall into a loop.
    """
    for readme in APPS_ROOT.glob("*/README.md"):
        dest = APPS_DOCS / f"{readme.parent.name}.md"

        # copy once, or when the source changed
        if (
            not dest.exists()
            or readme.stat().st_mtime > dest.stat().st_mtime
            or not filecmp.cmp(readme, dest, shallow=False)
        ):
            shutil.copy2(readme, dest)  # keeps mtime


def _fix_google_multireturns(app, what, name, obj, options, lines: List[str]):
    """
    Convert a Google-style multi-item Returns block

        Returns:
            name1: desc1
            name2: desc2

    into the tuple + bullet form understood by napoleon:

        Returns:
            tuple:
                * name1: desc1
                * name2: desc2
    """
    out: list[str] = []
    in_returns = False
    pending: list[str] = []

    def flush_pending():
        """Emit pending lines as bullet-prefixed items."""
        for ln in pending:
            stripped = ln.lstrip()
            if not stripped:
                out.append(ln)
                continue
            out.append(" " * 12 + "* " + stripped)
        pending.clear()

    for ln in lines:
        if ln.strip().startswith("Returns:"):
            in_returns = True
            out.append(ln)  # keep header
            continue

        # we are inside Returns but encounter a blank – end the block
        if in_returns and not ln.strip():
            if pending:
                out.append(" " * 8 + "tuple:")  # insert before first bullet
                flush_pending()
            in_returns = False
            out.append(ln)
            continue

        # still inside the block → collect for later rewriting
        if in_returns:
            pending.append(ln)
            continue

        out.append(ln)

    # EOF while still in Returns
    if in_returns and pending:
        out.append(" " * 8 + "tuple:")
        flush_pending()

    # replace the original list in-place
    lines[:] = out


def setup(app):
    app.connect("builder-inited", _copy_app_readmes)  # your existing hook
    app.connect("builder-inited", _copy_latex_fragments)
    app.connect("builder-inited", _copy_latex_pdf)
    app.connect("autodoc-process-docstring", _fix_google_multireturns, priority=100)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aided"
copyright = "Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved."
author = "J. Robert Michael, PhD"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Enables Markdown (.md) parsing
    "sphinx.ext.mathjax",  # For rendering math equations
    "sphinx.ext.autodoc",  # For auto-generating API docs
    "sphinx.ext.napoleon",  # For Google-style/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add [source] links
    "sphinx.ext.todo",  # Support for todo directives
    "sphinx.ext.githubpages",  # For linking to GitHub pages
]
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = [
    "_static",
    "../../contrib/imgs",  # Path to custom images
]
html_extra_path = [str(Path(__file__).resolve().parents[2] / "contrib/imgs")]
