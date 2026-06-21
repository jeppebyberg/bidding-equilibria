# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-21
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Centralized thesis figure-output policy.

Importing this module (anywhere in the project) makes every matplotlib figure
save as thesis-quality output:

  * a vector PDF (ideal for LaTeX / pdflatex inclusion), and
  * a high-DPI PNG (default 300) for quick previews.

It works by wrapping ``matplotlib.figure.Figure.savefig`` (which also backs
``matplotlib.pyplot.savefig``), so existing ``fig.savefig("foo.png", ...)`` calls
transparently produce both ``foo.pdf`` and ``foo.png`` at high resolution -- no
changes are needed at the individual call sites.

This module NEVER deletes files; it only writes/overwrites figure files. It does
not read, write, or delete any .json or .csv result artifacts.

Configuration via environment variables:
  THESIS_FIG_FORMATS   comma-separated list, default "pdf,png"
  THESIS_FIG_DPI       raster DPI (also used for rasterized parts of vector
                       backends), default "300"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.figure import Figure

# Output formats, in save order. PDF first so the vector copy is authoritative.
FORMATS = [
    f.strip().lower()
    for f in os.environ.get("THESIS_FIG_FORMATS", "pdf,png").split(",")
    if f.strip()
]
DPI = int(os.environ.get("THESIS_FIG_DPI", "300"))

# Extensions recognized as image targets; these are multiplexed across FORMATS.
_IMAGE_EXTS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".ps", ".tif", ".tiff"}


def _apply_rcparams() -> None:
    # Kept intentionally minimal so existing per-figure styling and layout are
    # preserved; only what is needed for crisp, embeddable output is set here.
    matplotlib.rcParams.update(
        {
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,  # embed TrueType, avoid Type-3 fonts
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _multi_savefig(orig: Any, fig: Figure, fname: Any, *args: Any, **kwargs: Any) -> Any:
    # Passthrough for file-like targets or an explicit format= (the caller is in
    # control of a single, specific save).
    if not isinstance(fname, (str, os.PathLike)) or "format" in kwargs:
        return orig(fig, fname, *args, **kwargs)

    path = Path(fname)
    base = path.with_suffix("") if path.suffix.lower() in _IMAGE_EXTS else path

    result = None
    for fmt in FORMATS:
        target = base.with_suffix("." + fmt)
        call_kwargs = dict(kwargs)
        call_kwargs["dpi"] = DPI
        result = orig(fig, str(target), *args, **call_kwargs)
    return result


def apply() -> None:
    """Apply rcParams and install the savefig wrapper (idempotent)."""
    _apply_rcparams()
    if getattr(Figure.savefig, "_thesis_wrapped", False):
        return
    _orig_savefig = Figure.savefig

    def savefig(self: Figure, fname: Any, *args: Any, **kwargs: Any) -> Any:
        return _multi_savefig(_orig_savefig, self, fname, *args, **kwargs)

    savefig._thesis_wrapped = True  # type: ignore[attr-defined]
    Figure.savefig = savefig  # type: ignore[assignment]


# Apply on import so a bare ``import results_viz._thesis_style`` is sufficient.
apply()
