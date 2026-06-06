"""Shared setup for the block-oriented thesis workflow.

Edit ``build_config`` when changing the case, horizon, seeds, eta grid, output
roots, or run toggles. All block scripts import this module, so PoA-only and
DRO-only runs stay aligned by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver.core.block0_core import (  # noqa: E402
    ProjectConfig,
    build_config,
    default_eta_grid,
    ensure_requested_policy_generators,
    pipeline_manifest,
    print_setup,
    write_manifest,
)

if __name__ == "__main__":
    print_setup()
