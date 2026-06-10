"""Block 4 DRO PoA pipeline using DRO_PoA_optimization_tmp.

Importing this module triggers the model-class patch in block4_core_tmp before
any pipeline function is called.  Everything else is identical to
block4_dro_poa_pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import driver.core.block4_core_tmp  # noqa: F401 — patches DRO_PoAOptimization into block4_core
from driver.block4_dro_poa_pipeline import run  # noqa: F401

__all__ = ["run"]
