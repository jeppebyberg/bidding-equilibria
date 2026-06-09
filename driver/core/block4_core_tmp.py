"""Re-export block4_core with DRO_PoA_optimization_tmp substituted for all model construction.

Both block4_core and tightening_main look up DRO_PoAOptimization in their own
module __dict__ at call time, so patching those dicts here is sufficient to
redirect every construction site (build_dro_optimizer, build_dro_tightening,
run_dro_eta_sweep, run_dro_tightening_for_regime, run_dro_essential_bounds_for_regime).
"""
from __future__ import annotations

import driver.core.block4_core as _bc4
import models.DRO_PoA.DRO_PoA_tightening.tightening_main as _tm
from models.DRO_PoA.DRO_PoA_optimization_tmp import DRO_PoAOptimization

_bc4.DRO_PoAOptimization = DRO_PoAOptimization  # type: ignore[attr-defined]
_tm.DRO_PoAOptimization = DRO_PoAOptimization  # type: ignore[attr-defined]

from driver.core.block4_core import *  # noqa: E402, F401, F403 — re-export all symbols
