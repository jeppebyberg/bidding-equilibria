"""Linear block-oriented full workflow.

This orchestrator is intentionally thin. Each block can also be run directly:

  1. block1_data_labels_pipeline.py
  2. block2_policy_training_pipeline.py
  3. block3_poa_pipeline.py
  4. block35_support_oos_pipeline.py
  5. block4_dro_poa_pipeline.py
  6. block45_oos_poa_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver_tmp import (  # noqa: E402
    block1_data_labels_pipeline,
    block2_policy_training_pipeline,
    block3_poa_pipeline,
    block35_support_oos_pipeline,
    block4_dro_poa_pipeline,
    block45_oos_poa_pipeline,
)
from driver_tmp.block0_system_setup import (  # noqa: E402
    build_config,
    pipeline_manifest,
    write_manifest,
)

def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()
    print("\nRunning linear block-oriented full pipeline")

    manifests = {
        "block0_system_setup": pipeline_manifest(cfg),
        "block1_data_labels": block1_data_labels_pipeline.run(cfg),
        "block2_policy_training": block2_policy_training_pipeline.run(cfg),
        "block3_poa": block3_poa_pipeline.run(cfg),
        "block35_support_oos": block35_support_oos_pipeline.run(cfg),
        "block4_dro_poa": block4_dro_poa_pipeline.run(cfg),
        "block45_oos_poa": block45_oos_poa_pipeline.run(cfg),
    }
    write_manifest("full_pipeline", manifests, cfg)
    print("\nLinear block-oriented full pipeline complete.")
    return manifests


if __name__ == "__main__":
    run()

