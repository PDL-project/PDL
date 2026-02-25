#!/usr/bin/env python3
"""
_lp_runner.py — Isolated subprocess runner for LP_Module.assign_subtasks_cp_sat.

Runs in a separate process to avoid ortools/protobuf version conflicts with
packages such as pyarrow that are loaded in the main planning_sar.py process.

Protocol:
  stdin  <- JSON payload (see keys below)
  stdout -> JSON result: {"subtask_id_str": robot_id_int, ...}
  exit 0 on success, exit 1 on failure (error message on stderr)

Input JSON keys:
  subtasks              : list[dict]  (each has "id")
  robot_ids             : list[int]
  robots_db             : list[dict]  (from sar_robots.robots)
  plan_actions_by_subtask : dict[str, list[str]]  (keys are str(subtask_id))
  objects_ai            : str
  binding_pairs         : list[[int, int]]  (optional)
  parallel_groups       : dict[str, list[int]]  (optional, keys are str(group_id))
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup — PDL scripts must be importable
# ---------------------------------------------------------------------------
_SAR_ROOT    = Path(__file__).resolve().parent.parent          # /PDL/SAR
_PDL_SCRIPTS = _SAR_ROOT.parent / "AI2Thor" / "baselines" / "PDL" / "scripts"

for _p in [str(_PDL_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())

        from LP_Module import assign_subtasks_cp_sat  # type: ignore

        subtasks               = payload["subtasks"]
        robot_ids              = [int(x) for x in payload["robot_ids"]]
        robots_db              = payload["robots_db"]
        plan_actions_by_subtask = {
            int(k): v for k, v in payload["plan_actions_by_subtask"].items()
        }
        objects_ai             = payload["objects_ai"]
        binding_pairs          = [tuple(p) for p in (payload.get("binding_pairs") or [])]
        parallel_groups        = payload.get("parallel_groups")  # dict[str, list] or None

        result: dict = assign_subtasks_cp_sat(
            subtasks=subtasks,
            robot_ids=robot_ids,
            robots_db=robots_db,
            plan_actions_by_subtask=plan_actions_by_subtask,
            objects_ai=objects_ai,
            binding_pairs=binding_pairs if binding_pairs else None,
            parallel_groups=parallel_groups,
        )

        # Output: {str(subtask_id): robot_id}
        print(json.dumps({str(k): int(v) for k, v in result.items()}))
        sys.exit(0)

    except Exception as exc:
        import traceback
        print(f"LP_ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
