"""
Step 7 (SARExecutor) 단독 실행 스크립트
    Step 1~6 결과물(assignment.json, PDDL plans)이 이미 저장된 상태에서
    Step 7만 바로 실행

Usage:
    cd /home/nuc/Desktop/PDL/SAR
    python PDL_SAR/executor_Module.py --scene 1 --agents 3
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
_SAR_ROOT    = Path(__file__).resolve().parent.parent   # /PDL/SAR
_PDLSAR_ROOT = Path(__file__).resolve().parent          # /PDL/SAR/PDL_SAR

for _p in [str(_SAR_ROOT), str(_PDLSAR_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    parser = argparse.ArgumentParser(description="SAR Step 7 단독 실행")
    parser.add_argument("--scene",  type=int, default=1)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--task-idx", type=int, default=0)
    args = parser.parse_args()

    # --- 환경 초기화 ---
    from env import SAREnv
    print(f"\n[Step 7] SAREnv 초기화 (scene={args.scene}, agents={args.agents}, seed={args.seed})")
    sar_env = SAREnv(num_agents=args.agents, scene=args.scene, seed=args.seed)
    sar_env.reset()
    print(f"  Task: {sar_env.task}")

    # --- 오브젝트 이름 수집 ---
    all_object_names = list(sar_env.controller.all_names)
    print(f"  Objects: {all_object_names}")

    # --- SARExecutor 실행 ---
    from sar_executor import SARExecutor, _fires_still_active

    sar_pdl_root = str(_PDLSAR_ROOT)
    executor = SARExecutor(sar_pdl_root)
    executor.run(task_idx=args.task_idx, task_name="task", task_description=sar_env.task,
                 num_agents=sar_env.num_agents)
    executor.set_object_names(all_object_names)

    # Show assignment and parallel groups (conflict resolution already applied inside run())
    print("\n[Step 7] Task Assignment:")
    for sid, rid in sorted(executor.assignment.items()):
        print(f"  Subtask {sid} -> Robot {rid}")
    print(f"\n[Step 7] Parallel Groups: {executor.parallel_groups}")

    print("\n[Step 7] Executing plans in SAR environment...")
    results = executor.execute_in_sar(sar_env)

    # --- 결과 출력 ---
    print("\n" + "=" * 60)
    print("[PDL-SAR] Execution Summary")
    print("=" * 60)
    n_ok   = sum(1 for r in results.values() if r.success)
    n_fail = len(results) - n_ok
    print(f"  Succeeded: {n_ok} / {len(results)}")
    print(f"  Failed:    {n_fail} / {len(results)}")
    for sid, r in sorted(results.items()):
        status = "SUCCESS" if r.success else f"FAIL ({r.error_message})"
        print(f"    Subtask {sid}: {status}")

    checker = sar_env.checker
    if checker is not None:
        try:
            coverage       = checker.get_coverage()
            transport_rate = checker.get_transport_rate()
            finished       = checker.check_success()
            # Final state check: verify ALL fires (including spread regions) are out
            # still_active = _fires_still_active(sar_env)
            # if still_active:
            #     finished = False
            #     for fname in still_active:
            #         print(f"  [Final State] Fire '{fname}' still active — Fail")
            balance        = executor._compute_balance_metric(finished=checker.check_success())
            exec_rate      = executor._compute_exec_rate()
            print(
                f"\n  Coverage:{coverage:.3f}, Transport Rate:{transport_rate:.3f}, "
                f"Finished:{finished}, Balance:{balance:.3f}, Exec:{exec_rate:.3f}"
            )
        except Exception as _e:
            print(f"  [Metrics] Error computing metrics: {_e}")
    print("=" * 60)


if __name__ == "__main__":
    main()