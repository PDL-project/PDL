"""
SARExecutor: MultiRobotExecutor equivalent for the SAR environment.

Pipeline role (mirrors MultiRobotExecutor in AI2Thor/baselines/PDL/scripts/):
  1. run()              - Loads PDDL plan actions + assignment from disk.
  2. execute_in_sar()   - Executes plans in SAREnv, returns SubTaskExecutionResult per subtask.
  3. execute_in_sar_with_feedback() - Same but invokes on_subtask_failed callback for replanning.

PDDL action -> SAR action mapping:
  (gotoobject   r obj)        ->  NavigateTo(<obj>)
  (getsupply    r src s)      ->  GetSupply(<src>, <supply_type>)
  (usesupply    r reg s)      ->  UseSupply(<parent_fire>, <supply_type>)
  (explore      r)            ->  Explore()
  (carry        r person)     ->  Carry(<person>)
  (dropoff      r person dep) ->  DropOff(<deposit>, <person>)
  (storesupply  r dep s)      ->  StoreSupply(<deposit>)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------
# Make sure SAR root is importable
# -----------------------------------------------------------------------
_SAR_ROOT = Path(__file__).resolve().parent.parent   # SAR/
if str(_SAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAR_ROOT))


@dataclass
class SubTaskExecutionResult:
    """Per-subtask execution result (mirrors the same class in MultiRobotExecutor)."""
    subtask_id: int
    success: bool
    error_message: Optional[str] = None
    completed_actions: Optional[List[str]] = field(default_factory=list)


# -----------------------------------------------------------------------
# PDDL -> SAR action mapping
# -----------------------------------------------------------------------

def _pddl_to_sar(pddl_line: str, object_name_map: Dict[str, str]) -> str:
    """
    Convert one FastDownward plan line to a SAREnv action string.

    pddl_line examples (all lowercase from FastDownward):
        (gotoobject robot1 reservoirutah)
        (getsupply robot1 reservoirutah sand)
        (usesupply robot1 caldorfire_region_1 sand)
        (explore robot1)
        (carry robot1 lostpersontimmy)
        (dropoff robot1 lostpersontimmy depositfacility)
        (storesupply robot1 depositfacility sand)

    object_name_map: lowercase_name -> original_case_name
    """
    s = pddl_line.strip().strip("()")
    if not s or s.startswith(";"):
        return ""

    parts = s.split()
    action_lower = parts[0].lower()

    def canon(token: str) -> str:
        """Recover original-case name from lowercase PDDL token."""
        return object_name_map.get(token.lower(), token)

    # args[0] is robot id, actual args start at args[1]
    args = parts[1:]

    if action_lower == "gotoobject":
        return f"NavigateTo({canon(args[1])})"

    elif action_lower == "getsupply":
        src = canon(args[1])
        supply = canon(args[2]).capitalize()
        return f"GetSupply({src}, {supply})"

    elif action_lower == "usesupply":
        target = canon(args[1])
        supply = canon(args[2]).capitalize()
        fire_name = _region_to_fire(target, object_name_map)
        return f"UseSupply({fire_name}, {supply})"

    elif action_lower == "explore":
        return "Explore()"

    elif action_lower == "carry":
        return f"Carry({canon(args[1])})"

    elif action_lower == "dropoff":
        # PDDL: DropOff robot person deposit
        # SAR:  DropOff(deposit, person)
        person = canon(args[1])
        deposit = canon(args[2])
        return f"DropOff({deposit}, {person})"

    elif action_lower == "storesupply":
        deposit = canon(args[1])
        return f"StoreSupply({deposit})"

    elif action_lower in ("noop", "idle", "done"):
        return "NoOp()"

    else:
        print(f"[SARExecutor] Unknown PDDL action '{action_lower}', using NoOp")
        return "NoOp()"


def _region_to_fire(name: str, object_name_map: Dict[str, str]) -> str:
    """
    Given a fire or region name, return the parent fire name.
    e.g. "CaldorFire_Region_1" -> "CaldorFire"
         "CaldorFire"          -> "CaldorFire"
    """
    # If "_Region_" in name, strip the suffix
    m = re.match(r"^(.+?)_Region_\d+$", name, re.IGNORECASE)
    if m:
        fire_root = m.group(1)
        # Try to find it in the name map (case-insensitive)
        canon = object_name_map.get(fire_root.lower(), fire_root)
        return canon
    return name


# -----------------------------------------------------------------------
# SARExecutor
# -----------------------------------------------------------------------

class SARExecutor:
    """
    Executes PDL PDDL plans inside SAREnv.

    Usage
    -----
    executor = SARExecutor(base_path)
    executor.run(task_idx=0, task_name="task", task_description=task)
    results = executor.execute_in_sar(sar_env)
    # or with feedback:
    results = executor.execute_in_sar_with_feedback(sar_env, ..., on_subtask_failed=fn)
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.resources_path = os.path.join(base_path, "resources")

        # Populated by run()
        self.task_idx: int = 0
        self.task_name: str = "task"
        self.task_description: str = ""
        self.assignment: Dict[int, int] = {}          # subtask_id -> robot_id (1-based)
        self.parallel_groups: Dict[int, List[int]] = {}  # group_id -> [subtask_ids]
        self._plan_actions: Dict[int, List[str]] = {} # subtask_id -> [SAR action strs]
        self._subtask_results: Dict[int, SubTaskExecutionResult] = {}
        self._dropped_subtasks: Set[int] = set()

        # Metrics
        self.agent_success_counts: List[int] = []  # per-agent successful action count
        self.total_exec: int = 0                   # total non-NoOp actions attempted
        self.success_exec: int = 0                 # successful non-NoOp actions

        self._object_name_map: Dict[str, str] = {}   # lowercase -> original

    # ------------------------------------------------------------------
    # Preparation phase (called by planning_sar.py after PDDL planning)
    # ------------------------------------------------------------------

    def run(
        self,
        task_idx: int = 0,
        task_name: str = "task",
        task_description: str = "",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Load PDDL plan actions + assignment + parallel_groups from disk.
        Returns a summary string (for compatibility with MultiRobotExecutor.run interface).
        """
        self.task_idx = task_idx
        self.task_name = task_name
        self.task_description = task_description

        dag_dir = os.path.join(self.resources_path, "dag_outputs")

        # Load assignment
        assignment_file = os.path.join(dag_dir, f"task_{task_idx}_assignment.json")
        if os.path.exists(assignment_file):
            with open(assignment_file) as f:
                data = json.load(f)
            self.assignment = {int(k): int(v) for k, v in data.get("assignment", {}).items()}
        else:
            print(f"[SARExecutor] Warning: assignment file not found: {assignment_file}")
            self.assignment = {}

        # Load parallel_groups from SubtaskDAG JSON
        dag_json = os.path.join(dag_dir, f"{task_name}_SUBTASK_DAG.json")
        if os.path.exists(dag_json):
            with open(dag_json) as f:
                dag_data = json.load(f)
            raw_pg = dag_data.get("parallel_groups", {})
            self.parallel_groups = {int(k): list(map(int, v)) for k, v in raw_pg.items()}
        else:
            # Fallback: single group with all subtask IDs from assignment
            print(f"[SARExecutor] Warning: DAG JSON not found, using flat group: {dag_json}")
            all_ids = list(self.assignment.keys())
            self.parallel_groups = {0: all_ids}

        # Load raw PDDL plan actions
        self._pddl_actions_raw = self._load_pddl_actions()

        summary = (
            f"SARExecutor ready: {len(self.assignment)} subtask(s), "
            f"{len(self.parallel_groups)} parallel group(s)"
        )
        print(f"[SARExecutor] {summary}")
        return summary

    def _load_pddl_actions(self) -> Dict[int, List[str]]:
        """Read subtask_XX_*_actions.txt files from resources/subtask_pddl_plans/."""
        plans_dir = os.path.join(self.resources_path, "subtask_pddl_plans")
        result: Dict[int, List[str]] = {}
        if not os.path.exists(plans_dir):
            return result
        for fname in os.listdir(plans_dir):
            if not fname.endswith("_actions.txt"):
                continue
            m = re.match(r"subtask_(\d+)_.*_actions\.txt$", fname)
            if not m:
                continue
            sid = int(m.group(1))
            with open(os.path.join(plans_dir, fname)) as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith(";")]
            result[sid] = lines
        return result

    # ------------------------------------------------------------------
    # Object name resolution
    # ------------------------------------------------------------------

    def set_object_names(self, names: List[str]) -> None:
        """
        Register all known object names from the SAR environment.
        Must be called before execute_in_sar() so PDDL tokens can be resolved.
        """
        self._object_name_map = {n.lower(): n for n in names}

    # ------------------------------------------------------------------
    # Execution: translate + run in SAREnv
    # ------------------------------------------------------------------

    def _build_sar_plan(self, subtask_id: int) -> List[str]:
        """Translate raw PDDL actions for one subtask to SAR action strings.

        Also ensures NavigateTo is always present before actions that require
        the agent to be at a specific location (GetSupply, UseSupply, Carry, DropOff).
        PDDL planners sometimes omit NavigateTo when the robot is assumed to
        already be at the target location in the initial state.
        """
        raw_lines = self._pddl_actions_raw.get(subtask_id, [])
        sar_actions: List[str] = []
        for line in raw_lines:
            sar_act = _pddl_to_sar(line, self._object_name_map)
            if sar_act:
                sar_actions.append(sar_act)

        # --- Auto-insert NavigateTo before location-dependent actions ---
        patched: List[str] = []
        for act in sar_actions:
            target = _extract_navigate_target(act)
            if target is not None:
                already_there = (
                    patched
                    and patched[-1].startswith("NavigateTo(")
                    and patched[-1] == f"NavigateTo({target})"
                )
                if not already_there:
                    patched.append(f"NavigateTo({target})")
            patched.append(act)

        return patched

    def execute_in_sar(self, sar_env) -> Dict[int, SubTaskExecutionResult]:
        """
        Execute all subtask plans in the given SAREnv (no feedback/replanning).
        Returns dict of subtask_id -> SubTaskExecutionResult.
        """
        return self._execute(sar_env, on_subtask_failed=None)

    def execute_in_sar_with_feedback(
        self,
        sar_env,
        task_name: str = "task",
        task_description: str = "",
        state_store=None,
        on_subtask_failed: Optional[Callable] = None,
    ) -> Dict[int, SubTaskExecutionResult]:
        """
        Execute all subtask plans in SAREnv with optional feedback/replanning.
        On subtask failure, calls on_subtask_failed(SubTaskExecutionResult).
        If callback returns True, reload plan actions and retry.
        """
        self.task_name = task_name
        self.task_description = task_description
        return self._execute(sar_env, on_subtask_failed=on_subtask_failed)

    def _execute(
        self,
        sar_env,
        on_subtask_failed: Optional[Callable],
    ) -> Dict[int, SubTaskExecutionResult]:
        """
        Core execution loop:
          For each parallel group (in sorted order):
            - Build per-agent action queues for the subtasks in this group
            - At each step: every agent executes its next action (or NoOp)
            - Repeat until all queues are empty
          Track success/failure per subtask.
        """
        self._subtask_results = {}
        self._dropped_subtasks = set()

        num_agents = sar_env.num_agents
        agent_names = sar_env.agent_names   # ['Alice', 'Bob', ...]

        # Reset metrics
        self.agent_success_counts = [0] * num_agents
        self.total_exec = 0
        self.success_exec = 0

        # robot_id (1-based) -> agent_idx (0-based)
        def robot_to_idx(rid: int) -> int:
            return (rid - 1) % num_agents

        # Track checker state before & after for success heuristic
        task_timeout = sar_env.task_timeout

        for group_id in sorted(self.parallel_groups.keys()):
            group_sids = self.parallel_groups[group_id]
            print(f"\n[SARExecutor] === Parallel Group {group_id}: subtasks {group_sids} ===")

            # Build action queues per agent for this group
            # agent_idx -> deque of SAR action strings
            agent_queues: Dict[int, deque] = {i: deque() for i in range(num_agents)}
            # agent_idx -> subtask_id (for tracking which subtask this agent is executing)
            agent_subtask: Dict[int, Optional[int]] = {i: None for i in range(num_agents)}
            # subtask_id -> completed actions
            subtask_completed: Dict[int, List[str]] = {sid: [] for sid in group_sids}
            # subtask_id -> failed flag + error
            subtask_failed: Dict[int, Optional[str]] = {sid: None for sid in group_sids}

            for sid in group_sids:
                sar_plan = self._build_sar_plan(sid)
                if not sar_plan:
                    print(f"[SARExecutor] Subtask {sid}: No plan actions found, marking as failed")
                    subtask_failed[sid] = "No plan actions generated by PDDL planner"
                    continue

                robot_id = self.assignment.get(sid)
                if robot_id is None:
                    print(f"[SARExecutor] Subtask {sid}: No robot assignment, skipping")
                    subtask_failed[sid] = "No robot assigned to subtask"
                    continue

                agent_idx = robot_to_idx(robot_id)
                agent_queues[agent_idx] = deque(sar_plan)
                agent_subtask[agent_idx] = sid
                print(
                    f"[SARExecutor]  Subtask {sid} -> Agent {agent_names[agent_idx]}"
                    f" ({len(sar_plan)} actions)"
                )
                for act in sar_plan:
                    print(f"    {act}")

            # Execution loop for this parallel group
            step = 0
            while any(len(q) > 0 for q in agent_queues.values()):
                step += 1
                if step > task_timeout:
                    # Mark remaining subtasks as timed out
                    for idx, q in agent_queues.items():
                        if q:
                            sid = agent_subtask.get(idx)
                            if sid is not None and subtask_failed.get(sid) is None:
                                subtask_failed[sid] = f"Timeout after {task_timeout} steps"
                    break

                actions_this_step: List[str] = []
                step_agent_sids: List[Optional[int]] = []

                for agent_idx in range(num_agents):
                    q = agent_queues[agent_idx]
                    if q:
                        act = q.popleft()
                        actions_this_step.append(act)
                        step_agent_sids.append(agent_subtask[agent_idx])
                    else:
                        actions_this_step.append("NoOp()")
                        step_agent_sids.append(None)

                print(f"[SARExecutor] Step {step}: {list(zip(agent_names, actions_this_step))}")

                try:
                    _, successes = sar_env.step(actions_this_step)
                except Exception as e:
                    print(f"[SARExecutor] env.step error: {e}")
                    # Mark all active subtasks as failed
                    for agent_idx, sid in enumerate(step_agent_sids):
                        if sid is not None and subtask_failed.get(sid) is None:
                            subtask_failed[sid] = f"Environment error: {e}"
                    break

                # Process results for each agent
                for agent_idx, (act, sid, success) in enumerate(
                    zip(actions_this_step, step_agent_sids, successes)
                ):
                    if sid is None or act == "NoOp()":
                        continue

                    self.total_exec += 1
                    if success:
                        self.success_exec += 1
                        self.agent_success_counts[agent_idx] += 1
                        subtask_completed[sid].append(act)
                    else:
                        # Non-critical failures (e.g., already extinguished region)
                        # are logged but don't abort the subtask immediately
                        print(
                            f"[SARExecutor]  {agent_names[agent_idx]} action FAILED: {act}"
                        )
                        subtask_completed[sid].append(f"FAILED:{act}")

                        # If it's a critical action (UseSupply with wrong type, Carry, DropOff)
                        # abort this subtask
                        if _is_critical_action(act):
                            subtask_failed[sid] = (
                                f"Critical action failed: {act}"
                            )
                            # Clear remaining queue for this agent
                            agent_queues[agent_idx].clear()

            # Build results for this group
            for sid in group_sids:
                error = subtask_failed.get(sid)
                completed = subtask_completed.get(sid, [])
                result = SubTaskExecutionResult(
                    subtask_id=sid,
                    success=(error is None),
                    error_message=error,
                    completed_actions=completed,
                )
                self._subtask_results[sid] = result

                if error is None:
                    print(f"[SARExecutor] Subtask {sid}: SUCCESS ({len(completed)} actions)")
                else:
                    print(f"[SARExecutor] Subtask {sid}: FAILED — {error}")

                    # Invoke feedback/replan callback
                    if on_subtask_failed is not None:
                        replanned = on_subtask_failed(result)
                        if replanned:
                            # Reload plan actions after replanning and retry subtask
                            print(f"[SARExecutor] Replan accepted for subtask {sid}, reloading plan")
                            self._pddl_actions_raw = self._load_pddl_actions()
                            new_plan = self._build_sar_plan(sid)
                            robot_id = self.assignment.get(sid)
                            if robot_id is not None and new_plan:
                                agent_idx = robot_to_idx(robot_id)
                                # Re-execute single subtask
                                retry_result = self._execute_single_subtask(
                                    sar_env, sid, agent_idx, new_plan, num_agents, agent_names, task_timeout
                                )
                                self._subtask_results[sid] = retry_result

        return self._subtask_results

    def _execute_single_subtask(
        self,
        sar_env,
        subtask_id: int,
        agent_idx: int,
        sar_plan: List[str],
        num_agents: int,
        agent_names: List[str],
        task_timeout: int,
    ) -> SubTaskExecutionResult:
        """Execute a single subtask (used for replanning retries)."""
        queue = deque(sar_plan)
        completed: List[str] = []
        error: Optional[str] = None

        step = 0
        while queue:
            step += 1
            if step > task_timeout:
                error = f"Retry timeout after {task_timeout} steps"
                break

            act = queue.popleft()
            actions_this_step = ["NoOp()"] * num_agents
            actions_this_step[agent_idx] = act

            try:
                _, successes = sar_env.step(actions_this_step)
                success = successes[agent_idx]
            except Exception as e:
                error = f"Environment error on retry: {e}"
                break

            self.total_exec += 1
            if success:
                self.success_exec += 1
                if agent_idx < len(self.agent_success_counts):
                    self.agent_success_counts[agent_idx] += 1
                completed.append(act)
            else:
                print(f"[SARExecutor] Retry — {agent_names[agent_idx]} FAILED: {act}")
                completed.append(f"FAILED:{act}")
                if _is_critical_action(act):
                    error = f"Critical action failed on retry: {act}"
                    break

        return SubTaskExecutionResult(
            subtask_id=subtask_id,
            success=(error is None),
            error_message=error,
            completed_actions=completed,
        )

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def _compute_balance_metric(self) -> float:
        """balance = min(x_i) / max(x_i), x_i = agent i의 성공 액션 수.
        모든 서브태스크가 성공한 경우에만 계산, 그렇지 않으면 0."""
        if self._subtask_results and any(
            not r.success for r in self._subtask_results.values()
        ):
            return 0.0
        if not self.agent_success_counts:
            return 0.0
        mx = max(self.agent_success_counts)
        if mx == 0:
            return 0.0
        return min(self.agent_success_counts) / mx

    def _compute_exec_rate(self) -> float:
        """Exec Rate = successful low-level actions / attempted actions."""
        if self.total_exec <= 0:
            return 0.0
        return float(self.success_exec) / float(self.total_exec)

    # ------------------------------------------------------------------
    # Reload after replanning (called by SARTaskManager.reload_executor_with_integrated_dag)
    # ------------------------------------------------------------------

    def reload_plans_and_dag(self) -> bool:
        """Reload plan actions and parallel_groups from disk after replanning."""
        dag_dir = os.path.join(self.resources_path, "dag_outputs")

        # Reload assignment
        assignment_file = os.path.join(dag_dir, f"task_{self.task_idx}_assignment.json")
        if os.path.exists(assignment_file):
            with open(assignment_file) as f:
                data = json.load(f)
            self.assignment = {int(k): int(v) for k, v in data.get("assignment", {}).items()}

        # Reload parallel_groups
        dag_json = os.path.join(dag_dir, f"{self.task_name}_SUBTASK_DAG.json")
        if os.path.exists(dag_json):
            with open(dag_json) as f:
                dag_data = json.load(f)
            raw_pg = dag_data.get("parallel_groups", {})
            self.parallel_groups = {int(k): list(map(int, v)) for k, v in raw_pg.items()}

        # Reload PDDL actions
        self._pddl_actions_raw = self._load_pddl_actions()
        return True


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _is_critical_action(sar_action: str) -> bool:
    """Return True if a failed action should abort the subtask.

    All actions in a PDDL-generated plan are sequential and dependent:
    - NavigateTo must succeed before GetSupply/UseSupply/Carry/DropOff
    - GetSupply must succeed before UseSupply (otherwise inventory is empty)
    - Explore must succeed before NavigateTo(person)/Carry
    Any failure breaks the dependency chain, so all actions are critical.
    """
    non_critical_prefixes = ("NoOp(",)
    return not any(sar_action.startswith(p) for p in non_critical_prefixes)


def _extract_navigate_target(sar_action: str) -> Optional[str]:
    """
    For actions that require the agent to be at a specific location,
    return the target name so a NavigateTo can be auto-inserted before them.

      GetSupply(ReservoirUtah, Sand)       -> "ReservoirUtah"
      Carry(LostPersonTimmy)               -> "LostPersonTimmy"
      DropOff(DepositFacility, Timmy)      -> "DepositFacility"
      StoreSupply(DepositFacility)         -> "DepositFacility"
      NavigateTo / Explore / NoOp          -> None (no insert needed)
    """
    import re as _re
    m = _re.match(r"^(\w+)\((.+)\)$", sar_action)
    if not m:
        return None
    action_name, inner = m.group(1), m.group(2)

    if action_name == "GetSupply":
        # 자동화: insert NavigateTo when PDDL omits gotoobject before getsupply
        return inner.split(",")[0].strip()
    elif action_name == "Carry":
        return inner.strip()
    elif action_name == "DropOff":
        return inner.split(",")[0].strip()
    elif action_name == "StoreSupply":
        return inner.strip()
    return None