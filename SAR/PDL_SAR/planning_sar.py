"""
planning_sar.py — PDL pipeline entry point for the SAR environment.

Runs the full PDL pipeline:
  1. Get SAR scene objects
  2. LLM task decomposition  (SAR-specific prompt)
  3. LLM precondition generation
  4. LLM PDDL problem generation (using allactionrobot.pddl)
  5. FastDownward planning
  6. DAG dependency analysis
  7. Task-to-robot assignment  (round-robin when CP-SAT unavailable)
  8. SARExecutor execution inside SAREnv
  9. (Optional) FeedbackLoop replanning

Usage:
    cd /home/nuc/Desktop/PDL/SAR
    python PDL_SAR/planning_sar.py \\
        --scene 1 --agents 3 \\
        --gpt-version gpt-4o \\
        --api-key-file ~/openai_key.json \\
        [--with-feedback]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------
# Path setup — SAR root, PDL scripts, PDL resources
# (must be done BEFORE any PDL imports)
# -----------------------------------------------------------------------
_SAR_ROOT    = Path(__file__).resolve().parent.parent          # /PDL/SAR
_PDL_SCRIPTS = _SAR_ROOT.parent / "AI2Thor" / "baselines" / "PDL" / "scripts"
_PDL_RESOURCES = _SAR_ROOT.parent / "AI2Thor" / "baselines" / "PDL" / "resources"
_PDLSAR_ROOT = Path(__file__).resolve().parent                 # /PDL/SAR/PDL_SAR
_PDLSAR_RESOURCES = _PDLSAR_ROOT / "resources"

for _p in [str(_SAR_ROOT), str(_PDL_SCRIPTS), str(_PDL_RESOURCES),
           str(_PDLSAR_ROOT), str(_PDLSAR_RESOURCES)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------------------------------------------------
# Mock problematic imports before touching PDL modules
# -----------------------------------------------------------------------

def _install_mocks() -> None:
    """
    Install lightweight stubs for modules that are either not installed
    (ai2thor) or have native-library conflicts (ortools via LP_Module).
    Must be called before importing any PDL module that pulls these in.
    """

    # --- ai2thor ---
    if "ai2thor" not in sys.modules:
        _m = types.ModuleType("ai2thor")
        _c = types.ModuleType("ai2thor.controller")
        class _Ctrl: pass
        _c.Controller = _Ctrl
        _m.controller = _c
        sys.modules["ai2thor"] = _m
        sys.modules["ai2thor.controller"] = _c

    # --- AI2Thor.Tasks.get_scene_init ---
    for _name in ["AI2Thor", "AI2Thor.Tasks", "AI2Thor.Tasks.get_scene_init"]:
        if _name not in sys.modules:
            sys.modules[_name] = types.ModuleType(_name)
    if not hasattr(sys.modules["AI2Thor.Tasks.get_scene_init"], "get_scene_initializer"):
        sys.modules["AI2Thor.Tasks.get_scene_init"].get_scene_initializer = (
            lambda *a, **k: (None, None)
        )

    # --- LP_Module (wraps ortools CP-SAT; native conflict on this system) ---
    if "LP_Module" not in sys.modules:
        _lp = types.ModuleType("LP_Module")
        _lp.assign_subtasks_cp_sat = _sar_assign_subtasks   # SAR fallback
        _lp.binding_pairs_from_subtask_dag = lambda dag: []
        sys.modules["LP_Module"] = _lp

    # --- MultiRobotExecutor (AI2Thor executor; not needed for SAR) ---
    if "MultiRobotExecutor" not in sys.modules:
        _mr = types.ModuleType("MultiRobotExecutor")
        _mr.MultiRobotExecutor = type("MultiRobotExecutor", (), {})
        # SubTaskExecutionResult is defined locally in sar_executor.py
        _mr.SubTaskExecutionResult = type(
            "SubTaskExecutionResult",
            (),
            {"subtask_id": 0, "success": False, "error_message": None, "completed_actions": []},
        )
        _mr._TASK_NAME_MAP = {}
        _mr._TASK_NAME_MAP_LOWER = {}
        sys.modules["MultiRobotExecutor"] = _mr

    # --- auto_config (AI2Thor config helper; not needed for SAR) ---
    if "auto_config" not in sys.modules:
        _ac = types.ModuleType("auto_config")
        _ac.AutoConfig = type("AutoConfig", (), {})
        sys.modules["auto_config"] = _ac


def _sar_assign_subtasks(
    subtasks: List[Dict],
    robot_ids: List[int],
    **_kwargs,
) -> Dict[int, Any]:
    """
    Round-robin assignment. Person rescue subtasks get a pair [lead, support]
    """
    n = len(robot_ids)
    assignment: Dict[int, Any] = {}
    robot_counter = 0
    for st in subtasks:
        sid = st["id"]
        title = st.get("title", "")
        if "rescue" in title.lower():
            lead    = robot_ids[robot_counter % n]
            support = robot_ids[(robot_counter + 1) % n]
            assignment[sid] = [lead, support]
            robot_counter += 2
        else:
            assignment[sid] = robot_ids[robot_counter % n]
            robot_counter += 1
    return assignment


def _augment_assignment_for_rescue(
    assignment: Dict[int, Any],
    parsed_subtasks: List[Dict],
    robot_ids: List[int],
    parallel_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[int, Any]:
    """
    Post-process a CP-SAT assignment (int values only) to give rescue subtasks a second support robot.
    """
    n = len(robot_ids)
    result: Dict[int, Any] = dict(assignment)
    for st in parsed_subtasks:
        sid   = st["id"]
        title = st.get("title", "")
        if "rescue" not in title.lower():
            continue
        lead = result.get(sid)
        if not isinstance(lead, int):
            continue
        lead_idx = next((i for i, r in enumerate(robot_ids) if r == lead), 0)

        # Find robots already busy in the same parallel group
        busy_robots: set = set()
        if parallel_groups:
            for grp_sids in parallel_groups.values():
                if sid in grp_sids:
                    for other_sid in grp_sids:
                        if other_sid == sid:
                            continue
                        other = result.get(other_sid)
                        if isinstance(other, list):
                            busy_robots.update(other)
                        elif other is not None:
                            busy_robots.add(other)
                    break

        # Pick nearest available support robot
        support = None
        for offset in range(1, n):
            candidate = robot_ids[(lead_idx + offset) % n]
            if candidate != lead and candidate not in busy_robots:
                support = candidate
                break
        if support is None:
            support = robot_ids[(lead_idx + 1) % n]  # fallback

        result[sid] = [lead, support]
    return result


def _binding_pairs_from_subtask_dag(subtask_dag) -> List[Tuple[int, int]]:
    """
    Extract binding-constraint pairs from the subtask DAG.
    Returns list of (sid_a, sid_b) where both subtasks must be assigned
    to the same robot (same as binding_pairs_from_subtask_dag in LP_Module).
    """
    pairs: List[Tuple[int, int]] = []
    if subtask_dag is None:
        return pairs
    for e in getattr(subtask_dag, "edges", []) or []:
        t = (getattr(e, "dependency_type", "") or "").lower().strip()
        if t == "binding":
            pairs.append((int(e.from_id), int(e.to_id)))
    return sorted({(a, b) for (a, b) in pairs})


def _lp_assign_via_subprocess(
    subtasks: List[Dict],
    robot_ids: List[int],
    robots_db: List[Dict],
    plan_actions_by_subtask: Dict[int, List[str]],
    objects_ai: str,
    binding_pairs: Optional[List[Tuple[int, int]]] = None,
    parallel_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[int, int]:
    """
    Run LP_Module.assign_subtasks_cp_sat via a subprocess to avoid the
    ortools/protobuf native-library conflict in the main process.
    Falls back to round-robin if the subprocess fails or crashes.
    """
    runner = str(Path(__file__).resolve().parent / "_lp_runner.py")
    payload: Dict[str, Any] = {
        "subtasks":               subtasks,
        "robot_ids":              robot_ids,
        "robots_db":              robots_db,
        "plan_actions_by_subtask": {str(k): v for k, v in plan_actions_by_subtask.items()},
        "objects_ai":             objects_ai,
        "binding_pairs":          [[a, b] for a, b in (binding_pairs or [])],
        "parallel_groups":        parallel_groups,
    }
    try:
        proc = subprocess.run(
            [sys.executable, runner],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            raw = json.loads(proc.stdout)
            print("[LP] CP-SAT assignment succeeded")
            return {int(k): int(v) for k, v in raw.items()}
        print(f"[LP] Subprocess failed (rc={proc.returncode}):\n{proc.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("[LP] Subprocess timed out (60 s)")
    except Exception as exc:
        print(f"[LP] Subprocess error: {exc}")

    print("[LP] Falling back to round-robin assignment")
    return _sar_assign_subtasks(subtasks, robot_ids)


_install_mocks()

# -----------------------------------------------------------------------
# PDL module imports (safe after mocks)
# -----------------------------------------------------------------------
from PDDL_Module import TaskManager  # type: ignore[import]  # noqa: E402
from FeedbackLoopModule import (     # type: ignore[import]  # noqa: E402
    SharedTaskStateStore,
    load_subtask_precond_effects,
    sync_execution_results_to_store,
)

# SAR-specific executor
from sar_executor import SARExecutor, SubTaskExecutionResult, _fires_still_active  # noqa: E402


# -----------------------------------------------------------------------
# SAR object extractor
# -----------------------------------------------------------------------

def get_sar_objects(scene: int, num_agents: int = 3) -> Tuple[str, List[str]]:
    """
    Returns (objects_str, all_names) for the given SAR scene.

    objects_str : formatted Python list string for LLM context.
    all_names   : original-case object names for PDDL token resolution.
    """
    import core
    from env import SAREnv
    from Scenes.get_scene_init import get_scene_initializer

    scene_cls, _ = get_scene_initializer(scene)
    si = scene_cls.SceneInitializer()
    params = si.params

    fire_mapper = core.Field.READABLE_TYPE_MAPPER_FIRE     # {'A': 'CHEMICAL', ...}
    res_mapper  = core.Field.READABLE_TYPE_MAPPER_RESOURCE # {'A': 'SAND', 'B': 'WATER', ...}

    def fire_type_str(tp: str) -> str:
        return fire_mapper.get(tp.upper(), tp).lower()

    def res_type_str(tp: str) -> str:
        return res_mapper.get(tp.upper(), tp).capitalize()

    # fire -> supply mapping  (same type key: 'a' fire needs 'a' resource = Sand)
    fire_supply: Dict[str, str] = {arg.name: res_type_str(arg.tp)
                                   for arg in params.get("fires", [])}

    # Spin up a minimal env to get region names from the controller
    tmp_env = SAREnv(num_agents=1, scene=scene, seed=42)
    tmp_env.reset()
    ctrl = tmp_env.controller
    all_env_names: List[str] = list(ctrl.all_names)

    # Build region -> parent fire mapping from name pattern
    region_parent: Dict[str, str] = {}
    for n in all_env_names:
        m = re.match(r"^(.+?)_Region_\d+$", n, re.IGNORECASE)
        if m:
            parent_lower = m.group(1).lower()
            parent = next(
                (arg.name for arg in params.get("fires", [])
                 if arg.name.lower() == parent_lower),
                m.group(1),
            )
            region_parent[n] = parent

    objects: List[Dict[str, Any]] = []

    # Fires
    for arg in params.get("fires", []):
        regions = sorted(r for r, p in region_parent.items() if p == arg.name)
        objects.append({
            "name": arg.name,
            "type": "Fire",
            "fire_type": fire_type_str(arg.tp),
            "extinguish_with": fire_supply[arg.name],
            "regions": regions,
        })

    # Flammable regions
    for rname, parent in sorted(region_parent.items()):
        objects.append({
            "name": rname,
            "type": "Flammable",
            "parent_fire": parent,
            "extinguish_with": fire_supply.get(parent, "Water"),
        })

    # Reservoirs
    for arg in params.get("reservoirs", []):
        objects.append({
            "name": arg.name,
            "type": "Reservoir",
            "provides_supply": res_type_str(arg.tp),
        })

    # Deposits
    for arg in params.get("deposits", []):
        objects.append({"name": arg.name, "type": "Deposit"})

    # Persons
    for arg in params.get("persons", []):
        objects.append({
            "name": arg.name,
            "type": "Person",
            "status": "unknown_location (must Explore to find)",
        })

    # Supply type nodes
    supply_types = sorted({fire_supply[f] for f in fire_supply})
    for s in supply_types:
        objects.append({"name": s, "type": "Supply"})

    all_names = [o["name"] for o in objects]
    objects_str = f"\n\nobjects = {objects}"
    return objects_str, all_names


# -----------------------------------------------------------------------
# SARTaskManager
# -----------------------------------------------------------------------

class SARTaskManager(TaskManager):
    """
    TaskManager subclass for SAR:
    - SAR PDDL domain  (allactionrobot.pddl)
    - SAR decompose prompt  (sar_decompose.py)
    - SAR robot skills  (sar_robots.py)
    - SARExecutor instead of MultiRobotExecutor
    - Round-robin task assignment instead of CP-SAT
    """

    def __init__(self, sar_pdl_root: str, gpt_version: str, api_key_file: str):
        super().__init__(
            base_path=sar_pdl_root,
            gpt_version=gpt_version,
            api_key_file=api_key_file,
        )
        self._sar_pdl_root = sar_pdl_root

    # ------------------------------------------------------------------
    # Robot skills
    # ------------------------------------------------------------------

    def _available_robot_skills(self, _robot_ids) -> List[str]:
        import sar_robots
        return sorted(sar_robots.robots[0]["skills"])

    # ------------------------------------------------------------------
    # Decompose prompt -> sar_decompose.py
    # ------------------------------------------------------------------

    def _generate_decomposed_plan(
        self, task: str, domain_content: str, robots, objects_ai: str
    ) -> str:
        prompt_path = os.path.join(
            self._sar_pdl_root, "data", "pythonic_plans", "sar_decompose.py"
        )
        decompose_prompt = self.file_processor.read_file(prompt_path)

        prompt = (
            "The following list is the ONLY set of objects that exist in the current SAR environment.\n"
            "When writing subtasks and actions, you MUST ground every referenced object to this list.\n"
            f"\nENVIRONMENT OBJECTS = {objects_ai}\n\n"
            "If you reference an object not in ENVIRONMENT OBJECTS, your answer is INVALID.\n\n"
            f"\nAVAILABLE ROBOT SKILLS = {robots}\n\n"
            "You MUST use only actions whose names are included in AVAILABLE ROBOT SKILLS.\n\n"
            "IMPORTANT CONSTRAINT: To move a group of humans, at least 2 agents are required. "
            "Thus, explicit multi-agent collaboration is required for any person rescue subtask. "
            "Every subtask that involves rescuing (Carry + DropOff) a person MUST be annotated "
            "with [Multi-Agent: 2 robots required for Carry+DropOff].\n\n"
            "The following are examples of the expected output format.\n\n"
        )
        prompt += decompose_prompt
        prompt += "\n\n# GENERAL TASK DECOMPOSITION\n"
        prompt += "Decompose and parallelize subtasks where ever possible.\n\n"
        prompt += f"# Task Description: {task}"

        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(
                prompt, self.gpt_version, max_tokens=3000, stop=None, frequency_penalty=0.0
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(
                messages, self.gpt_version, max_tokens=3000, frequency_penalty=0.0
            )
        return text

    # ------------------------------------------------------------------
    # LLM PDDL Validator (SAR-specific rules, no AI2Thor OPENABLE rules)
    # ------------------------------------------------------------------

    def run_llmvalidator(self) -> None:
        """SAR-specific PDDL validator — overrides the AI2Thor-specific prompt."""
        try:
            src_dir  = self.file_processor.subtask_pddl_problems_path
            src_dir2 = self.file_processor.precondition_subtasks_path

            problem_files      = sorted(f for f in os.listdir(src_dir)  if f.endswith(".pddl"))
            precondition_files = sorted(f for f in os.listdir(src_dir2) if f.endswith(".txt"))

            self.file_processor.clean_directory(self.file_processor.validated_subtask_path)
            self.validated_plan = []

            for problem_file, precondition_file in zip(problem_files, precondition_files):
                problem_file_full      = os.path.join(src_dir,  problem_file)
                precondition_file_full = os.path.join(src_dir2, precondition_file)

                domain_name = self.file_processor.extract_domain_name(problem_file_full)
                if not domain_name:
                    print(f"[VALIDATOR] No domain name in {problem_file}")
                    continue

                domain_file = self.file_processor.find_domain_file(domain_name)
                if not domain_file:
                    print(f"[VALIDATOR] Domain file not found for '{domain_name}'")
                    continue

                domain_content       = self.file_processor.read_file(domain_file)
                problem_content      = self.file_processor.read_file(problem_file_full)
                precondition_content = self.file_processor.read_file(precondition_file_full)

                prompt = (
                    "You are a strict PDDL problem validator and repair system for Fast Downward.\n"
                    "The DOMAIN is the single source of truth.\n"
                    "Your job is to REWRITE the PROBLEM so that it is consistent with the DOMAIN "
                    "and solvable when the task intent is achievable.\n\n"

                    "CRITICAL RULES for SAR PDDL:\n"
                    "1. Supply types (e.g., Sand, Water) must be declared as objects of type object "
                    "and have (is-supply <supply>) in :init.\n"
                    "2. Fire regions must have (is-region <reg>), (fire-active <reg>), and "
                    "(region-of <reg> <fire>) in :init.\n"
                    "3. Fire objects must have (is-fire <f>) in :init.\n"
                    "4. (supply-for-fire <supply> <fire>) must link each supply type to its fire.\n"
                    "5. Reservoirs must have (is-reservoir <res>) and (has-resource <res> <supply>) "
                    "in :init.\n"
                    "6. Deposit locations must have (is-deposit <d>) in :init.\n"
                    "7. Lost persons must have (is-person <p>) in :init. "
                    "Do NOT set (person-found ?p) in :init unless Explore has already completed.\n"
                    "8. The robot must be declared under :objects as type robot.\n"
                    "9. GoToObject has NO preconditions — the robot can navigate freely.\n"
                    "10. Explore sets (person-found ?p) for ALL persons; it must precede Carry.\n"
                    "11. Carry requires (at ?r ?p), (is-person ?p), and (person-found ?p).\n"
                    "12. DropOff requires (at ?r ?d), (carrying ?r ?p), and (is-deposit ?d).\n"
                    "13. UseSupply targets a fire REGION (not the fire object itself). "
                    "The region must satisfy (region-of ?reg ?fire) and (supply-for-fire ?s ?fire).\n"
                    "14. Only reference objects that already exist in the environment — "
                    "do not invent new object names.\n\n"

                    f"Precondition description:\n{precondition_content}\n\n"
                    f"Domain (authoritative):\n{domain_content}\n\n"
                    f"Problem (to be repaired):\n{problem_content}\n\n"

                    "Output ONLY the corrected PDDL starting with (define"
                )

                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(
                        prompt, self.gpt_version,
                        max_tokens=1400, stop=["def"], frequency_penalty=0.0
                    )
                else:
                    messages = [
                        {"role": "system",
                         "content": "You are a PDDL validator. Output ONLY corrected PDDL."},
                        {"role": "user", "content": prompt},
                    ]
                    _, text = self.llm.query_model(
                        messages, self.gpt_version,
                        max_tokens=1400, frequency_penalty=0.0
                    )

                validated = self.file_processor.normalize_pddl(text)
                self.validated_plan.append(validated)

                out_path = os.path.join(self.file_processor.validated_subtask_path, problem_file)
                self.file_processor.write_file(out_path, validated)
                print(f"[VALIDATOR] Wrote: {out_path}")

        except Exception as exc:
            print(f"Error in run_llmvalidator: {exc}")
            raise

    # ------------------------------------------------------------------
    # Fast-Downward planner (override to point at the PDL installation)
    # ------------------------------------------------------------------

    def run_planners(self) -> None:
        """
        Override run_planners to use the fast-downward installation that
        lives alongside the AI2Thor PDL scripts, not under PDL_SAR/.
        """
        pdl_base = str(_PDL_SCRIPTS.parent)   # .../AI2Thor/baselines/PDL/
        saved_base_path = self.base_path
        self.base_path = pdl_base
        try:
            super().run_planners()
        finally:
            self.base_path = saved_base_path

    # ------------------------------------------------------------------
    # Main pipeline: process_tasks_sar
    # ------------------------------------------------------------------

    def process_tasks_sar(
        self,
        test_tasks: List[str],
        robot_ids: List[List[int]],
        objects_ai: str,
        all_object_names: List[str],
        sar_env,
        run_with_feedback: bool = False,
        max_replan_retries: int = 2,
    ) -> None:
        """
        Full PDL pipeline for SAR.

        Parameters
        ----------
        test_tasks       : list of natural-language task strings
        robot_ids        : robot-id lists per task, e.g. [[1, 2, 3]]
        objects_ai       : formatted objects string for LLM context
        all_object_names : original-case object names (for PDDL token resolution)
        sar_env          : initialized SAREnv (already .reset())
        run_with_feedback: enable feedback/replanning loop
        max_replan_retries: max replan attempts per subtask failure
        """
        self.objects_ai              = objects_ai
        self.available_robot_skills  = self._available_robot_skills(robot_ids)

        # Reset result containers
        self.decomposed_plan        = []
        self.parsed_subtasks        = []
        self.precondition_subtasks  = []
        self.subtask_pddl_problems  = []
        self.validated_plan         = []
        self.subtask_pddl_plans     = []

        # Load SAR PDDL domain (allactionrobot.pddl lives in resources/)
        domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
        domain_content = self.file_processor.read_file(domain_path)

        for task_idx, (task, task_robot_ids) in enumerate(zip(test_tasks, robot_ids)):
            print(f"\n{'='*60}")
            print(f"[PDL-SAR] Task {task_idx + 1}/{len(test_tasks)}: {task}")
            print(f"{'='*60}")

            self.clean_all_resources_directories()

            # ---- Step 1: Task Decomposition ----
            print("\n[Step 1] Task decomposition (LLM)...")
            decomposed_plan = self._generate_decomposed_plan(
                task, domain_content, self.available_robot_skills, objects_ai
            )
            self.decomposed_plan.append(decomposed_plan)
            print("✓ Decomposed plan:\n", decomposed_plan)

            parsed_subtasks = self._decomposed_plan_to_subtasks(decomposed_plan)
            print(f"✓ Parsed {len(parsed_subtasks)} subtasks")

            # ---- Step 2: Preconditions ----
            print("\n[Step 2] Generating preconditions (LLM)...")
            precondition_subtasks = self._generate_precondition_subtasks(
                parsed_subtasks, domain_content, self.available_robot_skills, objects_ai
            )
            self.precondition_subtasks.append(precondition_subtasks)
            print("✓ Preconditions generated")

            for item in precondition_subtasks:
                sid   = item.get("subtask_id", -1)
                title = item.get("subtask_title", "untitled")
                text  = item.get("pre_goal_text", "")
                safe  = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title).strip("_")
                fname = f"pre_{sid:02d}_{safe}.txt"
                self.file_processor.write_file(
                    os.path.join(self.file_processor.precondition_subtasks_path, fname), text
                )

            # ---- Step 3: PDDL Problem Generation ----
            print("\n[Step 3] Generating PDDL problems (LLM)...")
            subtask_pddl_problems = self._generate_subtask_pddl_problems(
                precondition_subtasks, domain_content, self.available_robot_skills, objects_ai
            )
            self.subtask_pddl_problems.append(subtask_pddl_problems)
            print("✓ PDDL problems generated")

            for item in subtask_pddl_problems:
                sid   = item["subtask_id"]
                title = item["subtask_title"]
                text  = item["problem_text"]
                safe  = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title).strip("_")
                fname = f"subtask_{sid:02d}_{safe}.pddl"
                self.file_processor.write_file(
                    os.path.join(self.file_processor.subtask_pddl_problems_path, fname), text
                )

            # ---- Step 4: LLM Validation + FastDownward Planning ----
            print("\n[Step 4] Validating PDDL (LLM) and running FastDownward planner...")
            self._validate_and_plan()
            print("✓ PDDL validated and plans generated")

            # ---- Step 5: DAG Analysis ----
            print("\n[Step 5] Generating dependency DAG...")
            self.generate_dag()
            print("✓ DAG generated")

            # ---- Step 6: Task Assignment (CP-SAT LP) ----
            print("\n[Step 6] Assigning subtasks to agents (CP-SAT LP)...")
            plan_actions_by_sid = self._load_plan_actions_by_subtask_id()

            binding_pairs = _binding_pairs_from_subtask_dag(self.subtask_dag)
            pg_for_lp: Optional[Dict[str, List[int]]] = None
            if self.subtask_dag and hasattr(self.subtask_dag, "parallel_groups"):
                pg_for_lp = {
                    str(k): v for k, v in self.subtask_dag.parallel_groups.items()
                }

            import sar_robots as _sar_robots_mod  # type: ignore[import]  # resolved via sys.path
            assignment = _lp_assign_via_subprocess(
                subtasks               = parsed_subtasks,
                robot_ids              = task_robot_ids,
                robots_db              = _sar_robots_mod.robots,
                plan_actions_by_subtask= plan_actions_by_sid,
                objects_ai             = objects_ai,
                binding_pairs          = binding_pairs,
                parallel_groups        = pg_for_lp,
            )
            
            assignment = _augment_assignment_for_rescue(
                assignment, parsed_subtasks, task_robot_ids,
                parallel_groups=pg_for_lp,
            )

            self.task_assignment = assignment

            print("\n" + "=" * 50)
            print("✓ Task Assignment")
            print("=" * 50)
            for sid, rid in sorted(assignment.items()):
                title = next(
                    (st["title"] for st in parsed_subtasks if st["id"] == sid),
                    f"Subtask {sid}",
                )
                if isinstance(rid, list):
                    names = "+".join(
                        sar_env.agent_names[(r - 1) % sar_env.num_agents] for r in rid
                    )
                    print(f"  Subtask {sid} ({title}) -> Robots {rid} ({names}) [multi-agent]")
                else:
                    agent_name = sar_env.agent_names[(rid - 1) % sar_env.num_agents]
                    print(f"  Subtask {sid} ({title}) -> Robot {rid} ({agent_name})")
            print("=" * 50)

            # Save assignment JSON (read by SARExecutor)
            dag_dir = os.path.join(self.resources_path, "dag_outputs")
            os.makedirs(dag_dir, exist_ok=True)
            assignment_data = {
                "task_idx": task_idx,
                "agent_count": len(task_robot_ids),
                "assignment": {str(k): v for k, v in assignment.items()},
                "subtasks": [
                    {"id": st["id"], "title": st["title"], "robot": assignment.get(st["id"])}
                    for st in parsed_subtasks
                ],
            }
            assignment_path = os.path.join(dag_dir, f"task_{task_idx}_assignment.json")
            with open(assignment_path, "w") as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Assignment saved: {assignment_path}")

            # ---- Step 7: SARExecutor ----
            print("\n[Step 7] Executing plans in SAR environment...")
            executor = SARExecutor(self._sar_pdl_root)
            executor.run(task_idx=task_idx, task_name="task", task_description=task)
            executor.set_object_names(all_object_names)

            if run_with_feedback:
                task_name_fb = "task"
                state_store  = SharedTaskStateStore(self._sar_pdl_root, task_name_fb)
                retry_counts: Dict[int, int] = {}

                def _on_fail(failed_result: SubTaskExecutionResult) -> bool:
                    sid   = failed_result.subtask_id
                    count = retry_counts.get(sid, 0)
                    if count >= max_replan_retries:
                        print(f"[Feedback] Max retries ({max_replan_retries}) for subtask {sid}")
                        return False
                    print(f"[Feedback] Replanning subtask {sid} (attempt {count + 1})")
                    retry_counts[sid] = count + 1

                    current = dict(executor._subtask_results)
                    precond_eff = load_subtask_precond_effects(self._sar_pdl_root)
                    eff_ok = {s: precond_eff.get(s, []) for s, r in current.items() if r.success}
                    sync_execution_results_to_store(state_store, current,
                                                    effects_by_subtask_id=eff_ok)

                    replan_result = self.run_feedback_replan(
                        task_idx=task_idx,
                        task_name=task_name_fb,
                        execution_results=current,
                        domain_content=domain_content,
                        objects_ai=objects_ai,
                        state_store=state_store,
                        task_robot_ids=task_robot_ids,
                        floor_plan=None,
                        executor=executor,
                    )
                    if replan_result == "fully_replanned":
                        executor.reload_plans_and_dag()
                        return True
                    return False

                results = executor.execute_in_sar_with_feedback(
                    sar_env,
                    task_name=task_name_fb,
                    task_description=task,
                    state_store=state_store,
                    on_subtask_failed=_on_fail,
                )
            else:
                results = executor.execute_in_sar(sar_env)

            # ---- Summary ----
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


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PDL pipeline for the SAR (Search And Rescue) environment"
    )
    p.add_argument("--scene",            type=int,  default=1,
                   help="SAR scene number 1-5  (default: 1)")
    p.add_argument("--agents",           type=int,  default=3,
                   help="Number of agents 1-6  (default: 3)")
    p.add_argument("--gpt-version",      type=str,  default="gpt-4o",
                   help="OpenAI model name  (default: gpt-4o)")
    p.add_argument("--api-key-file",     type=str,
                   default=str(Path.home() / "openai_key.json"),
                   help="Path to OpenAI API key JSON file")
    p.add_argument("--with-feedback",    action="store_true",
                   help="Enable feedback/replanning loop on subtask failure")
    p.add_argument("--max-replan-retries", type=int, default=2,
                   help="Max replan attempts per subtask  (default: 2)")
    p.add_argument("--seed",             type=int,  default=42,
                   help="Random seed for environment  (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sar_pdl_root = str(_PDLSAR_ROOT)

    print("\n" + "=" * 60)
    print("  PDL-SAR: Planning and Decision-making for Search & Rescue")
    print("=" * 60)
    print(f"  Scene:    {args.scene}")
    print(f"  Agents:   {args.agents}")
    print(f"  Model:    {args.gpt_version}")
    print(f"  Feedback: {args.with_feedback}")
    print("=" * 60)

    # Build SAR environment
    from env import SAREnv
    sar_env = SAREnv(num_agents=args.agents, scene=args.scene, seed=args.seed)
    sar_env.reset()
    task_str = sar_env.task
    print(f"\n  Task: {task_str}\n")

    # Extract objects for LLM context
    # NOTE: get_sar_objects() internally creates and resets a temporary SAREnv,
    # which can corrupt shared scene state. We re-reset sar_env afterward to restore it.
    objects_ai, all_object_names = get_sar_objects(args.scene, args.agents)
    print(f"  Objects: {all_object_names}\n")

    # Re-initialize sar_env to clear any shared state contamination from tmp_env in get_sar_objects()
    sar_env.reset()
    task_str = sar_env.task  # refresh task string after re-reset

    # Robot IDs: 1-based, one list per task
    robot_ids  = [list(range(1, args.agents + 1))]
    test_tasks = [task_str]

    # Run pipeline
    task_manager = SARTaskManager(
        sar_pdl_root = sar_pdl_root,
        gpt_version  = args.gpt_version,
        api_key_file = args.api_key_file,
    )
    task_manager.process_tasks_sar(
        test_tasks       = test_tasks,
        robot_ids        = robot_ids,
        objects_ai       = objects_ai,
        all_object_names = all_object_names,
        sar_env          = sar_env,
        run_with_feedback    = args.with_feedback,
        max_replan_retries   = args.max_replan_retries,
    )


if __name__ == "__main__":
    main()