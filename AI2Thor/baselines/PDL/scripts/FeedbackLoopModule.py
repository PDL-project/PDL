"""
피드백 루프: Core Execution Logic

[Core Execution Logic]
1. Grouping & Agent Assignment
   - DAG에서 의존성 없이 병렬 실행 가능한 서브태스크들을 하나의 '그룹'으로 묶음.
   - 그룹당 전담 에이전트(GroupAgent) 배정 → 실패 시 '로컬 피드백'.
   - 실패는 중앙에서 감지(CentralFailureDetector).

2. Real-time State Update
   - 그룹 단위가 아니라, 개별 서브태스크 하나가 끝날 때마다 즉시 그룹/전역 상태를
     업데이트하고 공유 데이터베이스(JSON)에 반영.

3. Partial Re-planning (Fault Tolerance)
   - 특정 서브태스크 실패 시 해당 병렬 그룹 내부에 대해서만 재계획.
   - 재계획 시 주입 정보: 로컬 환경, 성공 데이터(Effects, 불변), 실패 분석, 미수행 과제.
   - 재계획 범위: 실패 그룹을 다시 서브태스크로 분해(Recursive Decomposition) 후
     실패 시점/대체 로직부터 재계획. 이미 성공한 서브태스크는 유지.

[Technical Requirements]
- State Machine: PENDING -> RUNNING -> SUCCESS/FAILED (엄격).
- Context Isolation: 성공 데이터(Effects) 불변성 유지.
- Recursive Decomposition: 재계획 시 분해 로직 재호출.
- Concurrency: 병렬 그룹 간 데이터 경합 방지 (그룹별 비중첩 subtask ID + 저장 시 락).

[통합]
- 실시간 상태 반영: 실행기에서 서브태스크 시작 시 store.set_running(subtask_id),
  완료 시 store.update_subtask_on_completion(subtask_id, success, ...) 호출 권장.
- 실행 종료 후 한 번에 반영: sync_execution_results_to_store(store, execution_results).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, FrozenSet

# DAG 구조는 DAG_Module과 동일한 JSON 형식
# task_SUBTASK_DAG.json: { "nodes": [...], "edges": [...], "parallel_groups": {...} }


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class SubtaskState(Enum):
    """서브태스크 상태: 엄격한 상태 전이 PENDING -> RUNNING -> SUCCESS | FAILED"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def _state_from_success(success: bool) -> SubtaskState:
    return SubtaskState.SUCCESS if success else SubtaskState.FAILED


# ---------------------------------------------------------------------------
# Shared State Store (Real-time update per subtask, persist to DB)
# ---------------------------------------------------------------------------

class SharedTaskStateStore:
    """
    개별 서브태스크가 끝날 때마다 즉시 상태를 반영하고 공유 DB(JSON)에 저장.
    Concurrency: 그룹별로 서브태스크 ID가 겹치지 않으므로, 그룹 간 경합 없음.
    저장 시에만 락 사용하여 파일 경합 방지.
    """

    def __init__(self, base_path: str, task_name: str = "task"):
        self.base_path = base_path
        self.task_name = task_name
        self._path = self._state_file_path()
        self._store: Dict[int, Dict[str, Any]] = {}  # subtask_id -> {state, error_message?, effects?}
        self._lock = threading.Lock()

    def _state_file_path(self) -> str:
        resources = os.path.join(self.base_path, "resources")
        dag_out = os.path.join(resources, "dag_outputs")
        os.makedirs(dag_out, exist_ok=True)
        return os.path.join(dag_out, f"{self.task_name}_FEEDBACK_STATE.json")

    def init_from_dag(self, parallel_groups: Dict[int, List[int]]) -> None:
        """DAG parallel_groups로 모든 서브태스크를 PENDING으로 초기화."""
        with self._lock:
            self._store.clear()
            for _gid, sids in parallel_groups.items():
                for sid in sids:
                    if not isinstance(sid, int) or sid <= 0:
                        continue
                    self._store[sid] = {
                        "state": SubtaskState.PENDING.value,
                        "error_message": None,
                        "effects": None,
                    }
            self._persist_locked()

    def update_subtask_on_completion(
        self,
        subtask_id: int,
        success: bool,
        error_message: Optional[str] = None,
        effects: Optional[List[str]] = None,
        completed_actions: Optional[List[str]] = None,
    ) -> None:
        """
        개별 서브태스크 하나가 끝날 때마다 호출. 즉시 상태 반영 후 공유 DB에 저장.
        effects: 성공 시에만 전달하며, 불변으로 저장됨.
        completed_actions: 실패 전까지 성공적으로 실행된 액션 목록 (부분 effect 추적용).
        """
        state = _state_from_success(success)
        with self._lock:
            if subtask_id not in self._store:
                self._store[subtask_id] = {"state": SubtaskState.PENDING.value, "error_message": None, "effects": None}
            self._store[subtask_id]["state"] = state.value
            self._store[subtask_id]["error_message"] = error_message if not success else None
            if success and effects is not None:
                self._store[subtask_id]["effects"] = list(effects)
            if completed_actions is not None:
                self._store[subtask_id]["completed_actions"] = list(completed_actions)
            self._persist_locked()

    def set_running(self, subtask_id: int) -> None:
        """실행 시작 시 RUNNING으로 전이."""
        with self._lock:
            if subtask_id in self._store:
                self._store[subtask_id]["state"] = SubtaskState.RUNNING.value
                self._persist_locked()

    def _persist_locked(self) -> None:
        """호출 전에 self._lock이 잡혀 있어야 함."""
        try:
            with open(self._path, "w") as f:
                json.dump(self._store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[FeedbackLoop] Failed to persist state: {e}")

    def load(self) -> None:
        """디스크에서 상태 로드."""
        with self._lock:
            if os.path.exists(self._path):
                try:
                    with open(self._path, "r") as f:
                        self._store = json.load(f)
                    for sid in self._store:
                        self._store[sid] = {k: v for k, v in self._store[sid].items()}
                except Exception as e:
                    print(f"[FeedbackLoop] Failed to load state: {e}")

    def get_state(self, subtask_id: int) -> SubtaskState:
        with self._lock:
            s = self._store.get(subtask_id, {}).get("state", SubtaskState.PENDING.value)
        try:
            return SubtaskState(s)
        except ValueError:
            return SubtaskState.PENDING

    def get_success_effects_immutable(self) -> Dict[int, FrozenSet[str]]:
        """
        이미 성공한 서브태스크들의 Effects만 반환. 재계획 시 컨텍스트 주입용.
        불변성: 복사본(frozenset)으로 반환하여 수정/오염 방지.
        """
        with self._lock:
            out = {}
            for sid, rec in self._store.items():
                if rec.get("state") != SubtaskState.SUCCESS.value:
                    continue
                eff = rec.get("effects")
                if eff is not None:
                    out[sid] = frozenset(eff)
                else:
                    out[sid] = frozenset()
            return out

    def get_failed_ids(self) -> List[int]:
        with self._lock:
            return [sid for sid, rec in self._store.items() if rec.get("state") == SubtaskState.FAILED.value]

    def get_pending_ids(self) -> List[int]:
        with self._lock:
            return [sid for sid, rec in self._store.items() if rec.get("state") == SubtaskState.PENDING.value]

    def get_failure_info(self, subtask_id: int) -> Tuple[Optional[str], Optional[str]]:
        """(error_message, state) for failed subtask."""
        with self._lock:
            rec = self._store.get(subtask_id, {})
            return rec.get("error_message"), rec.get("state")

    def get_completed_actions(self, subtask_id: int) -> Optional[List[str]]:
        """실패 전까지 성공적으로 실행된 액션 목록 반환."""
        with self._lock:
            rec = self._store.get(subtask_id, {})
            return rec.get("completed_actions")

    def get_group_state(
        self, group_id: int, subtask_ids_in_group: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """그룹 내 (success_ids, failed_ids, pending_ids)."""
        success_ids = []
        failed_ids = []
        pending_ids = []
        for sid in subtask_ids_in_group:
            s = self.get_state(sid)
            if s == SubtaskState.SUCCESS:
                success_ids.append(sid)
            elif s == SubtaskState.FAILED:
                failed_ids.append(sid)
            else:
                pending_ids.append(sid)
        return success_ids, failed_ids, pending_ids


# ---------------------------------------------------------------------------
# Central Failure Detection
# ---------------------------------------------------------------------------

class CentralFailureDetector:
    """실패는 중앙에서 감지. SharedTaskStateStore를 읽어 실패한 서브태스크 ID 목록 반환."""

    def __init__(self, store: SharedTaskStateStore):
        self.store = store

    def get_failed_subtask_ids(self) -> List[int]:
        return self.store.get_failed_ids()

    def get_failed_group_ids(
        self, parallel_groups: Dict[int, List[int]]
    ) -> List[int]:
        """실패한 서브태스크가 속한 그룹 ID 목록 (중복 제거, 정렬)."""
        failed = set(self.store.get_failed_ids())
        group_ids = []
        for gid, sids in parallel_groups.items():
            if any(sid in failed for sid in sids):
                group_ids.append(gid)
        return sorted(group_ids)


# ---------------------------------------------------------------------------
# Group Agent (로컬 피드백: 그룹 전담 에이전트)
# ---------------------------------------------------------------------------

class GroupAgent:
    """
    그룹당 하나의 전담 에이전트. 해당 그룹 내 서브태스크 실패 시 로컬 피드백(재계획).
    """

    def __init__(self, llm_handler: Any, gpt_version: str = "gpt-4o"):
        self.llm = llm_handler
        self.gpt_version = gpt_version

    def replan_subtask(
        self,
        subtask_id: int,
        subtask_name: str,
        current_actions: List[str],
        error_message: str,
        domain_content: str,
        problem_content: str,
        success_effects_context: Optional[str] = None,
        completed_actions: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ) -> List[str]:
        """
        실패한 서브태스크에 대해 LLM으로 대안 액션 시퀀스 생성.
        success_effects_context: 이미 성공한 다른 서브태스크의 effects (재계획 시 반영용).
        completed_actions: 실패 전까지 성공적으로 실행된 액션 목록.
        반환: PDDL plan 형식과 동일한 액션 문자열 리스트.
        """
        actions_str = "\n".join(current_actions) if current_actions else "(none)"
        effects_section = ""
        if success_effects_context and success_effects_context.strip():
            effects_section = f"""
## Already achieved (success effects from other subtasks — use these as given, do not re-do)
{success_effects_context.strip()}

"""
        completed_section = ""
        if completed_actions:
            completed_str = "\n".join(completed_actions)
            completed_section = f"""
## Actions already executed successfully (these effects are now part of the current state)
{completed_str}

IMPORTANT: The above actions were already executed and their effects are real.
For example, if "pickupobject robot1 mug diningtable" was completed, the robot is NOW holding the mug.
Your new plan must account for this current state — do NOT repeat these actions.

"""
        prompt = f"""You are a Subtask Manager (Group Agent). You manage one SUBTASK. The subtask failed during execution. Propose a NEW sequence of primitive PDDL actions to achieve the SAME subtask goal, avoiding the cause of failure.
{effects_section}{completed_section}## Subtask
- ID: {subtask_id}
- Name: {subtask_name}

## Current plan (failed)
{actions_str}

## Execution error
{error_message}

## Domain (reference)
{domain_content[:3000]}...

## Problem (reference)
{problem_content[:2000]}...

## Rules
- Output ONLY one action per line, in the same format as the current plan (e.g. "gotoobject robot1 apple (1)").
- Use the same domain actions and object names from the problem.
- Take into account the "Already achieved" effects and "Actions already executed" above.
- Do not repeat actions that were already successfully executed.
- Do not include any explanation, only the action lines.
"""
        if "gpt" in self.gpt_version.lower():
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=max_tokens, frequency_penalty=0.0)
        else:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=max_tokens, stop=None, frequency_penalty=0.0)

        lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
        action_pattern = re.compile(r"^[a-zA-Z_]+\s+.*\s+\(\d+\)\s*$")
        actions = [ln for ln in lines if action_pattern.match(ln)]
        if not actions:
            actions = [ln for ln in lines if re.match(r"^[a-zA-Z_]+\s+", ln)]
        return actions


# ---------------------------------------------------------------------------
# Partial Re-planning: Context Injection & Recursive Decomposition
# ---------------------------------------------------------------------------

class ReplanContext:
    """재계획 시 주입할 컨텍스트 (불변 성공 데이터 포함)."""

    def __init__(
        self,
        local_env: str,
        success_effects: Dict[int, FrozenSet[str]],
        failed_subtask_id: int,
        failure_reason: str,
        remaining_pending_ids: List[int],
        completed_actions: Optional[List[str]] = None,
    ):
        self.local_env = local_env
        self.success_effects = success_effects  # 불변 유지
        self.failed_subtask_id = failed_subtask_id
        self.failure_reason = failure_reason
        self.remaining_pending_ids = remaining_pending_ids
        self.completed_actions = completed_actions  # 실패 전까지 성공한 액션 목록


class PartialReplanner:
    """
    특정 병렬 그룹 내부에 대해서만 재계획.
    - 컨텍스트 주입: 로컬 환경, 성공 데이터(불변), 실패 분석, 미수행 과제.
    - 재계획 범위: 실패 그룹을 다시 서브태스크로 분해(Recursive Decomposition) 후
      실패 시점/대체 로직부터 재계획. 이미 성공한 서브태스크는 유지.
    """

    def __init__(
        self,
        store: SharedTaskStateStore,
        group_agent: GroupAgent,
        decomposition_callback: Optional[Callable[[int, List[int], ReplanContext], Optional[Dict[int, List[str]]]]] = None,
    ):
        self.store = store
        self.group_agent = group_agent
        self.decomposition_callback = decomposition_callback
        # last replan metadata (used by caller to integrate DAG correctly)
        self.last_replaced_ids: List[int] = []
        self.last_replanned_ids: List[int] = []

    def build_context_for_replan(
        self,
        group_id: int,
        subtask_ids_in_group: List[int],
        local_env: str,
    ) -> Optional[ReplanContext]:
        """
        재계획 주입 정보 구성.
        ① 로컬 환경, ② 성공 데이터(Effects, 불변), ③ 실패 분석, ④ 미수행 과제.
        """
        success_ids, failed_ids, pending_ids = self.store.get_group_state(group_id, subtask_ids_in_group)
        if not failed_ids:
            return None
        failed_id = failed_ids[0]
        err, _ = self.store.get_failure_info(failed_id)
        success_effects = self.store.get_success_effects_immutable()
        completed_actions = self.store.get_completed_actions(failed_id)
        return ReplanContext(
            local_env=local_env,
            success_effects=success_effects,
            failed_subtask_id=failed_id,
            failure_reason=err or "Unknown error",
            remaining_pending_ids=pending_ids,
            completed_actions=completed_actions,
        )

    def replan_group(
        self,
        group_id: int,
        subtask_ids_in_group: List[int],
        context: ReplanContext,
        domain_content: str,
        problem_content_by_id: Dict[int, str],
        current_actions_by_id: Dict[int, List[str]],
        subtask_name_by_id: Dict[int, str],
    ) -> str:
        """
        그룹 내 실패 서브태스크에 대해 재계획 수행.

        1단계: decomposition_callback이 있으면 실패+미수행 서브태스크 전체를 재분해
        2단계: 콜백이 없거나 실패하면, group_agent로 실패한 1개만 재계획 (fallback)

        성공한 서브태스크 데이터는 절대 수정하지 않음.
        반환: "replanned" | "dropped" | "failed"
        """
        # reset last metadata
        self.last_replaced_ids = [sid for sid in subtask_ids_in_group if isinstance(sid, int) and sid > 0]
        self.last_replanned_ids = []

        # 1단계: decomposition_callback으로 그룹 전체 재분해
        if self.decomposition_callback:
            new_plans = self.decomposition_callback(group_id, subtask_ids_in_group, context)
            if new_plans is not None:  # None=콜백 실패, {}=불가능한 목표 제거됨, {id:actions}=정상 replan
                if new_plans:
                    self.last_replanned_ids = sorted(
                        sid for sid in new_plans.keys() if isinstance(sid, int) and sid > 0
                    )
                    print(f"[ReplanGroup] decomposition_callback succeeded: {list(new_plans.keys())}")
                    return "replanned"
                else:
                    print(f"[ReplanGroup] decomposition_callback: all impossible goals dropped (no new subtasks needed)")
                    return "dropped"

        # 2단계: fallback - 실패한 서브태스크 1개만 재계획
        print(f"[ReplanGroup] Falling back to single subtask replan for subtask {context.failed_subtask_id}")
        failed_id = context.failed_subtask_id
        actions = current_actions_by_id.get(failed_id, [])
        problem_content = problem_content_by_id.get(failed_id, "")
        name = subtask_name_by_id.get(failed_id, f"subtask_{failed_id}")
        new_actions = self.group_agent.replan_subtask(
            subtask_id=failed_id,
            subtask_name=name,
            current_actions=actions,
            error_message=context.failure_reason,
            domain_content=domain_content,
            problem_content=problem_content,
            completed_actions=context.completed_actions,
        )
        if new_actions:
            self.last_replanned_ids = [failed_id]
            self.last_replaced_ids = [failed_id]
        return "replanned" if new_actions else "failed"


# ---------------------------------------------------------------------------
# Central LLM (전역 조정: full_replan vs subtask_only 결정)
# ---------------------------------------------------------------------------

class CentralLLM:
    """전역 조정 LLM. 서브태스크 간 의존성/자원이 걸린 실패 시 task-level 재계획 여부 결정."""

    def __init__(self, llm_handler: Any, gpt_version: str = "gpt-4o"):
        self.llm = llm_handler
        self.gpt_version = gpt_version

    def decide_replan_strategy(
        self,
        failed_subtask_ids: List[int],
        edges: List[Tuple[int, int]],
        context: str,
        max_tokens: int = 500,
    ) -> str:
        """반환: 'full_replan' 또는 'subtask_only'."""
        if not failed_subtask_ids:
            return "subtask_only"
        edges_str = ", ".join([f"{a}->{b}" for a, b in edges])
        prompt = f"""The following subtasks failed: {failed_subtask_ids}.
Subtask dependency edges (from_id -> to_id): {edges_str or "None"}.
Context: {context}
Answer with exactly one word: full_replan or subtask_only.
"""
        if "gpt" in self.gpt_version.lower():
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=max_tokens, frequency_penalty=0.0)
        else:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=max_tokens, stop=None, frequency_penalty=0.0)
        answer = (text or "").strip().lower()
        return "full_replan" if "full_replan" in answer else "subtask_only"


# ---------------------------------------------------------------------------
# DAG Loaders (기존 호환)
# ---------------------------------------------------------------------------

def load_subtask_dag_edges(base_path: str, task_name: str = "task") -> List[Tuple[int, int]]:
    """서브태스크 DAG JSON에서 의존성 간선 리스트 (from_id, to_id) 반환."""
    resources = os.path.join(base_path, "resources")
    dag_path = os.path.join(resources, "dag_outputs", f"{task_name}_SUBTASK_DAG.json")
    if not os.path.exists(dag_path):
        return []
    with open(dag_path, "r") as f:
        data = json.load(f)
    edges = data.get("edges", [])
    out: List[Tuple[int, int]] = []
    for e in edges:
        if "from_id" not in e or "to_id" not in e:
            continue
        try:
            frm = int(e["from_id"])
            to = int(e["to_id"])
        except (TypeError, ValueError):
            continue
        if frm <= 0 or to <= 0:
            continue
        out.append((frm, to))
    return out


def load_subtask_dag_parallel_groups(base_path: str, task_name: str = "task") -> Dict[int, List[int]]:
    """서브태스크 DAG JSON에서 parallel_groups 로드. group_id -> [subtask_ids]."""
    resources = os.path.join(base_path, "resources")
    dag_path = os.path.join(resources, "dag_outputs", f"{task_name}_SUBTASK_DAG.json")
    if not os.path.exists(dag_path):
        return {}
    with open(dag_path, "r") as f:
        data = json.load(f)
    raw = data.get("parallel_groups", {})
    if not isinstance(raw, dict):
        return {}
    cleaned: Dict[int, List[int]] = {}
    for k, v in raw.items():
        try:
            gid = int(k)
        except (TypeError, ValueError):
            continue
        sids: List[int] = []
        if isinstance(v, list):
            for sid in v:
                try:
                    sid_i = int(sid)
                except (TypeError, ValueError):
                    continue
                if sid_i > 0:
                    sids.append(sid_i)
        if sids:
            cleaned[gid] = sids
    return cleaned


def load_subtask_dag_effects(base_path: str, task_name: str = "task") -> Dict[int, List[str]]:
    """서브태스크 DAG JSON에서 노드별 effects 로드. subtask_id -> [effect strings]. 로컬 피드백 시 성공 effects 반영용."""
    resources = os.path.join(base_path, "resources")
    dag_path = os.path.join(resources, "dag_outputs", f"{task_name}_SUBTASK_DAG.json")
    if not os.path.exists(dag_path):
        return {}
    with open(dag_path, "r") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    out: Dict[int, List[str]] = {}
    for n in nodes:
        sid = n.get("id")
        if sid is None:
            continue
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        if sid <= 0:
            continue
        eff = n.get("effects")
        out[sid] = list(eff) if isinstance(eff, list) else []
    return out


def format_success_effects_for_prompt(
    success_effects: Dict[int, FrozenSet[str]],
    exclude_subtask_id: Optional[int] = None,
) -> str:
    """재계획 프롬프트에 넣을 '이미 성공한 서브태스크 effects' 문자열 생성. exclude_subtask_id는 재계획 대상이므로 제외."""
    lines = []
    for sid in sorted(success_effects.keys()):
        if exclude_subtask_id is not None and sid == exclude_subtask_id:
            continue
        eff = success_effects[sid]
        if not eff:
            continue
        lines.append(f"- Subtask {sid}: " + "; ".join(sorted(eff)))
    return "\n".join(lines) if lines else ""


def subtask_has_dependency(subtask_id: int, edges: List[Tuple[int, int]]) -> bool:
    """해당 서브태스크가 다른 서브태스크와 (선행/후행) 의존성이 있으면 True."""
    for a, b in edges:
        if a == subtask_id or b == subtask_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Backward compatibility: alias SubtaskManagerLLM -> GroupAgent
# ---------------------------------------------------------------------------

SubtaskManagerLLM = GroupAgent


# ---------------------------------------------------------------------------
# Sync execution results to store (한 번에 반영 또는 실시간 호출과 병행)
# ---------------------------------------------------------------------------

def sync_execution_results_to_store(
    store: SharedTaskStateStore,
    execution_results: Dict[int, Any],
    effects_by_subtask_id: Optional[Dict[int, List[str]]] = None,
) -> None:
    """
    실행 결과를 공유 스토어에 반영.
    execution_results: subtask_id -> { success: bool, error_message?: str } 또는
                       subtask_id -> SubTaskExecutionResult.
    effects_by_subtask_id: 성공한 서브태스크의 effects (선택). 없으면 성공 시 effects=None.
    실시간 반영이 아닌 경우, 실행 종료 후 한 번 호출해 스토어를 채울 수 있음.
    """
    effects_by_subtask_id = effects_by_subtask_id or {}
    for sid, r in execution_results.items():
        if hasattr(r, "success"):
            success = r.success
            err = getattr(r, "error_message", None)
        else:
            success = bool(r.get("success"))
            err = r.get("error_message")
        effects = effects_by_subtask_id.get(sid) if success else None
        store.update_subtask_on_completion(sid, success=success, error_message=err, effects=effects)


# ---------------------------------------------------------------------------
# Planner for one subtask (기존 호환)
# ---------------------------------------------------------------------------

def run_planner_for_one_subtask(
    base_path: str,
    file_processor: Any,
    extract_plan_actions_fn: Any,
    subtask_id: int,
    validated_subtask_path: str,
    subtask_pddl_plans_path: str,
    planner_timeout: int = 300,
) -> Tuple[bool, List[str]]:
    """단일 서브태스크에 대해 Fast-Downward 실행. 반환: (성공 여부, plan_actions)."""
    planner_path = os.path.join(base_path, "downward", "fast-downward.py")
    if not os.path.exists(planner_path):
        return False, []

    prefix = f"subtask_{subtask_id:02d}_"
    problem_file = None
    for f in os.listdir(validated_subtask_path):
        if f.startswith(prefix) and f.endswith(".pddl"):
            problem_file = f
            break
    if not problem_file:
        return False, []

    problem_file_full = os.path.join(validated_subtask_path, problem_file)
    domain_name = file_processor.extract_domain_name(problem_file_full)
    if not domain_name:
        return False, []
    domain_file = file_processor.find_domain_file(domain_name)
    if not domain_file:
        return False, []

    try:
        result = subprocess.run(
            [planner_path, "--alias", "seq-opt-lmcut", domain_file, problem_file_full],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=planner_timeout,
        )
        plan_actions = extract_plan_actions_fn(result.stdout)
        base_name = os.path.splitext(problem_file)[0]
        actions_path = os.path.join(subtask_pddl_plans_path, f"{base_name}_actions.txt")
        with open(actions_path, "w") as f:
            f.write("\n".join(plan_actions))
        return True, plan_actions
    except (subprocess.TimeoutExpired, Exception):
        return False, []
