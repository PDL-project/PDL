"""
피드백 루프: 실제 환경에서 작업 실패 시 replanning을 위한 계층 구조.

계층 정의:
  - Task      : 상위 작업 (e.g. "clean house")
  - Subtask   : 하위 작업/목표 (e.g. "wash dishes", "put apple in fridge") — action이 아님
  - Action    : 원시 동작 (e.g. "pickup dish", "gotoobject robot1 apple")
  각 Subtask는 여러 Action으로 이루어진 plan(액션 시퀀스)으로 수행됨.

구조:
  중앙 LLM (Central LLM)
    - Subtask 간 의존성, 자원, 할당 등 전역 조정
    - 의존성이 있는 서브태스크 실패 시 task-level replan
  |_ Subtask1 Manager LLM  → Subtask 1 전담, 의존성 없을 때 단일 서브태스크 replan
  |_ Subtask2 Manager LLM  → Subtask 2 전담
  |_ ...

규칙: 다른 Subtask와 의존성이 없으면 Subtask Manager LLM이 replan,
      의존성이 있으면 중앙 LLM이 replan.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# DAG 구조는 DAG_Module과 동일한 JSON 형식 사용
# task_SUBTASK_DAG.json: { "nodes": [...], "edges": [{"from_id": 1, "to_id": 2, ...}], "parallel_groups": {...} }


def load_subtask_dag_edges(base_path: str, task_name: str = "task") -> List[Tuple[int, int]]:
    """서브태스크 DAG JSON에서 의존성 간선 리스트 (from_id, to_id) 반환."""
    resources = os.path.join(base_path, "resources")
    dag_path = os.path.join(resources, "dag_outputs", f"{task_name}_SUBTASK_DAG.json")
    if not os.path.exists(dag_path):
        return []
    with open(dag_path, "r") as f:
        data = json.load(f)
    edges = data.get("edges", [])
    return [(int(e["from_id"]), int(e["to_id"])) for e in edges if "from_id" in e and "to_id" in e]


def subtask_has_dependency(subtask_id: int, edges: List[Tuple[int, int]]) -> bool:
    """해당 서브태스크가 다른 서브태스크와 (선행/후행) 의존성이 있으면 True."""
    for a, b in edges:
        if a == subtask_id or b == subtask_id:
            return True
    return False


class SubtaskManagerLLM:
    """
    서브태스크 하나를 담당하는 LLM. 해당 서브태스크 실패 시 해당 서브태스크만 재계획.
    (다른 Subtask와 의존성이 없을 때만 사용)
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
        max_tokens: int = 1500,
    ) -> List[str]:
        """
        실패한 서브태스크에 대해 LLM으로 대안 액션 시퀀스를 생성.
        반환: PDDL plan 형식과 동일한 액션 문자열 리스트 (예: ["gotoobject robot1 apple (1)", ...]).
        """
        actions_str = "\n".join(current_actions) if current_actions else "(none)"
        prompt = f"""You are a Subtask Manager. You manage one SUBTASK (a sub-goal like "wash dishes" or "put apple in fridge"), not single actions. The subtask is achieved by a sequence of primitive ACTIONS (e.g. gotoobject, pickupobject, putobject). Your subtask failed during execution. Propose a NEW sequence of primitive PDDL actions to achieve the SAME subtask goal, avoiding the cause of failure.

## Subtask
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
- Output ONLY one action per line, in the same format as the current plan (e.g. "gotoobject robot1 apple (1)", "pickupobject robot1 apple countertop (1)").
- Use the same domain actions and object names from the problem.
- Do not include any explanation, only the action lines.
- Number of lines = number of actions.
"""
        if "gpt" in self.gpt_version.lower():
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=max_tokens, frequency_penalty=0.0)
        else:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=max_tokens, stop=None, frequency_penalty=0.0)

        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        # PDDL plan 형태 라인만 필터 (액션명 + 인자 + (숫자))
        action_pattern = re.compile(r"^[a-zA-Z_]+\s+.*\s+\(\d+\)\s*$")
        actions = [ln for ln in lines if action_pattern.match(ln)]
        if not actions:
            actions = [ln for ln in lines if re.match(r"^[a-zA-Z_]+\s+", ln)]
        return actions


class CentralLLM:
    """
    전역 조정 LLM. 서브태스크 간 의존성/자원이 걸린 실패 시 task-level 재계획 결정 또는 조정.
    """

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
        """
        실패한 서브태스크와 의존성 정보를 보고 재계획 전략 결정.
        반환: "full_replan" (전체 task 재계획) 또는 "subtask_only" (이미 의존성으로 분기했으므로 여기선 full만 추가 사용 가능).
        """
        if not failed_subtask_ids:
            return "subtask_only"
        edges_str = ", ".join([f"{a}->{b}" for a, b in edges])
        prompt = f"""The following subtasks failed during execution: {failed_subtask_ids}.
Subtask-level dependency edges (from_id -> to_id): {edges_str or "None"}.

Context: {context}

Should we do a full task replan (re-decompose and re-plan all subtasks) or try to fix only the failed subtasks?
Answer with exactly one word: full_replan or subtask_only.
"""
        if "gpt" in self.gpt_version.lower():
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=max_tokens, frequency_penalty=0.0)
        else:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=max_tokens, stop=None, frequency_penalty=0.0)
        answer = (text or "").strip().lower()
        if "full_replan" in answer:
            return "full_replan"
        return "subtask_only"


def run_planner_for_one_subtask(
    base_path: str,
    file_processor: Any,
    extract_plan_actions_fn: Any,
    subtask_id: int,
    validated_subtask_path: str,
    subtask_pddl_plans_path: str,
    planner_timeout: int = 300,
) -> Tuple[bool, List[str]]:
    """
    단일 서브태스크에 대해서만 Fast-Downward 실행.
    subtask_id에 해당하는 problem 파일을 찾아서 플래너 실행 후 _actions.txt 저장.
    반환: (성공 여부, plan_actions 리스트)
    """
    planner_path = os.path.join(base_path, "downward", "fast-downward.py")
    if not os.path.exists(planner_path):
        return False, []

    # subtask_01_xxx.pddl 형태 파일 찾기
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