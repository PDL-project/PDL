#!/usr/bin/env python3
"""
MultiRobotExecutor - Run multi-robot plan in AI2-THOR directly.
Based on the working ai2_thor_controller.py and aithor_connect.py patterns.
"""

import json
import math
import os
import re
import threading
import time
import random
import shutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from glob import glob
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from AI2Thor.Tasks.get_scene_init import get_scene_initializer

# task_mapper.py 딕셔너리 (torch 없이 자연어→폴더명 변환)
_TASK_NAME_MAP = {
    "Put the bread, lettuce, and tomato in the fridge": "1_put_bread_lettuce_tomato_fridge",
    "Put the computer, book, and remotecontrol on the sofa": "1_put_computer_book_remotecontrol_sofa",
    "Put the butter knife, bowl, and mug on the countertop": "1_put_knife_bowl_mug_countertop",
    "Put the plate, mug, and bowl in the fridge": "1_put_plate_mug_bowl_fridge",
    "Put the remotecontrol, keys, and watch in the box": "1_put_remotecontrol_keys_watch_box",
    "Put the vase, tissue box, and remote control on the table": "1_put_vase_tissuebox_remotecontrol_table",
    "Slice the bread, lettuce, tomato, and egg": "1_slice_bread_lettuce_tomato_egg",
    "Turn off the faucet and light if either is on": "1_turn_off_faucet_light",
    "Wash the bowl, mug, pot, and pan": "1_wash_bowl_mug_pot_pan",
    "Open all the drawers": "2_open_all_drawers",
    "Open all the cabinets": "2_open_all_cabinets",
    "Turn on all the stove knobs": "2_turn_on_all_stove_knobs",
    "Put all the vases on the countertop": "2_put_all_vases_countertop",
    "Put all the tomatoes and potatoes in the fridge": "2_put_all_tomatoes_potatoes_fridge",
    "Put all credit cards and remote controls in the box": "2_put_all_creditcards_remotecontrols_box",
    "Put all groceries in the fridge": "3_put_all_groceries_fridge",
    "Put all shakers in the fridge": "3_put_all_shakers_fridge",
    "Put all silverware in any drawer": "3_put_all_silverware_drawer",
    "Put all school supplies on the sofa": "3_put_all_school_supplies_sofa",
    "Move everything on the table to the sofa": "3_clear_table_to_sofa",
    "Put all kitchenware in the cardboard box": "3_put_all_kitchenware_box",
    "Clear the table by placing items at their appropriate positions": "4_clear_table_kitchen",
    "Clear the kitchen central countertop by placing items in their appropriate positions": "4_clear_countertop_kitchen",
    "Clear the couch by placing the items in other appropriate positions": "4_clear_couch_livingroom",
    "Make the living room dark": "4_make_livingroom_dark",
    "Slice all sliceable objects": "4_slice_all_sliceable",
    "Put appropriate utensils in storage": "4_put_appropriate_storage",
}
_TASK_NAME_MAP_LOWER = {k.lower(): v for k, v in _TASK_NAME_MAP.items()}


# -----------------------------
# 각 서브태스크의 ID, 이름, 담당 로봇 ID, 수행할 액션 리스트, 병렬 그룹 번호를 저장하는 데이터 구조
# -----------------------------
@dataclass
class SubtaskPlan:
    subtask_id: int
    subtask_name: str
    robot_id: int               # 1-based
    actions: List[str]
    parallel_group: int = 0

@dataclass
class YieldRequest: # 로봇 간 경로 양보(Yield) 요청 정보 저장
    requester_id: int # 길을 비켜달라고 요청한 로봇 ID
    timestamp: float # 요청 발생 시각
    reason: str # 요청 사유
    target_object: str # 이동 중 목표 객체
    attempts: int = 0 # 재시도 횟수
    last_distance: float = 0.0
    next_time: float = 0.0

@dataclass
class SubTaskExecutionResult:
    """서브태스크 단위 실행 결과 (피드백 루프용)"""
    subtask_id: int # 실행한 서브태스크 ID
    success: bool # 성공 여부
    error_message: Optional[str] = None # 실패 시 에러 메시지
    completed_actions: Optional[List[str]] = None # 실패 전까지 성공한 액션 목록

# -----------------------------
# 이동
# -----------------------------
def closest_node(node, nodes, no_robot, clost_node_location):
    """
    로봇이 특정 목적지까지 갈 때, 이동 가능한(Reachable) 접근 지점을 선택하는 함수.

    사분면 기반 선택: offset이 증가하면 목표물 주변의 다른 방향에서 접근을 시도.
      offset 0 = 전체 최근접
      offset 1 = 사분면 1 (북동) 최근접
      offset 2 = 사분면 2 (남동) 최근접
      offset 3 = 사분면 3 (남서) 최근접
      offset 4 = 사분면 4 (북서) 최근접
      offset 5+ = 각 사분면 내 2번째, 3번째... 최근접
    """
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))

    for i in range(no_robot):
        offset = clost_node_location[i]

        if offset == 0:
            # offset 0: 기존 동작 (전체 최근접)
            crps.append(nodes[dist_indices[0]])
        else:
            # 사분면 기반 선택
            target_x, target_z = node[0], node[2]
            quadrant = (offset - 1) % 4       # 0~3: 4개 사분면
            rank_in_quad = (offset - 1) // 4   # 사분면 내 몇 번째 점인지

            found = None
            quad_count = 0
            for idx in dist_indices:
                pt = nodes[idx]
                dx = pt[0] - target_x
                dz = pt[2] - target_z
                # 사분면 판별 (목표물 기준)
                if quadrant == 0 and dx >= 0 and dz >= 0:
                    pass  # 북동
                elif quadrant == 1 and dx >= 0 and dz < 0:
                    pass  # 남동
                elif quadrant == 2 and dx < 0 and dz < 0:
                    pass  # 남서
                elif quadrant == 3 and dx < 0 and dz >= 0:
                    pass  # 북서
                else:
                    continue
                if quad_count == rank_in_quad:
                    found = pt
                    break
                quad_count += 1

            if found is not None:
                crps.append(found)
            else:
                # 해당 사분면에 점이 없으면 전체에서 offset번째
                fallback_idx = min(offset, len(dist_indices) - 1)
                crps.append(nodes[dist_indices[fallback_idx]])

    return crps


def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    """두 지점 사이의 2차원 평면(x, z) 거리를 계산하는 함수"""
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

# -----------------------------
# 실행기
# -----------------------------
class MultiRobotExecutor:
    """
    다중 로봇 환경에서 PDDL 기반 서브태스크 계획을 실행하는 핵심 실행기 클래스

    주요 역할:
    - DAG 기반 병렬 서브태스크 실행 관리
    - 각 로봇의 Action Queue 생성 및 Round-Robin 방식 실행
    - 로봇 간 충돌 감지 및 회피(Yield Mechanism)
    - Navigation 실패(Oscillation, Stuck) 감지 및 Recovery 수행
    - Checker 기반 Subtask 성공률 평가
    """
    def __init__(self, base_path: str):
        self.base_path = base_path

        # 리소스 디렉토리 (DAG 결과 및 PDDL Plan 저장 위치)
        self.resources_path = os.path.join(base_path, "resources")
        self.dag_output_path = os.path.join(self.resources_path, "dag_outputs")
        self.plans_path = os.path.join(self.resources_path, "subtask_pddl_plans")

        # 각 서브태스크를 어떤 로봇이 수행할지에 대한 할당 정보
        self.assignment: Dict[int, int] = {}

        # DAG 기반 병렬 실행 그룹 (Group ID → Subtask ID 리스트)
        self.parallel_groups: Dict[int, List[int]] = {}

        # 각 서브태스크에 대한 실행 계획 저장
        self.subtask_plans: Dict[int, SubtaskPlan] = {}

        # AI2-THOR 시뮬레이터 관련 변수
        self.controller: Optional[Controller] = None
        self.no_robot = 1 # 로봇 수
        self.reachable_positions: List[Tuple[float, float, float]] = [] # 이동 가능 위치 리스트 (tuple)
        self.reachable_positions_: List[dict] = [] # 이동 가능 위치 리스트 (dict)

        # Action Queue 및 실행 스케줄링
        self.action_queues: List[deque] = [] # 각 로봇별로 실행할 액션을 저장하는 Queue
        self.action_lock = threading.Lock() # Queue 접근 시 race condition 방지를 위한 Lock
        self.rr_index = 0 # Round-Robin 방식 실행을 위한 인덱스
        self.task_over = False # 전체 Task 종료 여부
        self.actions_thread: Optional[threading.Thread] = None # Action 실행을 담당하는 백그라운드 스레드

        # 실행
        self.total_exec = 0 # 총 실행된 액션 수
        self.success_exec = 0 # 성공적으로 수행된 액션 수
        self.agent_success_counts: List[int] = []  # balance metric용 agent별 성공 액션 수

        # 피드백 루프: 서브태스크별 실패 추적 (subtask_id -> True if any action failed)
        self._thread_subtask_id: Dict[int, int] = {}
        self._subtask_failed: Dict[int, bool] = {}
        self._subtask_last_error: Dict[int, str] = {}
        self._subtask_completed_actions: Dict[int, List[str]] = {}  # 서브태스크별 성공한 액션 목록
        self._subtask_results: Dict[int, SubTaskExecutionResult] = {}
        # 물리적으로 불가능하여 drop된 서브태스크 (더 이상 실행하지 않음)
        self._dropped_subtasks: set = set()
        # 실시간 상태 반영용 (execute_in_ai2thor_with_feedback 호출 시에만 설정)
        self._feedback_state_store: Optional[Any] = None

        # Checker (Execution 평가 모듈)
        self.checker = None # Scene 내 Object를 사람이 읽을 수 있는 형태로 변환하기 위한 Dictionary
        self.object_dict: Dict[str, Dict[str, int]] = {} # 각 로봇의 현재 인벤토리 상태 저장
        self.inventory: List[str] = []
        self.scene_name: Optional[str] = None # 현재 실행 중인 Scene 이름

        # Yield 기반 충돌 회피 관련 변수
        self.bb_lock = threading.Lock()
        self.yield_requests: Dict[int, YieldRequest] = {}  # 현재 blocking 상태인 로봇에게 보내진 Yield 요청
        self.monitor_thread: Optional[threading.Thread] = None # Yield 요청을 모니터링하는 스레드
        self.yield_cooldown_s = 3.0 # 동일한 Yield 요청 재발생 방지를 위한 쿨다운 시간
        self.last_yield_request: Dict[Tuple[int, int], float] = {}  # (blocking_robot, requester_robot) 쌍별 마지막 요청 시간 기록
        self.yield_clear_distance = 1.5 # Yield 이후 확보되어야 하는 최소 거리
        self.yield_margin = 0.3
        self.yield_retry_delay_s = 0.5
        # yield 재시도 누적 시 blocker 강제 unstick (MoveBack+Rotate)
        self.yield_force_unstick_attempts = 4
        self.yield_force_unstick_cooldown_s = 1.0
        self._last_forced_unstick: Dict[int, float] = {}

        # Action 완료 여부 동기화
        self.action_cv = threading.Condition() # 특정 로봇의 Action 완료를 기다리기 위한 Condition Variable
        self.agent_action_counters: List[int] = [] # 각 로봇이 몇 개의 액션을 수행했는지 기록
        # _enqueue_and_wait가 "내가 넣은 액션" 완료를 정확히 기다리기 위한 토큰 상태
        self._next_wait_token: int = 1
        self._wait_token_result: Dict[int, bool] = {}

        # Receptacle cache (로봇ID, 객체패턴) → 실제 objectId 매핑
        self.receptacle_cache: Dict[Tuple[int, str], str] = {}

        # NAV rotation-only detection
        self.nav_rotation_only_count: List[int] = []
        self.nav_rotation_only_threshold = 3

        # [Navigation 실패 감지 관련 변수]
        # 회전만 반복하는 Deadlock 상태 감지용 카운터
        self.nav_position_history_size = 12  # 최근 N개 위치 저장 (실제 이동한 위치만)
        self.nav_oscillation_radius = 0.25   # 이 반경 이내면 "같은 위치"로 판단
        
        # Oscillation (앞뒤 진동) 감지 관련 파라미터
        self.nav_oscillation_threshold = 3   # N회 재방문 감지 시 oscillation으로 판정
        self.nav_oscillation_move_thresh = 0.15  # 이 거리 이상 움직여야 히스토리에 기록
        self.nav_oscillation_cooldown_iters = 10  # recovery 후 N회 반복동안 감지 건너뜀

    # -----------------------------
    # 피드백 관련 함수
    # -----------------------------
    def _get_current_subtask_id(self) -> Optional[int]:
        """피드백 모드에서 현재 스레드가 실행 중인 서브태스크 ID (액션 큐에 태깅용)."""
        return self._thread_subtask_id.get(threading.get_ident())

    def _fail_current_subtask(self, error_msg: str):
        """현재 스레드의 서브태스크를 실패로 표시. 고수준 액션 함수에서 실패 전파 용도."""
        sid = self._get_current_subtask_id()
        if sid is not None and not self._subtask_failed.get(sid, False):
            self._subtask_failed[sid] = True
            self._subtask_last_error[sid] = error_msg

    def _record_agent_success(self, agent_id: int):
        """agent_id의 성공 액션 카운트 +1 (balance metric용)."""
        if 0 <= agent_id < len(self.agent_success_counts):
            self.agent_success_counts[agent_id] += 1

    def _compute_balance_metric(self) -> float:
        """SmartLLM식 balance = min(x_i) / max(x_i), x_i=agent i의 성공 액션 수.
        모든 서브태스크가 성공한 경우에만 계산, 그렇지 않으면 0."""
        if self._subtask_results and any(
            not r.success for r in self._subtask_results.values()
        ):
            return 0.0
        active = [c for c in self.agent_success_counts if c > 0]
        if not active:
            return 0.0
        mx = max(active)
        if mx == 0:
            return 0.0
        return min(active) / mx

    def _compute_exec_rate(self) -> float:
        """Executability(Exec) = successful low-level actions / attempted actions."""
        if self.total_exec <= 0:
            return 0.0
        return float(self.success_exec) / float(self.total_exec)

    # -----------------------------
    # Checker helpers
    # -----------------------------
    def _parse_object(self, object_str: str):
        """
        AI2-THOR에서 제공하는 objectId를
        [객체 이름]과 [고유 ID]로 분리하는 함수.

        예:
        "Apple|+00.12|+01.22" →
        obj_name = Apple
        obj_id   = |+00.12|+01.22
        """
        obj_name = object_str.split("|")[0]
        obj_id = object_str.replace(obj_name, "")
        return obj_name, obj_id

    def _build_object_dict(self):
        """
        Scene 내 존재하는 모든 객체들을 순회하며
        사람이 읽을 수 있는 형식의 ID 매핑 딕셔너리 생성.

        예:
        Apple|123 → Apple_1
        Apple|456 → Apple_2
        """
        self.object_dict = {}
        for obj in self.controller.last_event.metadata["objects"]:
            obj_name, obj_id = self._parse_object(obj["objectId"])
            if obj_name not in self.object_dict:
                self.object_dict[obj_name] = {}
            if obj_id not in self.object_dict[obj_name]:
                self.object_dict[obj_name][obj_id] = len(self.object_dict[obj_name]) + 1

    def _convert_object_id_to_readable(self, object_id: str) -> str:
        """
        AI2-THOR에서 사용하는 내부 objectId를 사람이 읽을 수 있는 형태로 변환하는 함수.

        예:
        "Apple|+00.12|+01.22" → "Apple_1"
        """
        if not object_id:
            return "unknown"
        if "|" not in object_id:
            # 단순 객체 이름이 들어온 경우 실제 objectId 탐색
            resolved = self._find_object_id(object_id)
            if not resolved:
                return object_id
            object_id = resolved
        obj_name, obj_id = self._parse_object(object_id)
        if obj_name not in self.object_dict or obj_id not in self.object_dict[obj_name]:
            return obj_name
        obj_num = self.object_dict[obj_name][obj_id]
        return f"{obj_name}_{obj_num}"

    def _update_inventory(self, agent_id: int):
        """
        특정 로봇(agent)의 현재 인벤토리 상태를 갱신하는 함수.
        """
        inv = self.controller.last_event.events[agent_id].metadata.get("inventoryObjects", [])
        if not inv:
            self.inventory[agent_id] = "nothing"
            return
        # 여러 개 들고 있을 경우 첫 번째 객체만 사용
        self.inventory[agent_id] = self._convert_object_id_to_readable(inv[0]["objectId"])

    def _checker_report(self, agent_id: int, action_name: str, obj_readable: str, success: bool):
        """checker에 액션을 리포팅하는 헬퍼. checker가 없으면 무시."""
        if self.checker is None:
            return
        self._update_inventory(agent_id)
        action_str = f"{action_name}({obj_readable})"
        self.checker.perform_metric_check(action_str, success, self.inventory[agent_id])

    def _init_checker(self, task_description: str, scene_name: str):
        """
        Task 수행 결과를 평가하기 위한 Checker 모듈 초기화 함수.

        동작 과정:
        1. Task Description과 Scene 정보를 기반으로 SceneInitializer 및 Checker 로드
        2. Scene 내 모든 objectId를 Checker에 전달하여 평가 범위 설정
        3. 초기 Scene 상태를 Task 수행 전 상태로 설정

        Checker는 각 Action 수행 이후
        Task 성공 여부(Coverage, Transport Rate 등)를
        온라인으로 평가하는 역할을 수행함.
        """
        # 자연어 task description을 폴더명으로 변환 (task_mapper.py 딕셔너리와 동일, torch 불필요)
        task_folder = _TASK_NAME_MAP.get(task_description) \
                      or _TASK_NAME_MAP_LOWER.get(task_description.lower().strip()) \
                      or task_description
        print(f"[Checker] Mapped '{task_description}' -> '{task_folder}'")
        scene_initializer, checker_mod = get_scene_initializer(task_folder, scene_name)
        self.checker = checker_mod.Checker()
        # 현재 Scene 내 모든 객체를 Checker에 전달
        all_oids = [obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]
        if hasattr(self.checker, "all_objects"):
            self.checker.all_objects(obj_ids=all_oids, scene=scene_name)
        #print("_" * 50)
        #print("Subtasks to complete:")
        #try:
         #   print("\n".join(self.checker.subtasks))
        #except Exception:
        #    pass
        # Scene 초기 상태 설정
        if scene_initializer is not None:
            self.controller.last_event = scene_initializer.SceneInitializer().preinit(
                self.controller.last_event, self.controller
            )

    def _enqueue_front(self, actions: List[dict]):
        # 여러 개의 Action을 해당 로봇의 Action Queue 맨 앞에 삽입하는 함수
        with self.action_lock:
            for action in reversed(actions):
                self._enqueue_action_locked(action, front=True)

    def _enqueue_action_locked(self, action: dict, front: bool = False):
        #Lock이 걸린 상태에서 Action Queue에 Action을 삽입하는 내부 함수
        agent_id = int(action.get("agent_id", 0))
        if not self.action_queues:
            return
        if agent_id < 0 or agent_id >= len(self.action_queues):
            agent_id = 0
        if front:
            self.action_queues[agent_id].appendleft(action)
        else:
            self.action_queues[agent_id].append(action)

    def _enqueue_action(self, action: dict, front: bool = False):
        """
        Action Queue에 액션을 삽입하는 단일 엔트리 함수.
        - race condition 방지를 위해 lock 사용
        - 피드백 루프용 subtask_id 자동 태깅
        - front=True면 해당 로봇 큐의 앞쪽에 삽입 (recovery 등 우선 실행)
        """
        # 원본 dict 보호
        action = dict(action)

        # 피드백 루프: 현재 스레드의 subtask_id를 자동으로 붙임
        if "subtask_id" not in action:
            sid = self._get_current_subtask_id()
            if sid is not None:
                action["subtask_id"] = sid

        with self.action_lock:
            self._enqueue_action_locked(action, front=front)

    def _queue_total_len(self) -> int:
        # 모든 로봇의 Action Queue에 남아있는 전체 Action 개수를 반환하는 함수
        with self.action_lock:
            return sum(len(q) for q in self.action_queues)

    def _drain_action_queue(self, timeout: float = 15.0) -> None:
        """액션 큐가 빌 때까지 대기. timeout 초 초과 시 큐를 강제 비우고 반환.
        yield 모니터가 큐를 계속 채울 경우 무한 대기를 방지한다."""
        deadline = time.time() + timeout
        while self._queue_total_len() > 0:
            if time.time() > deadline:
                with self.action_lock:
                    for q in self.action_queues:
                        q.clear()
                break
            time.sleep(0.3)

    def _dequeue_action(self) -> Optional[dict]:
        # Round-Robin 방식으로 각 로봇의 Action Queue에서 다음 실행할 Action을 하나 꺼내는 함수
        with self.action_lock:
            if not self.action_queues:
                return None
            n = len(self.action_queues)
            for i in range(n):
                idx = (self.rr_index + i) % n
                if self.action_queues[idx]:
                    act = self.action_queues[idx].popleft()
                    self.rr_index = (idx + 1) % n
                    return act
        return None

    def _enqueue_and_wait(self, action: dict, agent_id: int, timeout: float = 10.0) -> bool:
        """특정 로봇의 Action을 Queue에 삽입한 후, 해당 Action이 실제로 실행될 때까지 대기하는 함수"""
        with self.action_cv:
            if not self.agent_action_counters:
                self._enqueue_action(action)
                return True
            token = self._next_wait_token
            self._next_wait_token += 1
            action = dict(action)
            action["wait_token"] = token
            self._enqueue_action(action)
            end_time = time.time() + timeout
            while token not in self._wait_token_result:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return False
                self.action_cv.wait(timeout=remaining)
            return bool(self._wait_token_result.pop(token, False))

    # -----------------------------
    # 데이터 로드
    # -----------------------------
    def load_assignment(self, task_idx: int = 0) -> Dict[int, int]:
        """"어떤 서브태스크를 어떤 로봇이 맡을지" 기록된 JSON 파일 읽어오는 함수"""
        assignment_file = os.path.join(self.dag_output_path, f"task_{task_idx}_assignment.json")
        if not os.path.exists(assignment_file):
            raise FileNotFoundError(f"Assignment file not found: {assignment_file}")

        with open(assignment_file, "r") as f:
            data = json.load(f)

        self.assignment = {int(k): int(v) for k, v in data.get("assignment", {}).items()}

        # 전체 에이전트 수 로드 (할당 안 된 로봇도 소환하기 위해)
        self.configured_agent_count = data.get("agent_count", None)

        # LP에서 결정된 스폰 좌표 로드 (실행 시 동일 위치 재배치용)
        raw_spawn = data.get("robot_spawn_positions", None)
        if raw_spawn:
            self.saved_spawn_positions = {int(k): tuple(v) for k, v in raw_spawn.items()}
        else:
            self.saved_spawn_positions = None

        #print(f"[Executor] Loaded assignment: {self.assignment}")
        #if self.configured_agent_count:
            #print(f"[Executor] Configured agent count: {self.configured_agent_count}")
        return self.assignment

    def load_subtask_dag(self, task_name: str = "task") -> Dict[int, List[int]]:
        """"어떤 서브태스크들이 동시에 실행 가능한지(Parallel Groups)" 기록된 DAG 결과 파일 읽어오는 함수"""
        dag_file = os.path.join(self.dag_output_path, f"{task_name}_SUBTASK_DAG.json")
        if not os.path.exists(dag_file):
            raise FileNotFoundError(f"Subtask DAG file not found: {dag_file}")

        with open(dag_file, "r") as f:
            data = json.load(f)

        raw_pg = data.get("parallel_groups", {})
        cleaned: Dict[int, List[int]] = {}
        for k, v in raw_pg.items():
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
        self.parallel_groups = cleaned
        #print(f"[Executor] Loaded parallel groups: {self.parallel_groups}")
        return self.parallel_groups

    def load_plan_actions(self) -> Dict[int, List[str]]:
        """각 서브태스크별로 수행해야 할 구체적인 PDDL 액션 리스트(_actions.txt)를 읽어와 SubtaskPlan 객체로 만드는 함수"""
        if not os.path.exists(self.plans_path):
            raise FileNotFoundError(f"Plans directory not found: {self.plans_path}")

        plan_files = [f for f in os.listdir(self.plans_path) if f.endswith("_actions.txt")]
        plan_actions: Dict[int, List[str]] = {}

        for plan_file in sorted(plan_files):
            m = re.search(r"subtask_(\d+)", plan_file)
            if not m:
                continue
            subtask_id = int(m.group(1))

            plan_path = os.path.join(self.plans_path, plan_file)
            with open(plan_path, "r") as f:
                actions = [ln.strip() for ln in f.readlines() if ln.strip()]

            plan_actions[subtask_id] = actions
            subtask_name = plan_file.replace("_actions.txt", "")

            robot_id = int(self.assignment.get(subtask_id, 1))
            pg = 0
            for gid, sids in self.parallel_groups.items():
                if subtask_id in sids:
                    pg = gid
                    break

            self.subtask_plans[subtask_id] = SubtaskPlan(
                subtask_id=subtask_id,
                subtask_name=subtask_name,
                robot_id=robot_id,
                actions=actions,
                parallel_group=pg,
            )

        #print(f"[Executor] Loaded {len(plan_actions)} subtask plans")
        return plan_actions

    # -----------------------------
    # Action Queue Executor 액션 실행기
    # -----------------------------
    def _exec_actions(self):
        """백그라운드에서 action_queue에 쌓인 명령들을 하나씩 꺼내 시뮬레이터(controller.step)에 전달하고, 화면(OpenCV 창)을 갱신"""
        c = self.controller
        img_counter = 0

        while not self.task_over:
            act = self._dequeue_action()
            if act is not None:
                try:
                    multi_agent_event = None
                    action_success = True

                    if act['action'] == 'ObjectNavExpertAction':
                        multi_agent_event = c.step(dict(
                            action=act['action'],
                            position=act['position'],
                            agentId=act['agent_id']
                        ))
                        # actionReturn을 두 곳에서 모두 확인 (AI2Thor 버전 호환성)
                        aid = act['agent_id']
                        next_action = None
                        # 1) 해당 에이전트의 per-agent metadata
                        try:
                            next_action = multi_agent_event.events[aid].metadata.get('actionReturn')
                        except Exception:
                            pass
                        # 2) fallback: 글로벌 metadata (단일 에이전트 호환)
                        if next_action is None:
                            next_action = multi_agent_event.metadata.get('actionReturn')

                        # 디버그: 처음 5번만 로그
                        if img_counter < 5:
                            success = multi_agent_event.events[aid].metadata.get('lastActionSuccess', '?')
                            err = multi_agent_event.events[aid].metadata.get('errorMessage', '')
                            #print(f"[NAV DEBUG] agent={aid}, actionReturn={next_action}, success={success}, err={err}")

                        # (A) actionReturn이 문자열이면 그대로 action으로 실행
                        if isinstance(next_action, str) and next_action:
                            multi_agent_event = c.step(
                                action=next_action,
                                agentId=aid,
                            )
                        # (B) actionReturn이 dict면 파라미터 포함하여 실행
                        elif isinstance(next_action, dict) and next_action.get("action"):
                            cmd = dict(next_action)
                            cmd["agentId"] = aid
                            multi_agent_event = c.step(cmd)
                        # (C) None이면 아무것도 안 함

                        # blocking 감지: errorMessage에 "blocking"이 있으면 회피 유도
                        try:
                            err = multi_agent_event.events[aid].metadata.get("errorMessage", "") or ""
                            if "blocking" in err.lower():
                                self._enqueue_action({
                                    'action': 'MoveBack',
                                    'agent_id': aid
                                }, front=True)
                        except Exception:
                            pass

                        # rotation-only detection (문자열 + dict 모두 처리)
                        try:
                            act_name = None
                            if isinstance(next_action, str):
                                act_name = next_action
                            elif isinstance(next_action, dict):
                                act_name = next_action.get("action")

                            if act_name in ("RotateLeft", "RotateRight"):
                                if aid < len(self.nav_rotation_only_count):
                                    self.nav_rotation_only_count[aid] += 1
                            else:
                                if aid < len(self.nav_rotation_only_count):
                                    self.nav_rotation_only_count[aid] = 0
                        except Exception:
                            pass

                    elif act['action'] == 'Teleport':
                        multi_agent_event = c.step(dict(
                            action="Teleport",
                            position=act['position'],
                            agentId=act['agent_id'],
                            forceAction=True
                        ))

                    elif act['action'] == 'MoveAhead':
                        multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])

                    elif act['action'] == 'MoveBack':
                        multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])

                    elif act['action'] == 'RotateLeft':
                        multi_agent_event = c.step(
                            action="RotateLeft",
                            degrees=act['degrees'],
                            agentId=act['agent_id']
                        )

                    elif act['action'] == 'RotateRight':
                        multi_agent_event = c.step(
                            action="RotateRight",
                            degrees=act['degrees'],
                            agentId=act['agent_id']
                        )

                    elif act['action'] == 'PickupObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="PickupObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[PickupObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'PutObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="PutObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        err_msg = multi_agent_event.metadata.get('errorMessage', "")
                        if err_msg != "":
                            action_success = False
                            print(f"[PutObject] Error: {err_msg}")
                            # Auto-recovery: if receptacle closed, open then retry once
                            if "CLOSED" in err_msg.upper():
                                retry = act.get("retry", 0)
                                if retry < 1:
                                    self._enqueue_action({
                                        'action': 'OpenObject',
                                        'objectId': act['objectId'],
                                        'agent_id': act['agent_id']
                                    }, front=True)
                                    new_act = dict(act)
                                    new_act["retry"] = retry + 1
                                    self._enqueue_action(new_act, front=True)
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'ToggleObjectOn':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ToggleObjectOn",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[ToggleObjectOn] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'ToggleObjectOff':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ToggleObjectOff",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[ToggleObjectOff] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'OpenObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="OpenObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[OpenObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'CloseObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="CloseObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[CloseObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'SliceObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="SliceObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[SliceObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'ThrowObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="ThrowObject",
                            moveMagnitude=7,
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[ThrowObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'BreakObject':
                        self.total_exec += 1
                        multi_agent_event = c.step(
                            action="BreakObject",
                            objectId=act['objectId'],
                            agentId=act['agent_id'],
                            forceAction=True
                        )
                        if multi_agent_event.metadata['errorMessage'] != "":
                            action_success = False
                            print(f"[BreakObject] Error: {multi_agent_event.metadata['errorMessage']}")
                        else:
                            self.success_exec += 1
                            self._record_agent_success(act['agent_id'])

                    elif act['action'] == 'Done':
                        multi_agent_event = c.step(action="Done")
                    
                    # 피드백 루프: 서브태스크별 실패 기록
                    # 네비게이션 액션(ObjectNavExpertAction, MoveBack, MoveAhead, Rotate*)의 에러는 무시
                    # 실제 목표 액션(Open/Close/Pickup/Put/Toggle/Slice 등)의 실패만 기록
                    _NAV_ACTIONS = {"ObjectNavExpertAction", "MoveBack", "MoveAhead",
                                    "RotateLeft", "RotateRight", "LookUp", "LookDown", "Teleport"}
                    if (multi_agent_event is not None
                            and multi_agent_event.metadata.get("errorMessage")
                            and act.get("action") not in _NAV_ACTIONS):
                        action_success = False
                        sid = act.get("subtask_id")
                        if sid is not None:
                            self._subtask_failed[sid] = True
                            self._subtask_last_error[sid] = multi_agent_event.metadata.get("errorMessage", "")

                except Exception as e:
                    action_success = False
                    print(f"[ExecActions] Exception: {e}")
                    sid = act.get("subtask_id")
                    if sid is not None:
                        self._subtask_failed[sid] = True
                        self._subtask_last_error[sid] = str(e)

                #if img_counter % 50 == 0:
                    #print(f"[ExecActions] processed={img_counter}, queue={self._queue_total_len()}")

                # 화면 뷰
                if multi_agent_event is not None:
                    try:
                        for i, e in enumerate(multi_agent_event.events):
                            cv2.imshow(f'Robot {i+1}', e.cv2img)
                        # 탑뷰
                        if c.last_event.events[0].third_party_camera_frames:
                            top_view_rgb = cv2.cvtColor(
                                c.last_event.events[0].third_party_camera_frames[-1],
                                cv2.COLOR_BGR2RGB
                            )
                            cv2.imshow('Top View', top_view_rgb)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        pass

                img_counter += 1
                try:
                    agent_id = act.get("agent_id", None)
                    if agent_id is not None:
                        with self.action_cv:
                            if agent_id < len(self.agent_action_counters):
                                self.agent_action_counters[agent_id] += 1
                            token = act.get("wait_token")
                            if token is not None:
                                self._wait_token_result[token] = bool(action_success)
                            self.action_cv.notify_all()
                except Exception:
                    pass
            else:
                time.sleep(0.05)

    # -----------------------------
    # High-level Actions(로봇 동작 제어)
    # -----------------------------
    def _object_matches_pattern(self, obj: dict, pattern: str) -> bool:
        """
        Pattern matcher shared by all object-resolution paths.
        Supports:
        - exact objectId / objectType match (case-insensitive)
        - numbered name patterns: "drawer1" → 1st Drawer in object_dict
        - regex compatibility with legacy plans
        """
        object_id = str(obj.get("objectId", ""))
        object_type = str(obj.get("objectType", ""))
        p = (pattern or "").strip()
        if not p:
            return False

        p_low = p.lower()
        if object_id.lower() == p_low or object_type.lower() == p_low:
            return True

        # numbered name pattern: e.g. "drawer1" → type="Drawer", num=1
        m = re.match(r'^([a-zA-Z]+?)(\d+)$', p)
        if m and hasattr(self, 'object_dict'):
            base_type = m.group(1).capitalize()
            target_num = int(m.group(2))
            if base_type in self.object_dict:
                # object_dict[base_type] = { obj_id_suffix: number, ... }
                obj_name, obj_id_suffix = self._parse_object(object_id)
                if obj_name == base_type and obj_id_suffix in self.object_dict[base_type]:
                    if self.object_dict[base_type][obj_id_suffix] == target_num:
                        return True

        # fallback: "drawer (1)" style from PDDL action strings
        m2 = re.match(r'^([a-zA-Z]+)\s*\((\d+)\)$', p)
        if m2 and hasattr(self, 'object_dict'):
            base_type = m2.group(1).capitalize()
            target_num = int(m2.group(2))
            if base_type in self.object_dict:
                obj_name, obj_id_suffix = self._parse_object(object_id)
                if obj_name == base_type and obj_id_suffix in self.object_dict[base_type]:
                    if self.object_dict[base_type][obj_id_suffix] == target_num:
                        return True

        # fallback for legacy regex-style patterns
        try:
            return bool(
                re.match(p, object_id, re.IGNORECASE)
                or re.match(p, object_type, re.IGNORECASE)
            )
        except re.error:
            return False

    def _distance_agent_to_obj(self, agent_id: int, obj: dict) -> float:
        agent_meta = self.controller.last_event.events[agent_id].metadata
        agent_pos = agent_meta["agent"]["position"]
        obj_pos = obj.get("position", {})
        ox = obj_pos.get("x")
        oz = obj_pos.get("z")
        if ox is None or oz is None:
            return float("inf")
        return ((agent_pos["x"] - ox) ** 2 + (agent_pos["z"] - oz) ** 2) ** 0.5

    def _resolve_object_ids(
        self,
        pattern: str,
        *,
        agent_id: Optional[int] = None,
        policy: str = "nearest",
        require_openable: Optional[bool] = None,
        require_receptacle: Optional[bool] = None,
        require_pickupable: Optional[bool] = None,
        require_visible: Optional[bool] = None,
        require_closed: Optional[bool] = None,
        require_open: Optional[bool] = None,
    ) -> List[str]:
        """
        Resolve object type/name/objectId pattern -> objectId list.
        This is the shared layer for multi-instance objects (Cabinet/Apple/Fridge/...).
        """
        objs = self.controller.last_event.metadata.get("objects", [])
        candidates = []

        for obj in objs:
            if not self._object_matches_pattern(obj, pattern):
                continue
            if require_openable is not None and bool(obj.get("openable", False)) != require_openable:
                continue
            if require_receptacle is not None and bool(obj.get("receptacle", False)) != require_receptacle:
                continue
            if require_pickupable is not None and bool(obj.get("pickupable", False)) != require_pickupable:
                continue
            if require_visible is not None and bool(obj.get("visible", False)) != require_visible:
                continue
            if require_closed is not None:
                is_open = bool(obj.get("isOpen", False))
                if require_closed and is_open:
                    continue
                if (not require_closed) and (not is_open):
                    continue
            if require_open is not None and bool(obj.get("isOpen", False)) != require_open:
                continue
            candidates.append(obj)

        # deterministic order first
        candidates.sort(key=lambda o: str(o.get("objectId", "")))

        if agent_id is not None:
            candidates.sort(key=lambda o: self._distance_agent_to_obj(agent_id, o))
        elif policy == "nearest":
            # fallback to deterministic if no agent context
            policy = "first"

        if policy == "first":
            return [c["objectId"] for c in candidates]
        if policy == "last":
            return [c["objectId"] for c in reversed(candidates)]
        if policy == "random":
            shuffled = list(candidates)
            random.shuffle(shuffled)
            return [c["objectId"] for c in shuffled]
        # default "nearest" (or unknown): sorted by distance if agent_id provided
        return [c["objectId"] for c in candidates]

    def _resolve_object_id(
        self,
        pattern: str,
        *,
        agent_id: Optional[int] = None,
        policy: str = "nearest",
        **kwargs,
    ) -> Optional[str]:
        ids = self._resolve_object_ids(
            pattern,
            agent_id=agent_id,
            policy=policy,
            **kwargs,
        )
        return ids[0] if ids else None

    def _find_object_id(self, obj_pattern: str) -> Optional[str]:
        """객체 이름/타입/ID 패턴을 objectId 하나로 resolve (기본: 첫/가까운 후보)."""
        return self._resolve_object_id(obj_pattern, policy="first")

    def _cache_key(self, agent_id: int, pattern: str) -> Tuple[int, str]:
        return (agent_id, pattern.strip().lower())

    def _get_cached_receptacle(self, agent_id: int, pattern: str) -> Optional[str]:
        return self.receptacle_cache.get(self._cache_key(agent_id, pattern))

    def _set_cached_receptacle(self, agent_id: int, pattern: str, obj_id: Optional[str]) -> None:
        if obj_id:
            self.receptacle_cache[self._cache_key(agent_id, pattern)] = obj_id

    def _find_object_with_center(
        self,
        obj_pattern: str,
        agent_id: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[dict]]:
        """객체 이름/타입/ID 패턴을 기준으로 objectId와 중심점(center) resolve."""
        obj_id = self._resolve_object_id(
            obj_pattern,
            agent_id=agent_id,
            policy="nearest",
        )
        if not obj_id:
            return None, None
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj.get("objectId") != obj_id:
                continue
            center = obj.get("axisAlignedBoundingBox", {}).get("center")
            if center and center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                return obj_id, center
            # fallback: if bbox center not available, use object position
            pos = obj.get("position")
            if pos:
                return obj_id, pos
        return None, None

    def _find_closest_receptacle(self, recp_pattern: str, agent_id: int) -> Optional[str]:
        """해당 agent 기준으로 가장 가까운 receptacle 찾기"""
        return self._resolve_object_id(
            recp_pattern,
            agent_id=agent_id,
            policy="nearest",
            require_receptacle=True,
        )

    def _agent_has_pending_actions(self, agent_id: int) -> bool:
        """해당 로봇의 액션이 action_queue에 남아 있는지 확인"""
        with self.action_lock:
            if not self.action_queues:
                return False
            if agent_id < 0 or agent_id >= len(self.action_queues):
                return False
            return len(self.action_queues[agent_id]) > 0

    def _identify_blocking_robot(self, agent_id: int, threshold: float = 0.65) -> Optional[int]:
        try:
            my_pos = self.controller.last_event.events[agent_id].metadata["agent"]["position"]

            best = None
            best_dist = float("inf")

            for other_id in range(self.no_robot):
                if other_id == agent_id:
                    continue

                # pending 체크 제거: 두 로봇이 동시에 막힌 경우에도 서로를 blocker로 인식해야 함
                other_pos = self.controller.last_event.events[other_id].metadata["agent"]["position"]
                dist = ((my_pos["x"] - other_pos["x"])**2 + (my_pos["z"] - other_pos["z"])**2) ** 0.5

                if dist < threshold and dist < best_dist:
                    best_dist = dist
                    best = other_id

            return best
        except Exception:
            return None

    def _count_nearby_robots(self, agent_id: int, threshold: float = 0.9) -> int:
        """threshold 이내에 있는 다른 로봇 수를 반환 (pending 여부 무관)."""
        try:
            my_pos = self.controller.last_event.events[agent_id].metadata["agent"]["position"]
            count = 0
            for other_id in range(self.no_robot):
                if other_id == agent_id:
                    continue
                other_pos = self.controller.last_event.events[other_id].metadata["agent"]["position"]
                dist = ((my_pos["x"] - other_pos["x"])**2 + (my_pos["z"] - other_pos["z"])**2) ** 0.5
                if dist < threshold:
                    count += 1
            return count
        except Exception:
            return 0

    def _try_detour(self, agent_id: int):
        """2대 이상의 로봇이 진로를 막을 때 큰 우회 기동: 뒤로 빠지고 큰 각도로 돌아 다른 루트로 진입."""
        # 뒤로 2~3발
        for _ in range(2):
            self._enqueue_and_wait({'action': 'MoveBack', 'agent_id': agent_id},
                                   agent_id=agent_id, timeout=3.0)
        # 90~150도 랜덤 회전 (좌우 중 선택)
        angle = 90 + random.randint(0, 60)
        rot_action = 'RotateRight' if random.random() < 0.5 else 'RotateLeft'
        self._enqueue_and_wait({'action': rot_action, 'degrees': angle, 'agent_id': agent_id},
                               agent_id=agent_id, timeout=3.0)
        # 앞으로 2발 전진 (다른 경로로 진입)
        for _ in range(2):
            self._enqueue_and_wait({'action': 'MoveAhead', 'agent_id': agent_id},
                                   agent_id=agent_id, timeout=3.0)
        print(f"[Robot{agent_id+1}] Detour: 2 robots blocking, trying alternate route ({rot_action} {angle}°)")

    def _issue_yield_request(self, blocking_id: int, requester_id: int, target_object: str) -> bool:
        now = time.time()
        with self.bb_lock:
            # 너무 자주 갱신하는 거 방지(예: 1초 쿨다운)
            old = self.yield_requests.get(blocking_id)
            if old and (now - old.timestamp) < self.yield_cooldown_s:
                return False
            last_pair = self.last_yield_request.get((blocking_id, requester_id))
            if last_pair and (now - last_pair) < self.yield_cooldown_s:
                return False

            # 상호 데드락 방지: requester도 blocking_id에게 양보 요청을 받고 있으면
            # agent_id가 높은 쪽이 양보 (낮은 쪽이 우선권)
            mutual = self.yield_requests.get(requester_id)
            if mutual and mutual.requester_id == blocking_id:
                if requester_id > blocking_id:
                    # 내가 우선순위 낮음 → 내가 양보해야 함, 상대에게 양보 요청 안 함
                    return False
                else:
                    # 내가 우선순위 높음 → 상대의 양보 요청 제거하고 내 요청 등록
                    del self.yield_requests[requester_id]

            self.yield_requests[blocking_id] = YieldRequest(
                requester_id=requester_id,
                timestamp=now,
                reason=f"yield_for_{target_object}",
                target_object=target_object,
                attempts=0,
                last_distance=0.0,
                next_time=now,
            )
            self.last_yield_request[(blocking_id, requester_id)] = now
        return True
    def _find_yield_position(self, blocker_pos: dict, requester_pos: dict, step: float = 0.75) -> Optional[dict]:
        # blocker -> requester 벡터
        vx = requester_pos["x"] - blocker_pos["x"]
        vz = requester_pos["z"] - blocker_pos["z"]
        norm = (vx*vx + vz*vz) ** 0.5
        if norm < 1e-6:
            return None

        vx /= norm
        vz /= norm

        # 8방향 후보 (우선순위: 수직 → 반대 → 대각선)
        inv2 = 0.707  # 1/sqrt(2) for diagonal normalization
        candidates = [
            ( -vz,  vx),                              # left (수직)
            (  vz, -vx),                              # right (수직)
            ( -vx, -vz),                              # backward (requester 반대)
            ( (-vx - vz) * inv2, (-vz + vx) * inv2),  # back-left (대각선)
            ( (-vx + vz) * inv2, (-vz - vx) * inv2),  # back-right (대각선)
            ( ( vx - vz) * inv2, ( vz + vx) * inv2),  # front-left (대각선)
            ( ( vx + vz) * inv2, ( vz - vx) * inv2),  # front-right (대각선)
        ]

        # 각 방향별로 가장 가까운 reachable 점 찾기 (방향 최선 유지)
        best = None
        best_d = float("inf")

        for px, pz in candidates:
            target_x = blocker_pos["x"] + px * step
            target_z = blocker_pos["z"] + pz * step

            # 이 방향 후보의 최적 reachable 점
            dir_best = None
            dir_best_d = float("inf")
            for rp in self.reachable_positions_:
                dx = rp["x"] - target_x
                dz = rp["z"] - target_z
                d = (dx*dx + dz*dz) ** 0.5
                if d < dir_best_d:
                    dir_best_d = d
                    dir_best = rp

            # 이 방향이 reachable하고 requester에서 멀어지는 점이면 우선
            if dir_best and dir_best_d < 1.0:
                # requester와의 거리가 현재보다 멀어지는지 확인
                new_dist_to_req = ((dir_best["x"] - requester_pos["x"])**2 +
                                   (dir_best["z"] - requester_pos["z"])**2) ** 0.5
                cur_dist_to_req = ((blocker_pos["x"] - requester_pos["x"])**2 +
                                   (blocker_pos["z"] - requester_pos["z"])**2) ** 0.5
                # requester에서 멀어지는 점 우선, 같은 거리면 reachable 정확도 우선
                score = dir_best_d - (new_dist_to_req - cur_dist_to_req) * 0.5
                if score < best_d:
                    best_d = score
                    best = dir_best

        if best is None:
            return None
        return dict(x=best["x"], y=best["y"], z=best["z"])

    def _force_unstick_blocker(self, blocking_id: int) -> bool:
        """yield가 반복 실패할 때 blocker를 강제로 뒤로 빼고 회전시켜 데드락 해소."""
        now = time.time()
        last = self._last_forced_unstick.get(blocking_id, 0.0)
        if (now - last) < self.yield_force_unstick_cooldown_s:
            return False
        if self._agent_has_pending_actions(blocking_id):
            return False

        self._enqueue_front([
            {"action": "RotateRight", "degrees": 45, "agent_id": blocking_id},
            {"action": "MoveBack", "agent_id": blocking_id},
            {"action": "MoveBack", "agent_id": blocking_id},
        ])
        self._last_forced_unstick[blocking_id] = now
        #print(f"[Yield] Force unstick Robot{blocking_id+1}: MoveBack+Rotate")
        return True

    def _monitor_path_clear_requests(self):
        while not self.task_over:
            req_items = []
            with self.bb_lock:
                # 복사해서 락 오래 안 잡기
                req_items = list(self.yield_requests.items())

            for blocking_id, req in req_items:
                try:
                    # 최대 재시도 횟수 초과: yield 요청 강제 종료
                    if req.attempts >= 15:
                        self._force_unstick_blocker(blocking_id)
                        with self.bb_lock:
                            if self.yield_requests.get(blocking_id) == req:
                                del self.yield_requests[blocking_id]
                        continue

                    blocker_pos = self.controller.last_event.events[blocking_id].metadata["agent"]["position"]
                    requester_pos = self.controller.last_event.events[req.requester_id].metadata["agent"]["position"]
                    dist = ((blocker_pos["x"] - requester_pos["x"])**2 + (blocker_pos["z"] - requester_pos["z"])**2) ** 0.5

                    # 이미 충분히 멀어졌다면 요청 종료
                    if dist >= (self.yield_clear_distance + self.yield_margin):
                        with self.bb_lock:
                            if self.yield_requests.get(blocking_id) == req:
                                del self.yield_requests[blocking_id]
                        continue

                    # 요청 쿨다운
                    now = time.time()
                    if req.next_time and now < req.next_time:
                        continue

                    if req.last_distance <= 0:
                        req.last_distance = dist

                    # 단계적으로 더 큰 step 시도
                    step_candidates = [0.75, 1.25, 1.75]
                    step = step_candidates[min(req.attempts, len(step_candidates) - 1)]
                    target_position = self._find_yield_position(blocker_pos, requester_pos, step=step)

                    if not target_position:
                        # reachable position을 찾지 못함 → 직접 MoveBack으로 requester 반대방향 이동
                        req.attempts += 1
                        # pending 여부 무관하게 _enqueue_front로 우선 삽입 (stuck 상황 타개)
                        self._enqueue_front([
                            {"action": "MoveBack", "agent_id": blocking_id},
                            {"action": "MoveBack", "agent_id": blocking_id},
                        ])
                        if req.attempts % self.yield_force_unstick_attempts == 0:
                            self._force_unstick_blocker(blocking_id)
                        req.next_time = now + self.yield_retry_delay_s
                        continue

                    # blocking 로봇에게 이동 명령: 3발 연속 (첫 발이 회전이어도 나머지로 실제 이동)
                    nav_step = {"action": "ObjectNavExpertAction", "position": target_position, "agent_id": blocking_id}
                    self._enqueue_front([nav_step, nav_step, nav_step])
                    req.attempts += 1
                    if req.attempts % self.yield_force_unstick_attempts == 0:
                        self._force_unstick_blocker(blocking_id)
                    req.next_time = now + self.yield_retry_delay_s

                    # 거리 증가 체크는 다음 루프에서 수행
                    req.last_distance = dist

                except Exception:
                    pass

            time.sleep(0.1)


    def _check_other_robot_blocking(self, agent_id: int, threshold: float = 1.0) -> bool:
        """충돌 방지 로직, 다른 로봇이 길을 막고 있는지 확인하고, 막혀 있다면 옆으로 비켜가는 회피 기동해주는 함수"""
        try:
            my_meta = self.controller.last_event.events[agent_id].metadata
            my_pos = my_meta["agent"]["position"]

            for other_id in range(self.no_robot):
                if other_id == agent_id:
                    continue
                other_meta = self.controller.last_event.events[other_id].metadata
                other_pos = other_meta["agent"]["position"]

                dist = ((my_pos["x"] - other_pos["x"])**2 +
                        (my_pos["z"] - other_pos["z"])**2)**0.5
                if dist < threshold:
                    return True
        except:
            pass
        return False

    def _try_avoid_collision(self, agent_id: int):
        """충돌 방지 로직, 다른 로봇이 길을 막고 있는지 확인하고, 막혀 있다면 옆으로 비켜가는 회피 기동해주는 함수"""

        avoidance_actions = [ #피하는 동작
            ('RotateRight', 45),
            ('MoveAhead', None),
            ('RotateLeft', 90),
            ('MoveAhead', None),
            ('RotateRight', 45),
        ]

        for action, param in avoidance_actions:
            if action == 'RotateRight':
                self._enqueue_action({
                    'action': 'RotateRight',
                    'degrees': param,
                    'agent_id': agent_id
                })
            elif action == 'RotateLeft':
                self._enqueue_action({
                    'action': 'RotateLeft',
                    'degrees': param,
                    'agent_id': agent_id
                })
            elif action == 'MoveAhead':
                self._enqueue_action({
                    'action': 'MoveAhead',
                    'agent_id': agent_id
                })
            time.sleep(0.3)

    def _clear_agent_queue(self, agent_id: int):
        """해당 로봇의 action queue를 비움 (네비게이션 복구 등 남은 액션 정리)"""
        with self.action_lock:
            if self.action_queues and agent_id < len(self.action_queues):
                cleared = len(self.action_queues[agent_id])
                self.action_queues[agent_id].clear()

    def GoToObject(self, agent_id: int, dest_obj: str) -> bool:
        """로봇을 목적지까지 이동시켜주는 함수, ObjectNavExpertAction을 활용해 장애물을 피해가며 목적지 도달 후 물체를 바라보도록 정렬시킴 """
        #print(f"[Robot{agent_id+1}] Going to {dest_obj}")

        # 대상 물체 찾기 및 위치 파악
        dest_obj_id, dest_obj_center = self._find_object_with_center(dest_obj, agent_id=agent_id)
        if not dest_obj_id or not dest_obj_center:
            print(f"[Robot{agent_id+1}] Cannot find {dest_obj}")
            self._fail_current_subtask(f"GoToObject: cannot find object '{dest_obj}'")
            return False

        #print(f"[Robot{agent_id+1}] Target: {dest_obj_id} at {dest_obj_center}")

        dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

        # 내비게이션(이동) 파라미터 설정
        dist_goal = 10.0
        prev_dist_goal = 10.0
        count_since_update = 0 # 이동 정체 횟수 카운트
        clost_node_location = [0] # 도달 가능한 지점 인덱스
        goal_thresh = 0.25 # 목표 지점 도달 인정 거리
        max_iterations = 100   # 최대 반복 횟수 (무한 루프 방지)
        max_recoveries = 3
        recovery_attempts = 0
        collision_retry_count = 0 # 충돌 회피 시도 횟수
        max_collision_retries = 3 # 최대 충돌 회피 시도 횟수
        prev_rot_only = self.nav_rotation_only_count[agent_id] if agent_id < len(self.nav_rotation_only_count) else 0
        rot_recovery_count = 0  # rotation-only 복구 연속 횟수 (복구 전략 에스컬레이션용)

        # Oscillation 감지용 위치 히스토리 (실제 이동한 위치만 기록)
        position_history: List[Tuple[float, float]] = []  # (x, z) 리스트
        oscillation_count = 0
        oscillation_cooldown = 0  # recovery 후 쿨다운 카운터

        # Yield 통합 관리: GoToObject 단위로 1개만 활성
        yield_active_for: Optional[int] = None  # 현재 yield 요청 중인 blocking robot id
        yield_wait_iters = 0  # yield 발행 후 blocker 이동 대기 카운터
        YIELD_WAIT_MAX = 8    # yield 후 최소 대기 iterations (blocker가 이동할 시간)
        ineffective_yield_counts: dict = {}  # per-blocker 연속 ineffective yield 횟수

        # 대상 물체와 가장 가까운 '이동 가능한 지점' 가져오기
        crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

        iteration = 0
        while dist_goal > goal_thresh and iteration < max_iterations:
            iteration += 1

            # 현재 로봇의 위치 정보(메타데이터) 가져오기
            metadata = self.controller.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
            }

            prev_dist_goal = dist_goal
            # 목표 지점과 현재 로봇 위치 사이의 거리 계산
            dist_goal = distance_pts(
                [location['x'], location['y'], location['z']],
                crp[0]
            )

            dist_del = abs(dist_goal - prev_dist_goal)

            # 진행 상황 로그 (10회마다)
            #if iteration % 10 == 0:
                #print(f"[Robot{agent_id+1}] Nav {iteration}/{max_iterations}: dist={dist_goal:.2f}, stall={count_since_update}, osc={oscillation_count}, queue={self._queue_total_len()}")

            # --- Oscillation (앞뒤 반복) 감지 ---
            cur_xz = (location['x'], location['z'])

            # 쿨다운 중이면 감지 건너뜀
            if oscillation_cooldown > 0:
                oscillation_cooldown -= 1
            else:
                # 실제로 이동한 경우에만 히스토리에 기록 (회전만 한 경우 제외)
                moved = True
                if position_history:
                    last_xz = position_history[-1]
                    move_dist = ((cur_xz[0] - last_xz[0])**2 + (cur_xz[1] - last_xz[1])**2) ** 0.5
                    if move_dist < self.nav_oscillation_move_thresh:
                        moved = False

                if moved:
                    # 히스토리에서 3개 이전 위치 중 재방문 확인 (실제 이동 기반)
                    revisit = False
                    if len(position_history) >= 3:
                        for old_xz in position_history[:-2]:
                            dx = cur_xz[0] - old_xz[0]
                            dz = cur_xz[1] - old_xz[1]
                            if (dx*dx + dz*dz) ** 0.5 < self.nav_oscillation_radius:
                                revisit = True
                                break
                    if revisit:
                        oscillation_count += 1
                    else:
                        oscillation_count = max(0, oscillation_count - 1)

                    position_history.append(cur_xz)
                    if len(position_history) > self.nav_position_history_size:
                        position_history.pop(0)

            # Oscillation 감지 시 강제 경로 전환
            if oscillation_count >= self.nav_oscillation_threshold:
                # Navigation 복구 로그 비활성화
                # print(f"[Robot{agent_id+1}] Oscillation detected! Switching to alternative path")
                # 뒤로 한 칸 이동 후 다른 접근 지점으로 전환
                self._enqueue_and_wait({
                    'action': 'MoveBack',
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)
                self._enqueue_and_wait({
                    'action': 'RotateRight',
                    'degrees': 60 + random.randint(0, 60),
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)
                self._enqueue_and_wait({
                    'action': 'MoveAhead',
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=5.0)

                clost_node_location[0] += 1
                max_positions = max(4 * 4, len(self.reachable_positions) // 5)  # 사분면4 × 순위4 = 최소 16
                if clost_node_location[0] >= max_positions:
                    clost_node_location[0] = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                oscillation_count = 0
                count_since_update = 0
                position_history.clear()
                oscillation_cooldown = self.nav_oscillation_cooldown_iters
                time.sleep(0.1)
                continue

            # 로봇이 갇혔거나 멈췄는지 확인
            if dist_del < 0.15:
                count_since_update += 1
            else:
                count_since_update = 0
                collision_retry_count = 0  # 이동 중이면 충돌 카운트 초기화
                rot_recovery_count = 0     # 실제 이동 성공 시 복구 에스컬레이션 리셋
                # 실제 이동 성공 → 활성 yield 해소 확인
                if yield_active_for is not None:
                    ineffective_yield_counts.pop(yield_active_for, None)
                    yield_active_for = None
                    yield_wait_iters = 0

            # --- 활성 yield 대기 중이면 blocker 이동 시간을 줌 ---
            if yield_active_for is not None:
                yield_wait_iters += 1
                if yield_wait_iters < YIELD_WAIT_MAX:
                    # blocker가 이동할 시간을 줌 (접근점 변경/yield 재발행 안 함)
                    time.sleep(0.1)
                    continue
                else:
                    # 대기 끝: yield가 효과 없었음
                    blocker_id = yield_active_for
                    ineffective_yield_counts[blocker_id] = ineffective_yield_counts.get(blocker_id, 0) + 1
                    # 2회 이상 yield 실패 + 주변에 2대 이상 → 우회 기동
                    if ineffective_yield_counts[blocker_id] >= 2:
                        nearby_count = self._count_nearby_robots(agent_id, threshold=1.0)
                        if nearby_count >= 2:
                            self._try_detour(agent_id)
                            ineffective_yield_counts[blocker_id] = 0
                            yield_active_for = None
                            yield_wait_iters = 0
                            clost_node_location[0] += 4  # 접근점 크게 건너뜀
                            max_positions = max(4 * 4, len(self.reachable_positions) // 5)
                            if clost_node_location[0] >= max_positions:
                                clost_node_location[0] = 0
                            crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                            count_since_update = 0
                            time.sleep(0.1)
                            continue
                    # 3회 연속 ineffective → 강제로 blocker 이동
                    if ineffective_yield_counts[blocker_id] >= 3:
                        self._force_unstick_blocker(blocker_id)
                        ineffective_yield_counts[blocker_id] = 0
                    yield_active_for = None
                    yield_wait_iters = 0
                    clost_node_location[0] += 1
                    max_positions = max(4 * 4, len(self.reachable_positions) // 5)  # 사분면4 × 순위4 = 최소 16
                    if clost_node_location[0] >= max_positions:
                        clost_node_location[0] = 0
                    crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                    count_since_update = 0
                    time.sleep(0.1)
                    continue

            # --- Stuck 감지 (rotation-only + general stall 통합) ---
            need_yield = False
            need_approach_change = False
            skip_amount = 1

            # rotation-only 반복 감지
            if agent_id < len(self.nav_rotation_only_count):
                rot_only = self.nav_rotation_only_count[agent_id]
                if rot_only - prev_rot_only >= self.nav_rotation_only_threshold:
                    rot_recovery_count += 1
                    skip_amount = min(rot_recovery_count, 5)
                    need_approach_change = True
                    # 로봇이 막고 있는지 확인
                    blocking = self._identify_blocking_robot(agent_id, threshold=0.65)
                    if blocking is not None and blocking != agent_id:
                        need_yield = True
                    # 카운터 리셋
                    self.nav_rotation_only_count[agent_id] = 0
                    prev_rot_only = 0

            # general stall 감지
            if count_since_update >= 5 and not need_approach_change:
                blocking = self._identify_blocking_robot(agent_id, threshold=0.65)
                if blocking is not None and blocking != agent_id:
                    # 2대 이상이 막고 있으면 yield 대신 즉시 우회 기동
                    nearby_count = self._count_nearby_robots(agent_id, threshold=0.9)
                    if nearby_count >= 2:
                        self._try_detour(agent_id)
                        skip_amount = 4  # 접근점도 크게 건너뜀
                        need_approach_change = True
                        count_since_update = 0
                    else:
                        need_yield = True
                        need_approach_change = False  # yield 결과를 먼저 기다림
                else:
                    collision_retry_count += 1
                    if collision_retry_count >= max_collision_retries:
                        need_approach_change = True
                        collision_retry_count = 0

            # --- 통합 yield 발행 (1회만) ---
            if need_yield:
                blocking = self._identify_blocking_robot(agent_id, threshold=0.65)
                if blocking is not None and blocking != agent_id:
                    issued = self._issue_yield_request(
                        blocking_id=blocking,
                        requester_id=agent_id,
                        target_object=dest_obj
                    )
                    if issued:
                        #print(f"[Robot{agent_id+1}] 요청: Robot{blocking+1} 길 비켜줘 ({dest_obj})")
                        yield_active_for = blocking
                        yield_wait_iters = 0
                    count_since_update = 0
                    time.sleep(0.1)
                    continue

            # --- 접근점 변경 ---
            if need_approach_change:
                clost_node_location[0] += skip_amount
                max_positions = max(4 * 4, len(self.reachable_positions) // 5)  # 사분면4 × 순위4 = 최소 16
                if clost_node_location[0] >= max_positions:
                    if collision_retry_count >= max_collision_retries:
                        #print(f"[Robot{agent_id+1}] Exhausted reachable positions, stopping navigation")
                        break
                    clost_node_location[0] = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)
                #print(f"[Robot{agent_id+1}] Switching approach point (skip +{skip_amount})")
                count_since_update = 0
                time.sleep(0.1)
                continue

            # 정상 이동 가능 시 경로 최적화 액션 수행
            if count_since_update < 5:
                ok = self._enqueue_and_wait({
                    'action': 'ObjectNavExpertAction',
                    'position': dict(x=crp[0][0], y=crp[0][1], z=crp[0][2]),
                    'agent_id': agent_id
                }, agent_id=agent_id, timeout=10.0)
                if not ok:
                    # if action not processed in time, continue loop to avoid stale state
                    time.sleep(0.1)
            else:
                # 5회 이상 정체 시, 다음으로 가까운 이동 가능 지점으로 목표 업데이트
                clost_node_location[0] += 1
                count_since_update = 0

                # 인덱스 범위 초과 확인 (안전 장치)
                max_positions = max(4 * 4, len(self.reachable_positions) // 5)  # 사분면4 × 순위4 = 최소 16
                if clost_node_location[0] >= max_positions:
                    # Navigation 복구 로그 비활성화
                    # print(f"[Robot{agent_id+1}] Exhausted reachable positions, stopping navigation")
                    break

                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

            time.sleep(0.1)

        if iteration >= max_iterations and dist_goal > goal_thresh:
            # Recovery loop: re-sample approach + wait + retry (limited)
            while recovery_attempts < max_recoveries and dist_goal > goal_thresh:
                recovery_attempts += 1
                # 회전 카운터 리셋: 이전 루프의 누적값이 recovery에 영향 주지 않도록
                if agent_id < len(self.nav_rotation_only_count):
                    self.nav_rotation_only_count[agent_id] = 0
                    prev_rot_only = 0
                # Navigation 복구 로그 비활성화
                # print(f"[Robot{agent_id+1}] Navigation timeout: recovery {recovery_attempts}/{max_recoveries}")

                # 1) wait briefly to let others move
                time.sleep(0.6)

                # 2) re-sample a different approach point
                clost_node_location[0] += 1
                max_positions = max(4 * 4, len(self.reachable_positions) // 5)  # 사분면4 × 순위4 = 최소 16
                if clost_node_location[0] >= max_positions:
                    clost_node_location[0] = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, 1, clost_node_location)

                # 3) retry a short navigation window (15회로 축소, stall 감지 포함)
                recovery_stall = 0
                prev_recovery_dist = dist_goal
                for ri in range(15):
                    metadata = self.controller.last_event.events[agent_id].metadata
                    location = {
                        "x": metadata["agent"]["position"]["x"],
                        "y": metadata["agent"]["position"]["y"],
                        "z": metadata["agent"]["position"]["z"],
                        "rotation": metadata["agent"]["rotation"]["y"],
                    }
                    dist_goal = distance_pts(
                        [location['x'], location['y'], location['z']],
                        crp[0]
                    )
                    if dist_goal <= goal_thresh:
                        break
                    # recovery 중에도 진행 상황 로그
                    if ri % 5 == 0:
                        # Navigation 복구 로그 비활성화
                        # print(f"[Robot{agent_id+1}] Recovery nav {ri}/15: dist={dist_goal:.2f}")
                        pass
                    # stall 감지: recovery 중에도 진전 없으면 빠르게 포기
                    if abs(dist_goal - prev_recovery_dist) < 0.1:
                        recovery_stall += 1
                    else:
                        recovery_stall = 0
                    prev_recovery_dist = dist_goal
                    if recovery_stall >= 5:
                        # Navigation 복구 로그 비활성화
                        # print(f"[Robot{agent_id+1}] Recovery stalled, trying next approach")
                        break
                    self._enqueue_and_wait({
                        'action': 'ObjectNavExpertAction',
                        'position': dict(x=crp[0][0], y=crp[0][1], z=crp[0][2]),
                        'agent_id': agent_id
                    }, agent_id=agent_id, timeout=5.0)

            if dist_goal > goal_thresh:
                # Navigation 복구 로그 비활성화
                # print(f"[Robot{agent_id+1}] Navigation timeout, giving up")
                self._clear_agent_queue(agent_id)
                self._fail_current_subtask(f"GoToObject: navigation timeout for '{dest_obj}' (dist={dist_goal:.2f})")
                return False

        # 네비게이션 성공 시 남은 복구 액션 정리
        self._clear_agent_queue(agent_id)

        # [회전 정렬] 로봇이 목적지 물체를 정면으로 바라보도록 회전
        try:
            metadata = self.controller.last_event.events[agent_id].metadata
            robot_location = {
                "x": metadata["agent"]["position"]["x"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
            }

            # 로봇에서 물체를 향하는 벡터 계산
            robot_object_vec = [
                dest_obj_pos[0] - robot_location['x'],
                dest_obj_pos[2] - robot_location['z']
            ]

            vec_magnitude = np.linalg.norm(robot_object_vec)
            if vec_magnitude > 0.01:  # 0으로 나누기 방지
                y_axis = [0, 1]
                unit_y = y_axis / np.linalg.norm(y_axis)
                unit_vector = robot_object_vec / vec_magnitude

                # 물체와의 각도 계산 후 회전 방향(좌/우) 및 각도 결정
                angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
                angle = 360 * angle / (2 * np.pi)
                angle = (angle + 360) % 360
                rot_angle = angle - robot_location['rotation']

                if rot_angle > 0:
                    self._enqueue_action({
                        'action': 'RotateRight',
                        'degrees': abs(rot_angle),
                        'agent_id': agent_id
                    })
                else:
                    self._enqueue_action({
                        'action': 'RotateLeft',
                        'degrees': abs(rot_angle),
                        'agent_id': agent_id
                    })
        except Exception as e:
            print(f"[Robot{agent_id+1}] Alignment error: {e}")

        if dist_goal > goal_thresh:
            #print(f"[Robot{agent_id+1}] FAIL to reach {dest_obj}, aborting")
            self._clear_agent_queue(agent_id)
            self._fail_current_subtask(f"GoToObject: failed to reach '{dest_obj}' (dist={dist_goal:.2f})")
            return False

        #print(f"[Robot{agent_id+1}] Reached {dest_obj}")
        # checker: NavigateTo 크레딧
        obj_id_nav = self._find_object_id(dest_obj)
        readable = self._convert_object_id_to_readable(obj_id_nav) if obj_id_nav else dest_obj
        self._checker_report(agent_id, "NavigateTo", readable, True)
        time.sleep(0.1)
        return True

    def PickupObject(self, agent_id: int, obj_pattern: str):
        """Pick up object."""
        #print(f"[Robot{agent_id+1}] Picking up {obj_pattern}")

        obj_id, _ = self._find_object_with_center(obj_pattern, agent_id=agent_id)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            self._fail_current_subtask(f"PickupObject: cannot find object '{obj_pattern}'")
            return

        #print(f"[Robot{agent_id+1}] Picking up {obj_id}")
        success = self._enqueue_and_wait({
            'action': 'PickupObject',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"PickupObject failed or timed out for '{obj_pattern}'")
        # checker: PickupObject (소문자 u — checker 형식에 맞춤)
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "PickUpObject", readable, success)

    def PutObject(self, agent_id: int, obj_pattern: str, recp_pattern: str):
        """Put held object on/in receptacle."""
        #print(f"[Robot{agent_id+1}] Putting {obj_pattern} on/in {recp_pattern}")

        recp_id = self._get_cached_receptacle(agent_id, recp_pattern)
        if not recp_id:
            recp_id = self._find_closest_receptacle(recp_pattern, agent_id)
            self._set_cached_receptacle(agent_id, recp_pattern, recp_id)
        if not recp_id:
            print(f"[Robot{agent_id+1}] Cannot find receptacle {recp_pattern}")
            self._fail_current_subtask(f"PutObject: cannot find receptacle '{recp_pattern}'")
            return

        #print(f"[Robot{agent_id+1}] Putting on {recp_id}")
        # checker: put 전 inventory 저장 (put 후엔 비어있으므로)
        self._update_inventory(agent_id)
        inv_before_put = self.inventory[agent_id]
        # NOTE: PutObject uses objectId for receptacle!
        success = self._enqueue_and_wait({
            'action': 'PutObject',
            'objectId': recp_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"PutObject failed or timed out for receptacle '{recp_pattern}'")
        # checker: PutObject(receptacle) with inventory_object
        if self.checker is not None:
            recp_readable = self._convert_object_id_to_readable(recp_id)
            action_str = f"PutObject({recp_readable})"
            self.checker.perform_metric_check(action_str, success, inv_before_put)

    def OpenObject(self, agent_id: int, obj_pattern: str):
        """Open object."""
        #print(f"[Robot{agent_id+1}] Opening {obj_pattern}")

        obj_id = self._get_cached_receptacle(agent_id, obj_pattern)
        if not obj_id:
            obj_id = self._find_object_id(obj_pattern)
            self._set_cached_receptacle(agent_id, obj_pattern, obj_id)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        # 서랍/냉장고 등 열리는 물체 앞에서 두 발 뒤로 빠져서 충돌 방지
        for _ in range(2):
            self._enqueue_and_wait({
                'action': 'MoveBack',
                'agent_id': agent_id
            }, agent_id=agent_id, timeout=3.0)
        time.sleep(0.5)

        success = self._enqueue_and_wait({
            'action': 'OpenObject',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"OpenObject failed or timed out for '{obj_pattern}'")
        # checker: OpenObject
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "OpenObject", readable, success)

    def CloseObject(self, agent_id: int, obj_pattern: str):
        """Close object."""
        #print(f"[Robot{agent_id+1}] Closing {obj_pattern}")

        obj_id = self._get_cached_receptacle(agent_id, obj_pattern)
        if not obj_id:
            obj_id = self._find_object_id(obj_pattern)
            self._set_cached_receptacle(agent_id, obj_pattern, obj_id)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        success = self._enqueue_and_wait({
            'action': 'CloseObject',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        # checker: CloseObject
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "CloseObject", readable, success)

    def SwitchOn(self, agent_id: int, obj_pattern: str):
        """Turn on object."""
        #print(f"[Robot{agent_id+1}] Switching on {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        success = self._enqueue_and_wait({
            'action': 'ToggleObjectOn',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"ToggleObjectOn failed or timed out for '{obj_pattern}'")
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "ToggleObjectOn", readable, success)

    def SwitchOff(self, agent_id: int, obj_pattern: str):
        """Turn off object."""
        #print(f"[Robot{agent_id+1}] Switching off {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        success = self._enqueue_and_wait({
            'action': 'ToggleObjectOff',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"ToggleObjectOff failed or timed out for '{obj_pattern}'")
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "ToggleObjectOff", readable, success)

    def SliceObject(self, agent_id: int, obj_pattern: str):
        """Slice object."""
        #print(f"[Robot{agent_id+1}] Slicing {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            self._fail_current_subtask(f"SliceObject: cannot find object '{obj_pattern}'")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        success = self._enqueue_and_wait({
            'action': 'SliceObject',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        if not success:
            self._fail_current_subtask(f"SliceObject failed or timed out for '{obj_pattern}'")
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "SliceObject", readable, success)

    def BreakObject(self, agent_id: int, obj_pattern: str):
        """Break object."""
        #print(f"[Robot{agent_id+1}] Breaking {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        if not obj_id:
            print(f"[Robot{agent_id+1}] Cannot find {obj_pattern}")
            return

        if not self.GoToObject(agent_id, obj_pattern):
            return

        success = self._enqueue_and_wait({
            'action': 'BreakObject',
            'objectId': obj_id,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        readable = self._convert_object_id_to_readable(obj_id)
        self._checker_report(agent_id, "BreakObject", readable, success)

    def ThrowObject(self, agent_id: int, obj_pattern: str):
        """Throw held object."""
        #print(f"[Robot{agent_id+1}] Throwing {obj_pattern}")

        obj_id = self._find_object_id(obj_pattern)
        success = self._enqueue_and_wait({
            'action': 'ThrowObject',
            'objectId': obj_id or obj_pattern,
            'agent_id': agent_id
        }, agent_id=agent_id, timeout=5.0)
        readable = self._convert_object_id_to_readable(obj_id) if obj_id else obj_pattern
        self._checker_report(agent_id, "ThrowObject", readable, success)

    # -----------------------------
    # PDDL Action Parser & Executor 실행기
    # -----------------------------
    def _parse_action(self, action_str: str) -> Tuple[str, str, List[str]]:
        """PDDL 형식의 문자열(예: pickupobject robot1 apple)을 분석하여 명령 종류, 로봇, 대상 물체로 쪼개는 함수"""
        action_str = re.sub(r"\s*\(\d+\)\s*$", "", action_str).strip()
        parts = action_str.split()
        if len(parts) < 1:
            return ("", "", [])
        if len(parts) == 1:
            return (parts[0].lower(), "", [])
        action_type = parts[0].lower()
        robot = parts[1]
        objects = parts[2:] if len(parts) > 2 else []
        return (action_type, robot, objects)

    def _execute_pddl_action(self, agent_id: int, action_str: str):
        """파싱된 결과에 따라 위에서 정의한 GoToObject, PickupObject, OpenObject 등의 고수준 함수를 호출하여 실제 명령 큐에 넣는 함수"""
        atype, _, objs = self._parse_action(action_str)

        if atype == "gotoobject" and len(objs) >= 1:
            # return value 체크: GoToObject 실패 시 이미 _fail_current_subtask 호출됨
            self.GoToObject(agent_id, objs[0])

        elif atype == "pickupobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return  # _fail_current_subtask는 GoToObject 내부에서 호출됨
            self.PickupObject(agent_id, objs[0])

        elif atype == "putobject" and len(objs) >= 2:
            if not self.GoToObject(agent_id, objs[1]):
                return  # _fail_current_subtask는 GoToObject 내부에서 호출됨
            self.PutObject(agent_id, objs[0], objs[1])

        elif atype == "putobjectinfridge" and len(objs) >= 1:
            if not self.GoToObject(agent_id, "Fridge"):
                return  # _fail_current_subtask는 GoToObject 내부에서 호출됨
            #self.OpenObject(agent_id, "Fridge")
            self.PutObject(agent_id, objs[0], "Fridge")
            #self.CloseObject(agent_id, "Fridge")

        elif atype == "openobject" and len(objs) >= 1:
            self.OpenObject(agent_id, objs[0])

        elif atype == "closeobject" and len(objs) >= 1:
            self.CloseObject(agent_id, objs[0])

        elif atype == "openfridge":
            self.OpenObject(agent_id, "Fridge")

        elif atype == "closefridge":
            self.CloseObject(agent_id, "Fridge")

        elif atype == "switchon" and len(objs) >= 1:
            self.SwitchOn(agent_id, objs[0])

        elif atype == "switchoff" and len(objs) >= 1:
            self.SwitchOff(agent_id, objs[0])

        elif atype == "sliceobject" and len(objs) >= 1:
            self.SliceObject(agent_id, objs[0])

        elif atype == "breakobject" and len(objs) >= 1:
            self.BreakObject(agent_id, objs[0])

        elif atype == "throwobject" and len(objs) >= 1:
            if len(objs) >= 2:
                # target이 있으면 GoTo(target) + PutObject로 변환 (던지지 않고 놓기)
                target = objs[1]
                #print(f"[Robot{agent_id+1}] throwobject → GoTo({target}) + PutObject 변환")
                if not self.GoToObject(agent_id, target):
                    return
                self.PutObject(agent_id, objs[0], target)
            else:
                # target이 없으면 기존 throw 사용
                self.ThrowObject(agent_id, objs[0])

        elif atype == "drophandobject":
            if len(objs) >= 2:
                # target(receptacle)이 있으면 PutObject 사용 (놓기)
                target = objs[1]
                #print(f"[Robot{agent_id+1}] drophandobject → PutObject({objs[0]} on {target})")
                self.PutObject(agent_id, objs[0], target)
            elif len(objs) >= 1:
                # target 없으면 그냥 drop
                #print(f"[Robot{agent_id+1}] drophandobject → PutObject({objs[0]})")
                self.PutObject(agent_id, objs[0], objs[0])
            else:
                # 아무 정보도 없으면 AI2Thor DropHandObject 사용
                self._enqueue_action({
                    'action': 'PutObject',
                    'objectId': '',
                    'agent_id': agent_id
                })
                time.sleep(1)

        elif atype == "pushobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return
            obj_id = self._find_object_id(objs[0])
            if obj_id:
                self._enqueue_action({
                    'action': 'PushObject',
                    'objectId': obj_id,
                    'agent_id': agent_id
                })
            time.sleep(1)

        elif atype == "pullobject" and len(objs) >= 1:
            if not self.GoToObject(agent_id, objs[0]):
                return
            obj_id = self._find_object_id(objs[0])
            if obj_id:
                self._enqueue_action({
                    'action': 'PullObject',
                    'objectId': obj_id,
                    'agent_id': agent_id
                })
            time.sleep(1)

        else:
            print(f"[Robot{agent_id+1}] SKIP/UNKNOWN: {action_str}")

    # -----------------------------
    # AI2-THOR 관리 함수 모음
    # -----------------------------

    def get_live_positions(self) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
        """
        현재 실행 중인 씬에서 로봇/오브젝트 좌표를 가져옴 (새 컨트롤러를 만들지 않음).

        Returns:
            robot_positions: {robot_id(1-based): (x, y, z)}
            object_positions: {object_name_lower: (x, y, z)}
        """
        if self.controller is None:
            raise RuntimeError("[get_live_positions] No active controller")

        robot_positions: Dict[int, Tuple[float, float, float]] = {}
        for i in range(self.no_robot):
            try:
                pos = self.controller.last_event.events[i].metadata["agent"]["position"]
                robot_positions[i + 1] = (pos["x"], pos["y"], pos["z"])
            except (IndexError, KeyError) as e:
                print(f"[get_live_positions] Robot {i+1} position error: {e}")

        object_positions: Dict[str, Tuple[float, float, float]] = {}
        for obj in self.controller.last_event.metadata.get("objects", []):
            name = obj["objectType"].strip().lower()
            p = obj.get("position", {})
            object_positions[name] = (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))

        #print(f"[get_live_positions] Collected {len(robot_positions)} robot positions, {len(object_positions)} object positions (live)")
        return robot_positions, object_positions

    def get_live_objects_ai(self) -> List[Dict[str, Any]]:
        """
        실행 중인 씬에서 objects_ai 형식의 실시간 오브젝트 정보를 가져옴.
        get_ai2_thor_objects()와 동일한 형식이지만, 실행 중인 controller에서 읽음.

        Returns:
            [{"name": str, "mass": float, "locations": [str], "position": {x,y,z}}, ...]
        """
        if self.controller is None:
            raise RuntimeError("[get_live_objects_ai] No active controller")

        objects_ai = []
        for obj in self.controller.last_event.metadata.get("objects", []):
            name = obj["objectType"]
            mass = obj.get("mass", 0.0)

            parents = obj.get("parentReceptacles")
            if parents:
                locations = [p.split("|")[0] for p in parents]
            else:
                locations = ["Floor"]

            position = obj.get("position", {})
            objects_ai.append({
                "name": name,
                "mass": mass,
                "locations": locations,
                "position": {
                    "x": position.get("x", 0.0),
                    "y": position.get("y", 0.0),
                    "z": position.get("z", 0.0),
                },
            })

        #print(f"[get_live_objects_ai] Collected {len(objects_ai)} objects from live scene")
        return objects_ai

    @staticmethod
    def spawn_and_get_positions(
        floor_plan: int, agent_count: int
    ) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
        """
        AI2-THOR를 잠깐 시작하여 로봇 스폰 좌표 + 오브젝트 좌표를 가져오고 종료.

        Returns:
            robot_positions: {robot_id(1-based): (x, y, z)}
            object_positions: {object_name_lower: (x, y, z)}
        """
        controller = Controller(height=300, width=300)
        controller.reset(f"FloorPlan{floor_plan}")
        controller.step(dict(
            action='Initialize', agentMode="default", snapGrid=False,
            gridSize=0.25, rotateStepDegrees=20,
            visibilityDistance=100, fieldOfView=90, agentCount=agent_count
        ))

        # 로봇 랜덤 배치 (start_ai2thor과 동일 로직)
        reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        used: List[dict] = []
        min_distance = 1.5

        for i in range(agent_count):
            best_pos = None
            best_min_dist = -1
            for _ in range(50):
                candidate = random.choice(reachable)
                if not used:
                    best_pos = candidate
                    break
                min_d = min(
                    ((candidate["x"] - p["x"])**2 + (candidate["z"] - p["z"])**2)**0.5
                    for p in used
                )
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
                if min_d >= min_distance:
                    break
            if best_pos:
                controller.step(dict(action="Teleport", position=best_pos, agentId=i))
                used.append(best_pos)

        # 로봇 좌표 수집
        robot_positions: Dict[int, Tuple[float, float, float]] = {}
        for i in range(agent_count):
            pos = controller.last_event.events[i].metadata["agent"]["position"]
            robot_positions[i + 1] = (pos["x"], pos["y"], pos["z"])  # 1-based

        # 오브젝트 좌표 수집
        object_positions: Dict[str, Tuple[float, float, float]] = {}
        for obj in controller.last_event.metadata["objects"]:
            name = obj["objectType"].strip().lower()
            p = obj.get("position", {})
            object_positions[name] = (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))

        controller.stop()
        #print(f"[spawn_and_get_positions] Collected {agent_count} robot positions, {len(object_positions)} object positions")
        return robot_positions, object_positions

    def start_ai2thor(self, floor_plan: int, agent_count: int,
                      spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None):
        """시뮬레이터를 초기화하고 로봇들을 생성. spawn_positions가 있으면 해당 좌표에 배치."""
        scene = f"FloorPlan{floor_plan}"
        self.scene_name = scene
        print(f"[AI2-THOR] Starting scene={scene}, agentCount={agent_count}")

        self.no_robot = agent_count
        self.controller = Controller(height=1000, width=1000)
        self.controller.reset(scene)

        # 로봇 초기 설정
        self.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=100,
            fieldOfView=90,
            agentCount=agent_count
        ))

        # 탑뷰(Top-down) 카메라 추가
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

        # 이동 가능한 위치 정보 가져오기
        self.reachable_positions_ = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        self.reachable_positions = [
            (p["x"], p["y"], p["z"]) for p in self.reachable_positions_
        ]

        # 로봇 배치
        if not self.reachable_positions_:
            raise RuntimeError("No reachable positions returned by AI2-THOR; cannot place agents safely on NavMesh.")

        min_spawn_distance = 1.0  # 로봇 간 최소 스폰 거리
        if spawn_positions:
            # LP에서 결정된 좌표를 가장 가까운 NavMesh 유효 위치로 보정하여 배치
            used_reachable = []  # 이미 사용된 reachable position (x, z) 좌표
            for i in range(agent_count):
                rid = i + 1
                if rid in spawn_positions:
                    pos = spawn_positions[rid]
                    # 가장 가까운 reachable position 찾기 (NavMesh 보장 + 최소 간격)
                    best_rp, best_d = None, float("inf")
                    for rp in self.reachable_positions_:
                        # 다른 로봇과 최소 거리 보장
                        too_close = False
                        for ux, uz in used_reachable:
                            if ((rp["x"] - ux)**2 + (rp["z"] - uz)**2)**0.5 < min_spawn_distance:
                                too_close = True
                                break
                        if too_close:
                            continue
                        d = ((rp["x"] - pos[0])**2 + (rp["z"] - pos[2])**2)**0.5
                        if d < best_d:
                            best_d, best_rp = d, rp
                    # 최소 간격 제약 때문에 후보가 없다면, 제약을 완화해서라도 NavMesh 위 좌표를 선택
                    if best_rp is None:
                        for rp in self.reachable_positions_:
                            d = ((rp["x"] - pos[0])**2 + (rp["z"] - pos[2])**2)**0.5
                            if d < best_d:
                                best_d, best_rp = d, rp
                    if best_rp:
                        used_reachable.append((best_rp["x"], best_rp["z"]))
                        self.controller.step(dict(action="Teleport",
                            position=best_rp, agentId=i, forceAction=True))
                        if best_d > 0.0:
                            print(f"[Robot{rid}] Requested ({pos[0]:.2f}, {pos[2]:.2f}) -> snapped to reachable ({best_rp['x']:.2f}, {best_rp['z']:.2f}), dist={best_d:.2f}")
                        else:
                            print(f"[Robot{rid}] Requested ({pos[0]:.2f}, {pos[2]:.2f}) -> snapped to reachable ({best_rp['x']:.2f}, {best_rp['z']:.2f}), dist=0.00")
                    else:
                        # 안전상 이 분기에는 오지 않아야 하지만, 예외적으로 reachable이 비면 중단
                        raise RuntimeError(
                            f"[Robot{rid}] No reachable position available for spawn; aborting to avoid off-NavMesh teleport."
                        )
                    # 실제 위치 확인
                    actual = self.controller.last_event.events[i].metadata["agent"]["position"]
                    print(f"[Robot{rid}] Actual position: ({actual['x']:.2f}, {actual['z']:.2f})")
                else:
                    # 스폰 위치 미지정 로봇 → 다른 로봇과 떨어진 reachable position에 배치
                    best_rp, best_d = None, -1
                    for rp in self.reachable_positions_:
                        rp_tuple = (rp["x"], rp["z"])
                        if rp_tuple in used_reachable:
                            continue
                        min_d = min(
                            (((rp["x"] - ux)**2 + (rp["z"] - uz)**2)**0.5
                             for ux, uz in used_reachable),
                            default=float("inf")
                        )
                        if min_d > best_d:
                            best_d, best_rp = min_d, rp
                    if best_rp:
                        used_reachable.append((best_rp["x"], best_rp["z"]))
                        self.controller.step(dict(action="Teleport",
                            position=best_rp, agentId=i, forceAction=True))
                        print(f"[Robot{rid}] No spawn pos assigned, placed at reachable ({best_rp['x']:.2f}, {best_rp['z']:.2f})")
        else:
            # 기존 랜덤 배치 - 로봇 간 거리를 고려하여 배치
            used_positions = []
            min_distance = 1.5  # 로봇들간의 최소거리

            for i in range(agent_count):
                best_pos = None
                best_min_dist = -1

                # 다른 로봇과 가장 멀리 떨어진 위치를 찾기 위해 최대 50번 시도
                for _ in range(50):
                    candidate = random.choice(self.reachable_positions_)

                    if not used_positions:
                        best_pos = candidate
                        break

                    # 이미 배치된 모든 위치와의 최소 거리 계산
                    min_dist_to_others = min(
                        ((candidate["x"] - p["x"])**2 + (candidate["z"] - p["z"])**2)**0.5
                        for p in used_positions
                    )

                    if min_dist_to_others > best_min_dist:
                        best_min_dist = min_dist_to_others
                        best_pos = candidate

                    if min_dist_to_others >= min_distance:
                        break

                if best_pos:
                    self.controller.step(dict(action="Teleport", position=best_pos, agentId=i))
                    used_positions.append(best_pos)
                    print(f"[Robot{i+1}] Spawned at ({best_pos['x']:.2f}, {best_pos['z']:.2f})")

        # 물체를 더 잘 보기 위해 카메라를 약간 아래로 조절
        for i in range(agent_count):
            self.controller.step(action="LookDown", degrees=35, agentId=i)

        # 사용 가능한 물체 리스트
        objs = [obj["objectType"] for obj in self.controller.last_event.metadata["objects"]]

        # 액션 실행 스레드 시작
        self._build_object_dict()
        self.inventory = ["nothing"] * agent_count
        self.agent_success_counts = [0] * agent_count
        self.agent_action_counters = [0] * agent_count
        self.action_queues = [deque() for _ in range(agent_count)]
        self.nav_rotation_only_count = [0] * agent_count
        self.rr_index = 0
        self.task_over = False
        self.actions_thread = threading.Thread(target=self._exec_actions)
        self.actions_thread.start()

        self.monitor_thread = threading.Thread(target=self._monitor_path_clear_requests, daemon=True)
        self.monitor_thread.start()

        print("[AI2-THOR] Ready!")

    def stop_ai2thor(self):
        """AI2-THOR 시뮬레이터를 정지하는 함수"""
        self.task_over = True
        if self.actions_thread:
            self.actions_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except:
            pass

        try:
            self.controller.stop()
        except:
            pass

        print(f"[AI2-THOR] Stopped. Success: {self.success_exec}/{self.total_exec}")

    # -----------------------------
    # Subtask 실행기
    # -----------------------------

    def _run_subtask(self, plan: SubtaskPlan):
        """하나의 서브태스크에 포함된 모든 액션을 순차적으로 실행하는 함수. 피드백 모드일 때 결과를 _subtask_results에 기록하고, 실시간 스토어가 있으면 즉시 반영."""
        tid = threading.get_ident()
        self._thread_subtask_id[tid] = plan.subtask_id
        self._subtask_failed[plan.subtask_id] = False  # 초기화

        store = getattr(self, "_feedback_state_store", None)
        if store is not None:
            store.set_running(plan.subtask_id)

        agent_id = plan.robot_id - 1  # 1-기반 ID를 0-기반 인덱스로 변환

        # 로봇별 Lock 획득 (같은 로봇에 여러 서브태스크 할당 시 순차 보장)
        robot_lock = self._robot_locks[plan.robot_id]
        robot_lock.acquire()
        print(f"\n[Subtask {plan.subtask_id}] {plan.subtask_name} -> Robot{plan.robot_id}")

        try:
            self._subtask_completed_actions[plan.subtask_id] = []

            for i, action in enumerate(plan.actions):
                # 이미 실패한 서브태스크의 남은 액션은 건너뜀
                if self._subtask_failed.get(plan.subtask_id, False):
                    remaining = len(plan.actions) - i
                    print(f"[Subtask {plan.subtask_id}] Aborting remaining {remaining} action(s) due to earlier failure")
                    break
                print(f"[Robot{plan.robot_id}] Action {i+1}/{len(plan.actions)}: {action}")
                self._execute_pddl_action(agent_id, action)
                # 액션 실행 후 실패가 발생하지 않았으면 완료 목록에 추가
                if not self._subtask_failed.get(plan.subtask_id, False):
                    self._subtask_completed_actions[plan.subtask_id].append(action)

            success = not self._subtask_failed.get(plan.subtask_id, False)
            err = self._subtask_last_error.get(plan.subtask_id, "")
            completed = self._subtask_completed_actions.get(plan.subtask_id, [])
            self._subtask_results[plan.subtask_id] = SubTaskExecutionResult(
                subtask_id=plan.subtask_id,
                success=success,
                error_message=err or None,
                completed_actions=completed if completed else None,
            )
            if store is not None:
                store.update_subtask_on_completion(
                    plan.subtask_id,
                    success=success,
                    error_message=err or None,
                    effects=None,
                    completed_actions=completed if completed else None,
                )
            if not success and completed:
                print(f"[Subtask {plan.subtask_id}] Partially completed: {len(completed)}/{len(plan.actions)} actions succeeded")
            print(f"[Subtask {plan.subtask_id}] Completed! success={success}")

        finally:
            # 로봇 Lock 해제 (같은 로봇의 다음 서브태스크가 실행 가능)
            if hasattr(self, '_robot_locks') and plan.robot_id in self._robot_locks:
                try:
                    self._robot_locks[plan.robot_id].release()
                except RuntimeError:
                    pass  # 이미 해제된 경우
            self._thread_subtask_id.pop(tid, None)

    def execute_in_ai2thor(self, floor_plan: int, task_description: Optional[str] = None,
                           agent_count: Optional[int] = None,
                           spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None):
        """할당된 모든 서브태스크를 AI2-THOR에서 실행하는 함수.
        agent_count: 소환할 총 로봇 수. None이면 할당된 로봇 ID 최댓값 사용.
        spawn_positions: LP에서 결정된 스폰 좌표. None이면 랜덤 배치.
        """
        if not self.assignment or not self.parallel_groups or not self.subtask_plans:
            raise RuntimeError("Load assignment/DAG/plans first.")

        if agent_count is None:
            agent_count = getattr(self, 'configured_agent_count', None) \
                          or (max(self.assignment.values()) if self.assignment else 1)
        if spawn_positions is None:
            spawn_positions = getattr(self, 'saved_spawn_positions', None)
        self.start_ai2thor(floor_plan=floor_plan, agent_count=agent_count,
                           spawn_positions=spawn_positions)
        if task_description:
            self._init_checker(task_description, self.scene_name)

        # 병렬 그룹별로 서브태스크 그룹화
        groups_to_plans: Dict[int, List[SubtaskPlan]] = defaultdict(list)
        for sid, p in self.subtask_plans.items():
            groups_to_plans[p.parallel_group].append(p)

        print("=" * 60)
        print("[EXEC] Starting Multi-Robot Execution")
        print("=" * 60)

        try:
            for gid in sorted(groups_to_plans.keys()):
                plans = groups_to_plans[gid]
                print(f"\n[Group {gid}] {len(plans)} subtask(s) in parallel")

                # 같은 로봇에 할당된 서브태스크끼리는 순차 실행해야 하므로
                # 로봇별로 묶어서 하나의 스레드 안에서 순차 처리
                robot_plans: Dict[int, List[SubtaskPlan]] = defaultdict(list)
                for p in plans:
                    robot_plans[p.robot_id].append(p)

                def _run_robot_plans(plan_list: List[SubtaskPlan]):
                    for p in plan_list:
                        self._run_subtask(p)

                threads = []
                for _, plan_list in robot_plans.items():
                    # 로봇마다 하나의 스레드: 서로 다른 로봇끼리만 병렬
                    t = threading.Thread(target=_run_robot_plans, args=(plan_list,), daemon=True)
                    threads.append(t)

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                print(f"[Group {gid}] Completed!")

            # 모든 액션 큐가 비워질 때까지 대기
            print("\n[EXEC] Waiting for action queue to empty...")
            self._drain_action_queue()

            # 종료 전 확인을 위해 잠시 대기
            print("[EXEC] All done! Press 'q' in any window to close.")
            time.sleep(3)

            if self.checker is not None:
                try:
                    coverage = self.checker.get_coverage()
                    transport_rate = self.checker.get_transport_rate()
                    finished = self.checker.check_success()
                    balance = self._compute_balance_metric()
                    exec_rate = self._compute_exec_rate()
                    print("\n[[ RESULT ]]\n")
                    print(
                        f"Coverage:{coverage:.3f}, Transport Rate:{transport_rate:.3f}, "
                        f"Finished:{finished}, Balance:{balance:.3f}, Exec:{exec_rate:.3f}"
                    )
                    print("\n")
                    completed = sorted(getattr(self.checker, "subtasks_completed_numerated", []))
                    expected = sorted(getattr(self.checker, "subtasks", []))
                    if hasattr(self.checker, 'get_missing_subtasks'):
                        missing = self.checker.get_missing_subtasks()
                    else:
                        missing = sorted(set(expected) - set(completed))
                    print(f"[Checker] Completed ({len(completed)}): {completed}")
                    print(f"[Checker] Missing ({len(missing)}): {missing}")
                except Exception:
                    pass

        finally:
            self.stop_ai2thor()

    def execute_in_ai2thor_with_feedback(
        self,
        floor_plan: int,
        task_name: str = "task",
        task_description: Optional[str] = None,
        state_store: Optional[Any] = None,
        on_subtask_failed: Optional[Any] = None,
    ) -> Dict[int, SubTaskExecutionResult]:
        """피드백 루프용: 서브태스크를 그룹별·순차 실행하고, 서브태스크별 성공/실패 결과를 반환.
        on_subtask_failed: 실패 즉시 호출되는 콜백. signature: fn(result) -> bool
        state_store가 주어지면 개별 서브태스크 완료 시마다 즉시 상태를 반영(실시간 스토어 반영).
        """
        if not self.assignment or not self.parallel_groups or not self.subtask_plans:
            raise RuntimeError("Load assignment/DAG/plans first.")

        self._subtask_failed = {}
        self._subtask_results = {}
        self._subtask_last_error = {}
        self._subtask_completed_actions = {}
        self._dropped_subtasks = set()
        # 로봇별 Lock: 같은 로봇에 할당된 서브태스크는 순차 실행
        self._robot_locks: Dict[int, threading.Lock] = defaultdict(threading.Lock)

        if state_store is not None:
            state_store.init_from_dag(self.parallel_groups)
            self._feedback_state_store = state_store
        else:
            self._feedback_state_store = None

        agent_count = max(self.assignment.values()) if self.assignment else 1
        self.start_ai2thor(floor_plan=floor_plan, agent_count=agent_count)
        if task_description:
            self._init_checker(task_description, self.scene_name)

        groups_to_plans: Dict[int, List[SubtaskPlan]] = defaultdict(list)
        for sid, p in self.subtask_plans.items():
            groups_to_plans[p.parallel_group].append(p)

        print("=" * 60)
        print("[EXEC] Multi-Robot Execution (Feedback Mode: sequential per subtask)")
        if state_store is not None:
            print("[EXEC] Real-time state store: ON")
        if on_subtask_failed is not None:
            print("[EXEC] Immediate replan callback: ON")
        print("=" * 60)

        try:
            completed_groups = set()
            while True:
                # 매 반복마다 groups_to_plans를 최신 상태로 재구성
                groups_to_plans = defaultdict(list)
                for sid, sp in self.subtask_plans.items():
                    groups_to_plans[sp.parallel_group].append(sp)

                # 아직 완료되지 않은 그룹 중 가장 작은 gid 선택
                remaining = sorted(gid for gid in groups_to_plans.keys() if gid not in completed_groups)
                if not remaining:
                    break
                gid = remaining[0]
                plans = groups_to_plans[gid]

                # 이미 성공했거나 drop된 서브태스크 필터
                to_run = []
                for p in plans:
                    if p.subtask_id in self._dropped_subtasks:
                        print(f"[Subtask {p.subtask_id}] Dropped (physically impossible), skipping")
                        continue
                    prev = self._subtask_results.get(p.subtask_id)
                    if prev and prev.success:
                        print(f"[Subtask {p.subtask_id}] Already succeeded, skipping")
                        continue
                    to_run.append(p)

                if not to_run:
                    print(f"[Group {gid}] All subtasks already succeeded, skipping")
                    completed_groups.add(gid)
                    continue

                # ── 그룹 내 서브태스크 병렬 실행 ──
                print(f"\n[Group {gid}] Starting {len(to_run)} subtask(s) in parallel: {[p.subtask_id for p in to_run]}")
                threads = []
                for p in to_run:
                    t = threading.Thread(target=self._run_subtask, args=(p,), daemon=True)
                    t.start()
                    threads.append((t, p))

                # 모든 스레드 완료 대기
                for t, _ in threads:
                    t.join()

                # 남은 액션 큐 소진 대기
                self._drain_action_queue()

                # ── 그룹 완료 후 실패 서브태스크 처리 ──
                failed_plans = []
                for _, p in threads:
                    result = self._subtask_results.get(p.subtask_id)
                    if result and not result.success:
                        failed_plans.append(p)

                replanned = False
                for p in failed_plans:
                    if on_subtask_failed is None:
                        continue
                    try:
                        replan_success = on_subtask_failed(self._subtask_results[p.subtask_id])
                    except Exception as cb_e:
                        print(f"[Feedback] on_subtask_failed callback error: {cb_e}")
                        replan_success = False

                    if replan_success:
                        replanned = True
                        failed_sid = p.subtask_id
                        new_plan = self.subtask_plans.get(failed_sid)
                        if new_plan:
                            print(f"[Feedback] Re-executing subtask {failed_sid} with new plan ({len(new_plan.actions)} actions)")
                            self._subtask_failed[failed_sid] = False
                            t2 = threading.Thread(target=self._run_subtask, args=(new_plan,), daemon=True)
                            t2.start()
                            t2.join()
                            self._drain_action_queue()

                print(f"[Group {gid}] Completed")

                if replanned:
                    # 재계획 후 교체된 subtask_plans 기반으로 그룹 재구성
                    completed_groups.clear()
                    replanned_groups: Dict[int, list] = defaultdict(list)
                    for _s, _sp in self.subtask_plans.items():
                        replanned_groups[_sp.parallel_group].append(_sp)
                    for prev_gid, prev_plans in replanned_groups.items():
                        if all(
                            self._subtask_results.get(p.subtask_id)
                            and self._subtask_results[p.subtask_id].success
                            for p in prev_plans
                        ):
                            completed_groups.add(prev_gid)
                else:
                    completed_groups.add(gid)
            self._drain_action_queue()
            
            if self.checker is not None:
                try:
                    coverage = self.checker.get_coverage()
                    transport_rate = self.checker.get_transport_rate()
                    finished = self.checker.check_success()
                    balance = self._compute_balance_metric()
                    exec_rate = self._compute_exec_rate()
                    print("\n[[ RESULT ]]\n")
                    print(
                        f"Coverage:{coverage:.3f}, Transport Rate:{transport_rate:.3f}, "
                        f"Finished:{finished}, Balance:{balance:.3f}, Exec:{exec_rate:.3f}"
                    )
                    print("\n")
                    completed = sorted(getattr(self.checker, "subtasks_completed_numerated", []))
                    expected = sorted(getattr(self.checker, "subtasks", []))
                    if hasattr(self.checker, 'get_missing_subtasks'):
                        missing = self.checker.get_missing_subtasks()
                    else:
                        missing = sorted(set(expected) - set(completed))
                    print(f"[Checker] Completed ({len(completed)}): {completed}")
                    print(f"[Checker] Missing ({len(missing)}): {missing}")
                    if self._dropped_subtasks:
                        print(f"[Checker] Dropped (physically impossible): subtask IDs {sorted(self._dropped_subtasks)}")
                except Exception:
                    pass
            return dict(self._subtask_results)
        finally:
            self._feedback_state_store = None
            self.stop_ai2thor()

    # -----------------------------
    # 코드 생성
    # -----------------------------
    def run(
        self,
        task_idx: int = 0,
        task_name: str = "task",
        task_description: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """멀티 로봇 실행 코드를 생성하고 파일로 저장"""
        self.load_assignment(task_idx)
        self.load_subtask_dag(task_name)
        self.load_plan_actions()

        code = self._generate_execution_code(task_description=task_description)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(code)
            #print(f"[Executor] Saved execution code to: {output_path}")

        return code

    def _generate_execution_code(self, task_description: Optional[str] = None) -> str:
        """멀티 로봇 실행을 위한 파이썬 코드 생성"""
        lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated multi-robot execution code.",
            "Run with: python <this_file> --floor-plan <N>",
            '"""',
            "",
            "import argparse",
            "import sys",
            "import os",
            "",
            "import sys",
            "from pathlib import Path",
            "PDL_ROOT = Path(__file__).resolve().parents[2]",
            "sys.path.insert(0, str(PDL_ROOT))",
            'sys.path.insert(0, str(PDL_ROOT / "scripts"))',
            'sys.path.insert(0, str(PDL_ROOT / "resources"))',

            "# Add scripts folder to path",
            "base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
            "scripts_path = os.path.join(base_path, 'scripts')",
            "if scripts_path not in sys.path:",
            "    sys.path.insert(0, scripts_path)",
            "",
            "from MultiRobotExecutor import MultiRobotExecutor, SubtaskPlan",
            "",
            "# --- Robot Assignment ---",
            f"ASSIGNMENT = {self.assignment}  # subtask_id -> robot_id",
            f"PARALLEL_GROUPS = {self.parallel_groups}  # group_id -> [subtask_ids]",
            f"AGENT_COUNT = {getattr(self, 'configured_agent_count', None) or (max(self.assignment.values()) if self.assignment else 1)}",
            f"SPAWN_POSITIONS = {getattr(self, 'saved_spawn_positions', None)}  # LP에서 결정된 스폰 좌표",
            "",
            "# --- Subtask Plans ---",
            "SUBTASK_PLANS = {",
        ]

        for sid, plan in sorted(self.subtask_plans.items()):
            actions_repr = repr(plan.actions)
            lines.append(f"    {sid}: {{")
            lines.append(f"        'name': {repr(plan.subtask_name)},")
            lines.append(f"        'robot_id': {plan.robot_id},")
            lines.append(f"        'actions': {actions_repr},")
            lines.append(f"        'parallel_group': {plan.parallel_group},")
            lines.append(f"    }},")

        lines.append("}")
        lines.append("")
        # Always emit TASK_DESCRIPTION in generated files so checker wiring is explicit.
        lines.append(f"TASK_DESCRIPTION = {repr(task_description or '')}")
        lines.append("")

        lines.extend([
            "",
            "def main():",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--floor-plan', type=int, default=1)",
            "    args = parser.parse_args()",
            "    ",
            "    executor = MultiRobotExecutor(base_path)",
            "    executor.assignment = ASSIGNMENT",
            "    executor.parallel_groups = PARALLEL_GROUPS",
            "    ",
            "    # Reconstruct subtask_plans",
            "    for sid, data in SUBTASK_PLANS.items():",
            "        executor.subtask_plans[sid] = SubtaskPlan(",
            "            subtask_id=sid,",
            "            subtask_name=data['name'],",
            "            robot_id=data['robot_id'],",
            "            actions=data['actions'],",
            "            parallel_group=data['parallel_group'],",
            "        )",
            "    ",
            "    # Execute in AI2-THOR",
            "    executor.execute_in_ai2thor(",
            "        floor_plan=args.floor_plan,",
            "        task_description=TASK_DESCRIPTION if 'TASK_DESCRIPTION' in globals() else None,",
            "        agent_count=AGENT_COUNT,",
            "        spawn_positions=SPAWN_POSITIONS,",
            "    )",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ])

        return "\n".join(lines)

    # -----------------------------
    # Pipeline 편의를 위한 함수
    # -----------------------------
    def run_and_execute(
        self,
        task_idx: int = 0,
        task_name: str = "task",
        floor_plan: int = 1,
        agent_count: Optional[int] = None,
        spawn_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ):
        """Load and execute directly."""
        print("\n" + "=" * 60)
        print("MultiRobotExecutor (Load + Execute)")
        print("=" * 60)

        self.load_assignment(task_idx)
        self.load_subtask_dag(task_name)
        self.load_plan_actions()

        self.execute_in_ai2thor(floor_plan=floor_plan, agent_count=agent_count,
                                spawn_positions=spawn_positions)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--task-name", type=str, default="task")
    parser.add_argument("--floor-plan", type=int, required=True)
    args = parser.parse_args()

    ex = MultiRobotExecutor(args.base_path)
    ex.run_and_execute(
        task_idx=args.task_idx,
        task_name=args.task_name,
        floor_plan=args.floor_plan,
    )


if __name__ == "__main__":
    main()
