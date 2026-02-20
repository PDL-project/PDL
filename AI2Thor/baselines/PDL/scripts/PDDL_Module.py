import copy
from email import parser
import glob
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import time
import re
import shutil
from typing import List, Dict, Tuple, Optional, Union, Any

import openai
import ai2thor.controller

import difflib
import sys

# 현재 실행 중인 파일의 절대 경로를 기준으로 폴더 위치 계산
SCRIPT_DIR = Path(__file__).resolve().parent

# scripts/의 부모 폴더(=repo)의 resources 폴더를 가리키도록 구성
RESOURCES_DIR = SCRIPT_DIR.parent / "resources"

# scripts/의 부모의 부모 폴더(=repo/..)의 utils 폴더를 가리키도록 구성
UTILS_DIR = SCRIPT_DIR.parent.parent / "utils"

if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import actions # resources/actions.py (로봇 액션 정의 등)
import robots  # resources/robots.py (로봇 스킬/질량 정보 등)

from DAG_Module import DAGGenerator # plan 기반 DAG(병렬성) 생성 모듈
from LP_Module import assign_subtasks_cp_sat, binding_pairs_from_subtask_dag # 작업할당(CP-SAT) 및 DAG 바인딩 추출

from MultiRobotExecutor import MultiRobotExecutor, SubTaskExecutionResult # 멀티 로봇 실행 코드 생성/실행 관리
from auto_config import AutoConfig # config 로딩/세팅 자동화

from FeedbackLoopModule import (
    load_subtask_dag_edges,
    load_subtask_dag_effects,
    load_subtask_dag_parallel_groups,
    load_dependency_groups_from_dag,
    PartialReplanner,
    GroupAgent,
    subtask_has_dependency,
    SubtaskManagerLLM,
    run_planner_for_one_subtask,
    SharedTaskStateStore,
    sync_execution_results_to_store,
    format_success_effects_for_prompt,
    build_local_env_for_group,
    local_env_to_str,
)

DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0 # 실험 재현성을 위해 0 고정(랜덤성 최소)
DEFAULT_RETRY_DELAY = 20
MAX_RETRIES = 3

def _parse_floorplan_number(scene_name: str) -> int:
    """
    유틸 함수: "FloorPlan15" 같은 문자열에서 15를 추출
    예: "FloorPlan15" -> 15
    AI2-THOR scene 이름 문자열에서 FloorPlan 번호만 뽑는다.
    """
    match = re.search(r"FloorPlan(\d+)", scene_name)
    if not match:
        raise ValueError(f"Invalid scene name (expected FloorPlan<number>): {scene_name}")
    return int(match.group(1))

def _available_robot_skills_from_ids(robot_ids: List[int]) -> List[str]:
    """
    유틸 함수: 로봇 id 리스트를 입력받아, 해당 로봇들이 가진 스킬(action) 합집합 반환

    robot_ids가 [1,2,3]이면 robots.robots[0], [1], [2]의 skills를 합쳐서 정렬해 반환
    (robots.py에서 robots는 1-based id를 0-based index로 접근해야 하므로 r_id - 1)
    """
    skill_set = set()
    for r_id in robot_ids:
        rob = robots.robots[r_id - 1]
        skill_set.update(rob["skills"])
    return sorted(skill_set)

class PDDLError(Exception):
    """Base exception class for PDDL-related errors."""
    pass

class LLMError(Exception):
    """LLM-related errors (API key, model calls, etc.)."""
    pass

class PDDLUtils:
    """
    AI2-THOR 환경에서 객체 불러와서, llm이나 pddl이 활용하기 좋은 형태로 변환해주는 namespace
    """
    
    @staticmethod #인스턴트 생성x, 그냥 사용하면 됨, 클래스 내부 상태와 상관없이 실행되는 함수라는 의미
    def convert_to_dict_objprop(objs: List[str], obj_mass: List[float]) -> List[Dict[str, Union[str, float]]]:
        """
        객체 리스트(사과, 칼), 각 객체의 질량(0.2, 0.5)을 zip으로 이름과 질량을 묶어서 딕셔너리 리스트로 변환해주는 함수

        llm이나, pddl에 넣기 좋은 형태가 딕셔너리 리스트라 그걸로 바꿔주기 위한 함수다!
        """
        return [{'name': obj, 'mass': mass} for obj, mass in zip(objs, obj_mass)]
    
    @staticmethod
    def get_ai2_thor_objects(floor_plan: int) -> List[Dict[str, Any]]:
        """
        입력 : floor_plan -> 15, 201 같은 데이터셋 숫자

        출력 : objects_ai가 될 형식임
        [
            {'name': 'Apple', 'mass': 0.2},
            {'name': 'Knife', 'mass': 0.5},
            {'name': 'Fridge', 'mass': 50.0},
            ...
        ] 
        """
        controller = None
        try:
            controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan}")
            objects_ai = []

            # 마지막 이벤트의 metadata에서 objects 목록을 순회
            for obj in controller.last_event.metadata["objects"]:
                name = obj["objectType"]
                mass = obj.get("mass", 0.0)

                # parentReceptacles: 현재 오브젝트가 어떤 receptacle(서랍/선반/테이블 등) 위/안에 있는지 정보
                parents = obj.get("parentReceptacles")
                if parents:
                    # ID는 "CounterTop|+00.00|..." 같은 형태이므로 '|' 앞부분만 가져와서 이름으로 씀
                    locations = [p.split("|")[0] for p in parents]
                else:
                    locations = ["Floor"]

                # 위치 정보(없으면 0으로 채움)
                position = obj.get("position", {})
                objects_ai.append({
                    "name": name,
                    "mass": mass,
                    "locations": locations,
                    "position": {
                        "x": position.get("x", 0.0),
                        "y": position.get("y", 0.0),
                        "z": position.get("z", 0.0)
                    }
                })

            return objects_ai

        finally:
            # 컨트롤러 종료
            if controller:
                controller.stop()

class FileProcessor:
    """
    파일 입출력
    PDDL 텍스트 정리 매니저

    - LLM이 한 번에 생성한 큰 PDDL 텍스트를 문제(서브테스크)별로 쪼개서 저장
    - 도메인 이름 추출
    - planner(FastDownward) 출력에서 plan 라인만 추출
    - 폴더 청소
    - LLM이 만든 "summary+sequence" 텍스트를 subtask 블록으로 파싱
    """
    
    def __init__(self, base_path: str):
        """file processor 생성자
        
        Args:
            base_path (str): 기본 루트
        """
        self.base_path = base_path

        #LLM이 만든 pddl problem을 쪼개서 저장하는 곳(검증전)
        self.subtask_path = os.path.join(base_path, "resources", "generated_subtask")

        #pddl problem을 검증한걸 저장하는 곳(검증후)
        self.validated_subtask_path = os.path.join(base_path, "resources", "validated_subtask")
        
        # 매 실행할때마다 실행별 기록을 저장하는 곳
        self.each_run_path = os.path.join(base_path, "resources", "each_run")

        #각 서브테스크에 대한 pddl problems 저장해놓는 곳
        self.subtask_pddl_problems_path = os.path.join(base_path, "resources", "subtask_pddl_problems")
        os.makedirs(self.subtask_pddl_problems_path, exist_ok=True)

        # 서브테스크별 fast-downward plan(action list)을 저장하는 폴더
        self.subtask_pddl_plans_path = os.path.join(base_path, "resources", "subtask_pddl_plans")
        os.makedirs(self.subtask_pddl_plans_path, exist_ok=True)

        # 서브테스크별 “precondition/goal 텍스트”를 저장하는 폴더(문제 생성 전 단계)
        self.precondition_subtasks_path = os.path.join(base_path, "resources", "precondition_subtasks")
        os.makedirs(self.precondition_subtasks_path, exist_ok=True)

        os.makedirs(self.subtask_path, exist_ok=True)
        os.makedirs(self.validated_subtask_path, exist_ok=True) 
        os.makedirs(self.each_run_path, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """
        텍스트 파일을 읽어서 문자열로 반환 함수
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise PDDLError(f"파일을 찾을수없음: {file_path}")
        except Exception as e:
            raise PDDLError(f"파일을 읽을수없음 {file_path}: {str(e)}")
    
    def write_file(self, file_path: str, content: str) -> None:
        """
        문자열을 파일로 저장
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
        except Exception as e:
            raise PDDLError(f"파일 생성 실패 {file_path}: {str(e)}")
    
    def normalize_pddl(self, text: str) -> str: #이거 함수 지우면 안댐!@@@
        """
        LLM 출력이 ```pddl ... ``` 같은 마크다운/잡텍스트를 포함하는 경우가 많아서
        FastDownward에 넣기 전에 정리하는 함수.

        동작:
            - ```pddl, ``` , ` 제거
            - (define부터 시작하도록 앞부분 잘라내기
            - '#'로 시작하는 라인은 주석으로 보고 제거
        """
        if not text:
            return ""
        text = text.replace("```pddl", "").replace("```", "").replace("`", "")
        # (define부터 시작하도록 자르기
        i = text.find("(define")
        if i != -1:
            text = text[i:]
        # '#'로 시작하는 라인 제거
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        return "\n".join(lines).strip()

    def balance_parentheses(self, content: str) -> str:
        """
        LLM이 만든 pddl이 종종 괄호가 깨져서, 첫번째 완전한 괄호 덩어리만 잘라서 반환해주는 함수
        """
        open_count = 0
        start_index = -1
        end_index = -1
        
        for i, char in enumerate(content):
            if char == '(':
                if open_count == 0:
                    start_index = i
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count == 0:
                    end_index = i
                    break
        
        if start_index != -1 and end_index != -1:
            return content[start_index:end_index+1]
        return ""
        
    def extract_domain_name(self, problem_file_path: str) -> Optional[str]:
        """
        PDDL problem 파일 안에서 (:domain xxx)를 찾아 xxx를 반환해주는 함수
        """
        try:
            domain_pattern = re.compile(r'\(\s*:domain\s+(\S+)\s*\)')
            content = self.read_file(problem_file_path)
            match = domain_pattern.search(content)
            return match.group(1) if match else None
        except Exception as e:
            print(f"도메인 이름 추출 못함 {problem_file_path}: {str(e)}")
            return None

    def find_domain_file(self, domain_name: str) -> Optional[str]:
        """
        resources/<domain_name>.pddl 파일이 있나 확인해서 path 반환해줌
        """
        try:
            domain_path = os.path.join(self.base_path, "resources", f"{domain_name}.pddl")
            return domain_path if os.path.isfile(domain_path) else None
        except Exception as e:
            print(f"도메인 파일 찾기 실패함 {domain_name}: {str(e)}")
            return None

    def clean_directory(self, directory_path: str) -> None:
        """
        특정 파일 내용 싹 비우는 함수(파일 삭제, 폴더 삭제)
        """
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def extract_plan_from_output(self, content: str) -> str:
        """
        fast-downward 출력이 긴데, 그 중에서 실제로 plan처럼 생긴 줄만 정규식으로 골라서 합쳐주는 함수
        """
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content provided to extract_plan_from_output")
            
        try:
            plan_pattern = re.compile(r"^\s*\w+\s+\w+\s+\w+\s+\(\d+\)\s*$", re.MULTILINE)
            plan = plan_pattern.findall(content)
            return "\n".join(plan) if plan else ""
        except Exception as e:
            print(f"결과에서 추출할때 에러남: {str(e)}")
            return ""

class LLMHandler:
    """
    LLM한테 입력값 보내고, 답변 받아오는 LLM 담당자 객체
    """
    
    def __init__(self, api_key_file: str):
        """LLMHandle 초기설정
        
        Args:
            api_key_file (str): api_key 있는 파일
        """
        self.setup_api(api_key_file)
    
    def setup_api(self, api_key_file: str) -> None:
        """
        OpenAI API key 파일을 읽어 openai.api_key에 설정한다.

        로직:
            1) api_key_file + ".txt"
            2) api_key_file
        """
        try:
            # 가능한 경로 후보를 여러 개 준비(실행 위치가 바뀌는 경우 대비)
            candidates = [
                Path(api_key_file + ".txt"),
                Path(api_key_file),
                # fallback: AI2Thor/baselines/PDL/{api_key_file(.txt)}
                (SCRIPT_DIR.parent / (api_key_file + ".txt")),
                (SCRIPT_DIR.parent / api_key_file),
            ]
            api_key = None
            used_path = None
            for p in candidates:
                try:
                    api_key = p.read_text().strip()
                    if not api_key:
                        raise ValueError("API key file is empty")
                    used_path = p
                    break
                except FileNotFoundError:
                    continue

            if not api_key:
                raise LLMError(
                    f"API key file not found: {api_key_file} or {api_key_file}.txt (also checked PDL folder)"
                )

            openai.api_key = api_key
            print("Successfully loaded API key from", used_path)
        except Exception as e:
            raise LLMError(f"Error reading API key file: {str(e)}")
    
    #프롬프트 보내고 텍스트 받기 함수
    def query_model(
        self, 
        prompt: Union[str, List[Dict]], 
        gpt_version: str, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1,
        frequency_penalty: float = 0
    ) -> Tuple[dict, str]:
        """
        
        Args:
            prompt: 문자열 혹은 dic
            gpt_version: 사용할 모델명
            max_tokens: gpt 답변 최대 토큰 수
            temperature: 랜덤성/창의성(0은 거의 항상 같은 값, 1은 항상 다른값, 실험을 위해 항상 0으로 설정해야함)
            stop: 특정 문자열이 나오면 항상 멈춤
            logprobs: 각 토큰이 얼마나 확률 높게 생성됐는지에 대한 정보
            frequency_penalty: 같은 단어나 구문 반복하지 않도록하는 패널티(0은 반복허용, 커질수록 반복 줄어듬 근데 너무 크면 문장이 부자연스러워짐)
         
        Returns:
            Tuple of (full response object, generated text)
            response, text = query_model(...) / response= ai의 원본 응답 번체, text는 우리가 실제로 쓰는 답변 문자열
            
            text -> pddl생성, 코드생성
            response -> 토큰 사용량 체크 등
        """
        retry_delay = DEFAULT_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES): #api 연결시도(현재 3번으로 설정됨)
            try:
                if "gpt" not in gpt_version: #모델명에 gpt가 없을경우 -> completion 스타일로 생성
                    response = openai.completions.create(
                        model=gpt_version, 
                        prompt=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        stop=stop, 
                        logprobs=logprobs, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].text.strip()
                else: #모델명에 gpt가 없을경우 -> chat 스타일로 생성
                    response = openai.chat.completions.create(
                        model=gpt_version, 
                        messages=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].message.content.strip()
                    
            except openai.RateLimitError: #ai한테 너무 빨리, 많이 요청해서 제한 걸렸을때 예외처리
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise LLMError("Rate limit exceeded")
                
            except (openai.APIError, openai.APITimeoutError) as e: #서버에러 또는 타임아웃 예외처리
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    continue
                raise LLMError(f"API Error after all retries: {str(e)}")
                
            except Exception as e:
                raise LLMError(f"Unexpected error in LLM query: {str(e)}")

class TaskManager:
    """ 
    파이프라인 총괄 매니저, 여러 모듈들 관리
    """
    
    def __init__(self, base_path: str, gpt_version: str, api_key_file: str):
        """task manager 생성자, 초기 설정
        """
        self.base_path = base_path
        self.gpt_version = gpt_version

        # 핵심 모듈 생성
        self.llm = LLMHandler(api_key_file) #모델 호출 모듈
        self.file_processor = FileProcessor(base_path) # 파일 읽기, 쓰기, pddl 테스크 분리, 정규식 파싱 모듈
        
        # 파일경로 설정, log저장용 폴더 만들기 코드
        self.resources_path = os.path.join(base_path, "resources") # 가져올 리소스 폴더 위치 설정
        self.logs_path = os.path.join(".", "logs")  # 로그 폴더 생성
        os.makedirs(self.logs_path, exist_ok=True)
        
        # subtask 폴더 비우기
        self.clean_all_resources_directories()
        
        # 결과 저장소 초기화
        self.decomposed_plan: List[str] = []
        self.parsed_subtasks: List[List[dict]] = []
        self.precondition_subtasks: List[List[dict]] = []

        self.subtask_pddl_problems: List[List[dict]] = []
        self.subtask_pddl_plans: List[str] = []

        self.subtask_dag = None
        self.task_assignment = None

        self.objects_ai = None

    def clean_all_resources_directories(self) -> None:
        """
        resources/. 아래 생성된 폴더들의 파일들 전부 지우는 함수
        """
        target_dirs = [
            "generated_subtask",
            "precondition_subtasks",
            "subtask_pddl_problems",
            "validated_subtask",
            "subtask_pddl_plans",
            "dag_outputs",
        ]

        for dir_name in target_dirs:
            directory = os.path.join(self.resources_path, dir_name)

            try:
                if not os.path.exists(directory):
                    continue

                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"[CLEAN ERROR] Failed to remove {file_path}: {e}")

            except Exception as e:
                print(f"[CLEAN ERROR] Failed to access directory {directory}: {e}")
    
    def log_results(self, task: str, idx: int, available_robots: List[dict], 
                   gt_test_tasks: List[str], trans_cnt_tasks: List[int], 
                   min_trans_cnt_tasks: List[int], objects_ai: str,
                   bddl_file_path: Optional[str] = None):

        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        task_name = "_".join(task.split()).replace('\n', '')
        folder_name = f"{task_name}_plans_{date_time}"
        log_folder = os.path.join(self.logs_path, folder_name)
        
        #print(f"Creating log folder: {log_folder}")
        os.makedirs(log_folder)

        # 1) 서브테스크 PDDL problem 저장
        subtask_pddl_dir = os.path.join(log_folder, "subtask_pddl_problems")
        os.makedirs(subtask_pddl_dir, exist_ok=True)

        for item in self.subtask_pddl_problems[idx]:
            sid = item.get("subtask_id", "unknown")
            title = item.get("subtask_title", "untitled")
            pddl_text = item.get("problem_text", "")

            safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
            filename = f"subtask_{sid:02d}_{safe_title}.pddl"

            path = os.path.join(subtask_pddl_dir, filename)
            self.file_processor.write_file(path, pddl_text)
        
        # 2) precondition/goal 텍스트 저장
        precondition_dir = os.path.join(log_folder, "precondition_subtasks")
        os.makedirs(precondition_dir, exist_ok=True)

        for item in self.precondition_subtasks[idx]:
            sid = item.get("subtask_id", "unknown")
            title = item.get("subtask_title", "untitled")
            pre_text = item.get("pre_goal_text", "")

            safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
            filename = f"pre_{sid:02d}_{safe_title}.txt"

            path = os.path.join(precondition_dir, filename)
            self.file_processor.write_file(path, pre_text)

        # 3) validator 결과(problem) 저장
        val_subtask_pddl_dir = os.path.join(log_folder, "val_subtask_pddl_problems")
        os.makedirs(val_subtask_pddl_dir, exist_ok=True)

        src_val_dir = self.file_processor.validated_subtask_path  # resources/validated_subtask
        for file_name in os.listdir(src_val_dir):
            if file_name.endswith(".pddl"):
                src_path = os.path.join(src_val_dir, file_name)
                dst_path = os.path.join(val_subtask_pddl_dir, file_name)
                shutil.copy(src_path, dst_path)

        # 4) fast-downward plan(action list) 저장
        plan_action_pddl_dir = os.path.join(log_folder, "subtask_pddl_plans")
        os.makedirs(plan_action_pddl_dir, exist_ok=True)

        plans_dir = self.file_processor.subtask_pddl_plans_path  # resources/validated_subtask
        for file_name in os.listdir(plans_dir):
            if file_name.endswith("actions.txt"):
                main_path = os.path.join(plans_dir, file_name)
                sub_path = os.path.join(plan_action_pddl_dir, file_name)
                shutil.copy(main_path, sub_path)
        
        # 5) DAG 산출물 저장(json/png 복사)
        dag_output_folder = os.path.join(log_folder, "dag_outputs")
        os.makedirs(dag_output_folder, exist_ok=True)
        source_dag_folder = os.path.join(self.resources_path, "dag_outputs")
        if os.path.exists(source_dag_folder):
            for file_name in os.listdir(source_dag_folder):
                full_file_name = os.path.join(source_dag_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dag_output_folder)

        try:
            # decomposed_plan, validated_plan 텍스트 저장
            self._write_plan(log_folder, "decomposed_plan.py", self.decomposed_plan[idx])
            self._write_plan(log_folder, "validated_plan.py", self.validated_plan[idx])
            
            # 주요 로그 텍스트 기록(log.txt)
            with open(os.path.join(log_folder, "log.txt"), 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {self.gpt_version}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {available_robots[idx]}")
                f.write(f"\nground_truth = {gt_test_tasks[idx]}")
                f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
                f.write(f"\nmin_trans = {min_trans_cnt_tasks[idx]}")
            
            # generated_subtask 복사
            subtask_folder = os.path.join(log_folder, "generated_subtask")
            os.makedirs(subtask_folder)
            source_folder = os.path.join(self.resources_path, "generated_subtask")
            for file_name in os.listdir(source_folder):
                full_file_name = os.path.join(source_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, subtask_folder)
            
            # validated_subtask 복사
            validated_subtask_folder = os.path.join(log_folder, "validated_subtask")
            os.makedirs(validated_subtask_folder)
            source_validated_folder = os.path.join(self.resources_path, "validated_subtask")
            for file_name in os.listdir(source_validated_folder):
                full_file_name = os.path.join(source_validated_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, validated_subtask_folder)

        except Exception as e:
            print(f"Error writing plans for task {idx + 1}: {str(e)}")

    def _write_plan(self, folder: str, filename: str, content: Union[str, List]):
        """
        내용을 파일로 저장하는 헬퍼 함수.
        - content가 list면 여러 파일로 쪼개서 저장
        - content가 str이면 그대로 저장
        """
        if isinstance(content, list):
            for i, item in enumerate(content):
                with open(os.path.join(folder, f"{filename}.{i}"), 'w') as f:
                    f.write(str(item))
        else:
            with open(os.path.join(folder, filename), 'w') as f:
                f.write(content)
    
    def _available_robot_skills(self, _: List[int]) -> List[str]:
        """현재 구현은 입력 로봇 id 리스트를 무시하고
        robots.robots[0]의 skills만 반환한다.
        """
        return sorted(robots.robots[0]["skills"])

    def process_tasks(self, test_tasks: List[str], robot_ids: List[List[int]], objects_ai: str,
                       floor_plan: Optional[int] = None, run_with_feedback: bool = False, max_replan_retries: int = 2) -> None:
        """
        TaskManager 전체 파이프라인의 main함수
        test_tasks -> 자연어 task 리스트
        robot_ids -> 로봇 ID들
        objects_ai -> 추출한 오브젝트 이름과 무게
        floor_plan -> AI2-THOR FloorPlan 번호 (거리 기반 LP 최적화 시 사용, None이면 거리 비활성) (run_with_feedback=True일 때 필요)

        run_with_feedback -> True면 실행 후 실패 시 Subtask/Central LLM으로 재계획 후 재실행
        max_replan_retries -> 피드백 루프 최대 재시도 횟수
        """
        try:
            # 몇개의 task를 처리할건지 확인하는 부분
            print(f"\n[DIAGNOSTIC] Initial Task Count: {len(test_tasks)}")
            
            # objects_ai 정보 저장하기(다른 함수에서 쓰기 위해서)
            self.objects_ai = objects_ai

            # 로봇 스킬 목록(현재 구현은 robots[0]만 반환하는 함수 적용)
            self.available_robot_skills = self._available_robot_skills(robot_ids)

            # 결과 리스트 초기화
            self.decomposed_plan = []
            self.parsed_subtasks = []
            self.precondition_subtasks =[]
            self.subtask_pddl_problems = []
            self.validated_plan = []
            self.subtask_pddl_plans = []
            
            # PDDL 도메인 파일 가져오기
            #(allactionrobot.pddl ->로봇이 어떤 행동을 할수 있고, 어떤 조건이 필요하고, 행동하면 머가 변하는지를 정의한 규칙정리서)
            allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            domain_content = self.file_processor.read_file(allaction_domain_path)
            
            # task 단위로 하나씩 돌리기
            for task_idx, (task, task_robot_ids) in enumerate(zip(test_tasks, robot_ids)):
                print(f"\n{'='*50}")
                print(f"Processing Task: {task}: {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                
                # 이전 task 흔적 지우기
                self.clean_all_resources_directories()
                
                # 1. task 분해, decomposed_plan 생성 -> 자연어 task를 여러개의 subtask로 쪼개기 + 필요한 스킬과 오브젝트 선정
                decomposed_plan = self._generate_decomposed_plan(task, domain_content, self.available_robot_skills, objects_ai)
                self.decomposed_plan.append(decomposed_plan)
                
                print("✓ Decomposed plan generated")
                print("decomposed plan:\n", decomposed_plan)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                
                parsed_subtasks = self._decomposed_plan_to_subtasks(decomposed_plan) #분리
                print("✓ Parsed Decomposed Plan generated")
                print("parsed decomposed plan:\n", parsed_subtasks)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                precondition_subtasks = self._generate_precondition_subtasks(parsed_subtasks, domain_content, self.available_robot_skills, objects_ai)
                self.precondition_subtasks.append(precondition_subtasks) 
                print("✓ Precondition Decomposed Plan generated")
                print("Precondition Decomposed Plan:\n", precondition_subtasks)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                for item in precondition_subtasks:
                    sid = item.get("subtask_id", -1)
                    title = item.get("subtask_title", "untitled")
                    text = item.get("pre_goal_text", "")

                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"pre_{sid:02d}_{safe_title}.txt"   # 확장자 txt 추천 (PDDL problem이 아니라서)

                    out_path = os.path.join(self.file_processor.precondition_subtasks_path, filename)
                    self.file_processor.write_file(out_path, text)

                # 2. 서브테스크에 대한 pddl problem 정의
                subtask_pddl_problems = self._generate_subtask_pddl_problems(precondition_subtasks, domain_content, self.available_robot_skills, objects_ai)
                self.subtask_pddl_problems.append(subtask_pddl_problems)

                print("✓ PDDL problems generated")
                print("PDDL problems plan:\n", subtask_pddl_problems)
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                for item in subtask_pddl_problems:
                    sid = item["subtask_id"]
                    title = item["subtask_title"]
                    pddl_text = item["problem_text"]

                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"subtask_{sid:02d}_{safe_title}.pddl"

                    out_path = os.path.join(self.file_processor.subtask_pddl_problems_path, filename)
                    self.file_processor.write_file(out_path, pddl_text)

                # 3. FastDownward 돌려서, pddl plan 생성
                validated_plan = self._validate_and_plan()

                # 4. pddl plan기반 DAG 생성
                self.generate_dag() #DAG 생성 (병렬성 분석)

                # 5a. 시뮬레이터에서 로봇 스폰 좌표 + 오브젝트 좌표 가져오기 (거리 기반 LP용)
                robot_positions = None
                object_positions = None
                if floor_plan is not None:
                    robot_positions, object_positions = MultiRobotExecutor.spawn_and_get_positions(
                        floor_plan, len(task_robot_ids)
                    )
                    self.robot_spawn_positions = robot_positions
                    print(f"✓ Robot spawn positions: {robot_positions}")

                # 5b. LP 작업할당
                plan_actions_by_sid = self._load_plan_actions_by_subtask_id()

                binding_pairs = binding_pairs_from_subtask_dag(self.subtask_dag)

                assignment = assign_subtasks_cp_sat(
                    subtasks=parsed_subtasks,  # _decomposed_plan_to_subtasks() 결과 (id, skills 들어있어야 함)
                    robot_ids=task_robot_ids,  # 예: [1,2,3]
                    robots_db=robots.robots,   # robots.py의 robots 리스트
                    plan_actions_by_subtask=plan_actions_by_sid,
                    objects_ai=self.objects_ai,  # 저장해둔 objects_ai 문자열/리스트
                    binding_pairs=binding_pairs,
                    robot_positions=robot_positions,
                    object_positions=object_positions,
                )

                # 작업 할당 결과 출력
                print("\n" + "="*50)
                print("✓ LP Task Allocation Result")
                print("="*50)
                for sid, rid in sorted(assignment.items()):
                    subtask_title = next((st["title"] for st in parsed_subtasks if st["id"] == sid), f"Subtask {sid}")
                    print(f"  Subtask {sid} ({subtask_title}) -> Robot {rid}")
                print("="*50)

                # 할당 결과 저장
                self.task_assignment = assignment
                assignment_output = {
                    "task_idx": task_idx,
                    "agent_count": len(task_robot_ids),
                    "assignment": {str(k): v for k, v in assignment.items()},
                    "subtasks": [{"id": st["id"], "title": st["title"], "robot": assignment.get(st["id"])} for st in parsed_subtasks]
                }
                # 스폰 좌표가 있으면 저장 (실행 시 동일 위치 재배치용)
                if robot_positions is not None:
                    assignment_output["robot_spawn_positions"] = {
                        str(k): list(v) for k, v in robot_positions.items()
                    }
                assignment_path = os.path.join(self.resources_path, "dag_outputs", f"task_{task_idx}_assignment.json")
                with open(assignment_path, "w") as f:
                    json.dump(assignment_output, f, indent=2, ensure_ascii=False)
                print(f"✓ Assignment saved to: {assignment_path}")

                # 6. 멀티로봇 실행 코드 생성
                print("\n[Step 6] Generating multi-robot execution code...")
                executor = MultiRobotExecutor(self.base_path)
                execution_code = executor.run(
                    task_idx=task_idx,
                    task_name="task",
                    task_description=task,
                    output_path=os.path.join(self.resources_path, "dag_outputs", f"task_{task_idx}_execution.py")
                )
                print("✓ Multi-robot execution code generated")

                # 7. 피드백 루프: 실행 후 실패 시 Subtask Manager LLM / Central LLM으로 재계획 후 재실행
                if run_with_feedback and floor_plan is not None:
                    print("\n[Step 7] Running execution with feedback loop...")
                    task_name_fb = "task"
                    state_store = SharedTaskStateStore(self.base_path, task_name_fb)
                    replan_retry_count = [0]

                    def _on_subtask_failed(failed_result):
                        """재계획 시도 후 새 플랜 로드 여부 반환."""
                        if replan_retry_count[0] >= max_replan_retries:
                            print(f"[Feedback] Max replan retries ({max_replan_retries}) reached, skipping replan")
                            return False

                        print(f"[Feedback] Immediate replan triggered by subtask {failed_result.subtask_id}")
                        replan_retry_count[0] += 1

                        current_results = dict(executor._subtask_results)
                        dag_effects = load_subtask_dag_effects(self.base_path, task_name_fb)
                        effects_for_success = {
                            sid: dag_effects.get(sid, [])
                            for sid, r in current_results.items()
                            if r.success and dag_effects
                        }
                        sync_execution_results_to_store(
                            state_store, current_results, effects_by_subtask_id=effects_for_success
                        )

                        replan_result = self.run_feedback_replan(
                            task_idx=task_idx,
                            task_name=task_name_fb,
                            execution_results=current_results,
                            domain_content=domain_content,
                            objects_ai=objects_ai,
                            state_store=state_store,
                            task_robot_ids=task_robot_ids,
                            floor_plan=floor_plan,
                        )

                        if replan_result == "no_change":
                            print("[Feedback] Replan: no_change")
                            return False
                        if replan_result == "fully_replanned":
                            print("[Feedback] Replan: fully_replanned")
                            success = self.reload_executor_with_integrated_dag(
                                executor=executor,
                                task_idx=task_idx,
                                task_name=task_name_fb
                            )
                            return bool(success)
                        return False

                    for retry in range(max_replan_retries + 1):
                        results = executor.execute_in_ai2thor_with_feedback(
                            floor_plan,
                            task_name=task_name_fb,
                            state_store=state_store,
                            on_subtask_failed=_on_subtask_failed,  # ← 콜백 연결
                        )
                        # 실행 완료 후 최종 상태 스토어 반영
                        dag_effects = load_subtask_dag_effects(self.base_path, task_name_fb)
                        effects_for_success = {
                            sid: dag_effects.get(sid, [])
                            for sid, r in results.items()
                            if r.success and dag_effects
                        }
                        sync_execution_results_to_store(
                            state_store, results, effects_by_subtask_id=effects_for_success
                        )
                        failed = [sid for sid, r in results.items() if not r.success]
                        if not failed:
                            print("[Feedback] All subtasks succeeded.")
                            break
                        if retry >= max_replan_retries:
                            print("[Feedback] Max replan retries reached.")
                            break
                        replan_result = self.run_feedback_replan(
                            task_idx=task_idx,
                            task_name=task_name_fb,
                            execution_results=results,
                            domain_content=domain_content,
                            objects_ai=objects_ai,
                            state_store=state_store,
                            task_robot_ids=task_robot_ids,
                            floor_plan=floor_plan,
                        )
                        if replan_result == "no_change":
                            break
                        elif replan_result == "fully_replanned":
                            success = self.reload_executor_with_integrated_dag(
                                executor=executor,
                                task_idx=task_idx,
                                task_name=task_name_fb
                            )
                            
                            if not success:
                                print("[Feedback] Failed to reload executor")
                                break
                            print("[Feedback] Ready for re-execution with integrated DAG")
                        else:
                            executor.load_plan_actions()
                    print("✓ Feedback loop finished.")

        except Exception as e:
            print(f"\n[ERROR] Task Processing Failed:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Current task index: {task_idx if 'task_idx' in locals() else 'Not started'}")
            raise

    def _load_plan_actions_by_subtask_id(self) -> Dict[int, List[str]]:
        plans_dir = self.file_processor.subtask_pddl_plans_path
        out: Dict[int, List[str]] = {}

        for fname in os.listdir(plans_dir):
            if not fname.endswith("_actions.txt"):
                continue
            m = re.match(r"subtask_(\d+)_.*_actions\.txt$", fname)
            if not m:
                continue
            sid = int(m.group(1))
            with open(os.path.join(plans_dir, fname), "r") as f:
                out[sid] = [ln.strip() for ln in f.readlines() if ln.strip()]
        return out


    def _generate_decomposed_plan(self, task: str, domain_content: str, robots: List[dict], objects_ai: str) -> str:
        """하나의 자연어 task를 subtask로 나누는 함수"""
        prompt = "" 
        try:
            # decomposition prompt file 불러오기
            #decompose_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"{self.prompt_decompse_set}.py")
            decompose_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_decompose.py")
            decompose_prompt = self.file_processor.read_file(decompose_prompt_path)
            
            #Construct the prompt incrementally like the original
            #prompt = f"from pddl domain file with all possible AVAILABLE ROBOT SKILLS: \n{domain_content}\n\n"

            
            prompt += "The following list is the ONLY set of objects that exist in the current environment.\n"
            prompt += "When writing subtasks and actions, you MUST ground every referenced object to this list.\n"
            prompt += "If the task mentions something not present, solve it using the closest available objects from the list.\n"
            prompt += f"\nENVIRONMENT OBJECTS = {objects_ai}\n\n"
            prompt += "If you reference an object not in ENVIRONMENT OBJECTS, your answer will be considered INVALID.\n\n\n"

            prompt += f"\nAVAILABLE ROBOT SKILLS = {robots}\n\n"
            prompt += "You are NOT given specific robots. You are only given the set of skills that are currently available.\n"
            prompt += "You MUST use only actions whose names are included in AVAILABLE ROBOT SKILLS.\n\n\n"
            
            prompt += "The following is an example of the expected output format.\n\n\n"
            prompt += decompose_prompt
            prompt += "# GENERAL TASK DECOMPOSITION \n"
            prompt += "Decompose and parallel subtasks where ever possible.\n\n"
            prompt += f"# Task Description: {task}"
            
            if "gpt" not in self.gpt_version:
                _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=3000, stop=None, frequency_penalty=0.0)
            else:
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=3000, frequency_penalty=0.0)
            
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating decomposed plan: {str(e)}")
    
    def _decomposed_plan_to_subtasks(self, decomposed_text: str) -> List[Dict]:
        """
            LLM이 생성한 서브테스크 분해 결과 텍스트를 서브테스크 단위로 파싱하여 리스트로 반환하는 함수

            반환 형태:
            [
                {
                    "id": 서브테스크 번호,
                    "title": 서브테스크 제목,
                    "skills": [필요한 스킬 목록],
                    "objects": [관련 오브젝트 목록],
                    "raw_block": 해당 서브테스크 원본 텍스트
                },
                ...
            ]
        """
        text = decomposed_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        header_re = re.compile(r"(?im)^\s*#?\s*Sub\s*Task\s*(\d+)\s*:\s*(.+?)\s*$")
        headers = list(header_re.finditer(text))
        if not headers:
            return []

        # initial condition 블록 정규식:
        # "# Initial condition analyze ..." 다음 줄들(#1, #2, ...)까지를 한 덩어리로
        initcond_re = re.compile(
            r"(?ims)^\s*#\s*Initial\s+condition\s+analyze\s+due\s+to\s+previous\s+subtask\s*:\s*\n"
            r"(?:\s*#\s*\d+\.\s*.*\n?)+"
        )

        subtasks: List[Dict] = []

        for i, h in enumerate(headers):
            start = h.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            sub_id = int(h.group(1))
            title = h.group(2).strip()

            # SubTask 본문 블록
            body_block = text[start:end].strip()

            prev_start = headers[i - 1].start() if i > 0 else 0
            prev_region = text[prev_start:start]  # 이전 subtask 시작~현재 subtask 시작 직전

            # prev_region 안에 initial condition이 여러 개 있을 수 있으니, 마지막 매치를 선택
            init_matches = list(initcond_re.finditer(prev_region))
            init_block = init_matches[-1].group(0).strip() if init_matches else ""

            # raw_block 구성: init + subtask body
            raw_block = (init_block + "\n\n" + body_block).strip() if init_block else body_block

            # Skills / Related Objects 추출
            skills = []
            objects = []

            skills_match = re.search(r"(?im)^\s*-?\s*Skills\s+Required\s*:\s*(.+)$", body_block)
            if skills_match:
                skills = [s.strip() for s in skills_match.group(1).split(",") if s.strip()]

            obj_match = re.search(r"(?im)^\s*-?\s*Related\s+Objects?\s*:\s*(.+)$", body_block)
            if obj_match:
                objects = [o.strip() for o in obj_match.group(1).split(",") if o.strip()]

            subtasks.append({
                "id": sub_id,
                "title": title,
                "skills": skills,
                "objects": objects,
                "raw_block": raw_block,
                "initial_conditions": init_block,
            })

        return subtasks

    def extract_domain_header(self, domain_content: str) -> str:
        """
        도메인에서 action 정의 이전까지의 '헤더' 부분만 추출
        (domain 선언, requirements, types, predicates, functions 포함)
        """
        # 첫 action 시작 지점 찾기
        m = re.search(r"(?m)^\s*\(\s*:action\b", domain_content)
        if not m:
            # action이 없다면 그냥 전체 반환
            return domain_content.strip()
        return domain_content[:m.start()].rstrip()

    def extract_action_blocks(self, domain_content: str) -> Dict[str, str]:
        """
        domain_content에서 모든 (:action NAME ... ) 블록을 추출해
        {action_name: action_block_text} 형태로 반환
        """
        # (?s) = DOTALL, 줄바꿈 포함
        # 액션 이름: (:action <NAME>
        # 블록 끝: 다음 (:action ... 또는 도메인 끝의 마지막 ')'
        pattern = re.compile(
            r"(?s)\(\s*:action\s+([A-Za-z0-9_-]+)\b(.*?)\n\s*\)\s*",  
        )

        # 위 패턴은 "액션 끝 괄호"를 정확히 잡기 어려울 수 있어서
        # 더 안전한 방식으로 action 시작 위치들을 모두 찾고, 다음 action 시작 전까지 슬라이스
        starts = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^\s*\(\s*:action\s+([A-Za-z0-9_-]+)\b", domain_content)]
        actions = {}

        if not starts:
            return actions

        for i, (pos, name) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(domain_content)
            block = domain_content[pos:end].rstrip()
            actions[name] = block

        return actions

    def build_subdomain_for_skills(self, domain_content: str, required_skills: List[str]) -> str:
        """
        서브태스크에서 필요한 스킬(action)만 포함한 '축약 도메인 텍스트' 생성
        """
        header = self.extract_domain_header(domain_content)
        action_map = self.extract_action_blocks(domain_content)

        # 중복 제거 + 원래 순서 보존(가능하면)
        seen = set()
        filtered_skills = []
        for s in required_skills:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                filtered_skills.append(s)

        selected_blocks = []
        missing = []
        for skill in filtered_skills:
            if skill in action_map:
                selected_blocks.append(action_map[skill])
            else:
                missing.append(skill)

        # 누락된 스킬이 있다면 주석으로 표시(디버깅에 도움)
        missing_comment = ""
        if missing:
            missing_comment = "\n; WARNING: missing actions in domain: " + ", ".join(missing) + "\n"

        # 도메인 괄호를 닫아야 하므로, header가 "(define ...)"를 이미 열고 있다면 마지막에 ")" 추가 필요
        # 현재 header는 action 이전까지 잘라온 것이므로 보통 아직 domain의 마지막 ')'는 없음.
        subdomain = header.rstrip() + "\n" + missing_comment + "\n\n" + "\n\n".join(selected_blocks) + "\n\n)"
        return subdomain
    
    def _generate_precondition_subtasks(self, parsed_subtasks: List[Dict[str, Any]], domain_content: str, robots: List[dict], objects_ai: str) -> List[Dict[str, Any]]:
        """
        서브태스크 리스트를 입력받아, 각 서브태스크별로 LLM을 호출해 PDDL problem을 위한 precondition과 goal을 추가해 반환해주는 함수

        """
        results: List[Dict[str, Any]] = []

        try:
            # 스킬/오브젝트 목록은 프롬프트 안정성을 위해 정렬
            skills_sorted = sorted(set([s.strip() for s in robots if s and s.strip()]))

            problem_example_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_pre_pddl_problem.py")
            problem_example_prompt = self.file_processor.read_file(problem_example_prompt_path)

            for st in parsed_subtasks:
                sub_id = st.get("id")
                title = st.get("title", "").strip()
                st_skills = st.get("skills", [])
                st_objects = st.get("objects", [])

                sub_domain = self.build_subdomain_for_skills(domain_content, st_skills)

                prompt = ""
                prompt += "You are a Robot task-to-action expander.\n"
                prompt += "Your job is to EXPAND the given task decomposition into detailed action-level plans using a PDDL-style description.\n"

                prompt += f"CURRENT ENVIRONMENT OBJECT LIST (ground-truth):\n{objects_ai}\n\n"
                prompt += "This is the ONLY set of objects that exist in the current environment.\n"

                prompt += f"OBJECTS SELECTED FROM THE PREVIOUS STEP (for this subtask):\n{st_objects}\n\n"
                prompt += "These are the objects that were judged to be relevant/needed for this subtask.\n"
                prompt += "You should prioritize using these objects when constructing the PDDL problem.\n"
                prompt += "However, you may also use other objects from CURRENT ENVIRONMENT OBJECT LIST if needed.\n\n\n"

                prompt += f"AVAILABLE ROBOT SKILLS (action names):\n{robots}\n\n"
                prompt += "These are the ONLY actions you are allowed to use.\n"

                prompt += f"SKILLS REQUIRED FOR THIS SUBTASK (selected from the previous step):{st_skills}\n"
                prompt += "Your generated PDDL problem must be solvable using ONLY these skills.\n\n\n"

                prompt += f"DOMAIN:\n{sub_domain}\n\n"
                prompt += "There are the domain content containing ONLY the actions you are allowed/required to use.\n"
                prompt += "Use this domain as the sole reference for predicates, action preconditions/effects.\n"

                prompt += "OUTPUT FORMAT CONSTRAINT:\n"
                prompt += "YOUR OUTPUT MUST BE ONLY the following expanded text (no markdown, no explanations)\n\n"

                prompt += "=== EXAMPLE OUTPUT FORMAT START ===\n"
                prompt += problem_example_prompt 
                prompt += "\n=== EXAMPLE OUTPUT FORMAT END ===\n\n"
                prompt += "\n\n=== SUBTASK TO SOLVE ===\n"

                prompt += f"{st}\n\n"  
                
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=2000, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [{"role": "user", "content": prompt}]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=2000, frequency_penalty=0.0)
                result = {
                    "subtask_id": sub_id,
                    "subtask_title": title,
                    "skills": st_skills,
                    "objects": st_objects,
                    "pre_goal_text": text,
                    "raw_llm_output": text,
                }
                results.append(result)

            return results

        except Exception as e:
            raise PDDLError(f"Error generating subtask PDDL problems: {str(e)}") from e

    def _generate_subtask_pddl_problems(self, parsed_subtasks: List[Dict[str, Any]], domain_content: str, robots: List[dict], objects_ai: str) -> List[Dict[str, Any]]:
        """
        서브태스크 리스트를 입력받아, 각 서브태스크별로 LLM을 호출해 PDDL problem을 생성하고 반환해주는 함수

        출력:
            List[dict] 형태로 서브태스크별 PDDL problem 결과 반환
            [
              {
                "task_index": 0,
                "subtask_id": 1,
                "subtask_title": "...",
                "skills": [...],
                "objects": [...],
                "problem_name": "...",
                "problem_text": "(define (problem ...))",
                "raw_llm_output": "..."
              },
              ...
            ]
        """
        results: List[Dict[str, Any]] = []

        try:
            # 스킬/오브젝트 목록은 프롬프트 안정성을 위해 정렬
            skills_sorted = sorted(set([s.strip() for s in robots if s and s.strip()]))

            problem_example_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"chaerin_pddl_train_task_pddl_problem.py")
            problem_example_prompt = self.file_processor.read_file(problem_example_prompt_path)

            for st in parsed_subtasks:
                sub_id = st.get("subtask_id")
                title = st.get("subtask_title", "").strip()
                st_skills = st.get("skills", [])
                st_objects = st.get("objects", [])

                sub_domain = self.build_subdomain_for_skills(domain_content, st_skills)

                prompt = ""
                prompt += "You are a PDDL problem generation expert for robot manipulation tasks.\n"
                prompt += "Your job is to generate a valid PDDL *problem* file for the given subtask.\n\n\n"

                prompt += f"CURRENT ENVIRONMENT OBJECT LIST (ground-truth):\n{objects_ai}\n\n"
                prompt += "This is the ONLY set of objects that exist in the current environment.\n"

                prompt += f"OBJECTS SELECTED FROM THE PREVIOUS STEP (for this subtask):\n{st_objects}\n\n"
                prompt += "These are the objects that were judged to be relevant/needed for this subtask.\n"
                prompt += "You should prioritize using these objects when constructing the PDDL problem.\n"
                prompt += "However, you may also use other objects from CURRENT ENVIRONMENT OBJECT LIST if needed.\n\n\n"

                prompt += f"AVAILABLE ROBOT SKILLS (action names):\n{robots}\n\n"
                prompt += "These are the ONLY actions you are allowed to use.\n"

                prompt += f"SKILLS REQUIRED FOR THIS SUBTASK (selected from the previous step):{st_skills}\n"
                prompt += "Your generated PDDL problem must be solvable using ONLY these skills.\n\n\n"

                prompt += f"DOMAIN :\n{domain_content}\n\n"
                prompt += "Use this domain as the sole reference for predicates, action preconditions/effects.\n"

                prompt += "OUTPUT FORMAT CONSTRAINT:\n"
                prompt += "You MUST output ONLY a single complete PDDL problem file.\n"
                prompt += "Do NOT include explanations, markdown, or extra text.\n"
                prompt += "Follow exactly the example format below.\n\n"

                prompt += "=== EXAMPLE OUTPUT FORMAT START ===\n"
                prompt += problem_example_prompt 
                prompt += "\n=== EXAMPLE OUTPUT FORMAT END ===\n\n"
                prompt += "\n\n=== SUBTASK TO SOLVE ===\n"
                prompt += f"{st}\n\n"  

                prompt += "=== WHAT YOU MUST GENERATE ===\n"
                prompt += "Robot starts in the kitchen!\n"
                prompt += "1) (:objects ...) must include ONLY objects that appear in CURRENT ENVIRONMENT OBJECT LIST.\n"
                prompt += "2) (:init ...) must include ALL facts required to make the plan executable.\n"
                prompt += "3) (:goal ...) must represent completion of THIS subtask only.\n"
                prompt += "4) OPENABLE vs NON-OPENABLE receptacles:\n"
                prompt += "   OPENABLE (use object-close in :init): Drawer, Cabinet, Safe, Microwave, Dishwasher, Toilet, ShowerDoor, Box\n"
                prompt += "   NON-OPENABLE (do NOT use object-close): CounterTop, StoveBurner, CoffeeMachine, DiningTable, Shelf, SinkBasin, Plate, Bowl, Bed, Sofa, ArmChair, Desk, SideTable, Dresser, TVStand, GarbageCan, Bathtub, Ottoman, Footrest\n"
                prompt += "   - For OPENABLE receptacles ONLY: include (object-close robot1 <receptacle>) in :init\n"
                prompt += "   - If putting an object INTO an openable receptacle, :goal should include (object-close robot1 <receptacle>)\n"
                prompt += "   - PutObject REQUIRES (not (object-close ?r ?loc)), so the planner must OpenObject first\n"
                prompt += "   - For NON-OPENABLE receptacles: do NOT add (object-close), just use PutObject directly\n"
                prompt += "5) FRIDGE: use (is-fridge fridge) and (not (fridge-open fridge)) in :init (see Example 3)\n\n"

                
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=2000, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [{"role": "user", "content": prompt}]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=2000, frequency_penalty=0.0)
                result = {
                    "subtask_id": sub_id,
                    "subtask_title": title,
                    "skills": st_skills,
                    "objects": st_objects,
                    "problem_name": f"subtask_{sub_id}_{re.sub(r'[^a-zA-Z0-9_]+','_', title)[:40]}",
                    "problem_text": text,
                    "raw_llm_output": text,
                }
                results.append(result)

            return results

        except Exception as e:
            raise PDDLError(f"Error generating subtask PDDL problems: {str(e)}") from e

    def _validate_and_plan(self) -> None:
        """pddl problem 파일 검증 후 fastDownward 실행, DAG 생성"""
        try:

            self.run_llmvalidator() #검증기

            self.run_planners() #fastDownward

        except Exception as e:
            raise PDDLError(f"Error in validation and planning: {str(e)}")

    def generate_dag(self) -> None:
        """LLM을 사용하여 plan의 DAG 생성 (병렬성 분석)"""
        try:
            print("\n[DAG] Generating DAG for parallelism analysis...")

            dag_generator = DAGGenerator(gpt_version=self.gpt_version)

            # plan 파일들 순회
            plans_dir = self.file_processor.subtask_pddl_plans_path
            problems_dir = self.file_processor.subtask_pddl_problems_path
            precond_dir = self.file_processor.precondition_subtasks_path

            # DAG 출력 폴더 생성
            dag_output_dir = os.path.join(self.resources_path, "dag_outputs")
            os.makedirs(dag_output_dir, exist_ok=True)

            plan_files = [f for f in os.listdir(plans_dir) if f.endswith("_actions.txt")]

            self.plan_dags = []  # DAG 저장

            for plan_file in sorted(plan_files):
                print(f"[DAG] Processing: {plan_file}")

                # plan 읽기
                plan_path = os.path.join(plans_dir, plan_file)
                with open(plan_path, 'r') as f:
                    plan_actions = [line.strip() for line in f.readlines() if line.strip()]

                if not plan_actions:
                    continue

                # 매칭 파일 찾기: subtask_01_xxx_actions.txt -> subtask_01_xxx.pddl, pre_01_xxx.txt
                base_name = plan_file.replace("_actions.txt", "")
                problem_path = os.path.join(problems_dir, f"{base_name}.pddl")
                precond_path = os.path.join(precond_dir, base_name.replace("subtask_", "pre_") + ".txt")

                problem_content = ""
                precond_content = ""

                if os.path.exists(problem_path):
                    with open(problem_path, 'r') as f:
                        problem_content = f.read()

                if os.path.exists(precond_path):
                    with open(precond_path, 'r') as f:
                        precond_content = f.read()

                # DAG 생성
                dag = dag_generator.build_dag(base_name, plan_actions, problem_content, precond_content)
                self.plan_dags.append(dag)

                # JSON 저장
                json_path = os.path.join(dag_output_dir, f"{base_name}_dag.json")
                dag_generator.save_dag_json(dag, json_path)

                # 시각화 저장
                img_path = os.path.join(dag_output_dir, f"{base_name}_dag.png")
                dag_generator.visualize_dag(dag, img_path)

                # 병렬 그룹 출력
                print(f"[DAG] Parallel groups for {base_name}:")
                for group_idx, node_ids in dag.parallel_groups.items():
                    actions_in_group = [dag.nodes[nid].action for nid in node_ids if nid < len(dag.nodes)]
                    print(f"  Step {group_idx}: {actions_in_group}")
            # -----------------------------
            # Subtask-level DAG 생성
            # -----------------------------
            print("\n[SubtaskDAG] Building subtask-level DAG...")

            summaries = []

            for plan_file in sorted(plan_files):
                base_name = plan_file.replace("_actions.txt", "")

                # 파일명에서 subtask ID 추출 (subtask_01_xxx -> 1)
                # parsed_subtasks와 동일한 1-based ID 사용
                match = re.search(r'subtask_(\d+)', plan_file)
                if match:
                    sid = int(match.group(1))
                else:
                    continue  # ID 추출 실패시 스킵

                plan_path = os.path.join(plans_dir, plan_file)
                with open(plan_path, "r") as f:
                    plan_actions = [line.strip() for line in f.readlines() if line.strip()]

                if not plan_actions:
                    continue

                problem_path = os.path.join(problems_dir, f"{base_name}.pddl")
                precond_path = os.path.join(precond_dir, base_name.replace("subtask_", "pre_") + ".txt")

                problem_content = ""
                precond_content = ""

                if os.path.exists(problem_path):
                    with open(problem_path, "r") as f:
                        problem_content = f.read()

                if os.path.exists(precond_path):
                    with open(precond_path, "r") as f:
                        precond_content = f.read()

                # 서브테스크 요약 생성 (1-based ID 사용)
                s = dag_generator.build_subtask_summary(
                    subtask_id=sid,
                    subtask_name=base_name,
                    plan_actions=plan_actions,
                    problem_content=problem_content,
                    precondition_content=precond_content
                )
                summaries.append(s)

            # 서브테스크 DAG 생성 + 저장
            task_name = "task"  # 원하면 task 이름 넣어도 됨
            subtask_dag = dag_generator.build_subtask_dag(task_name=task_name, summaries=summaries)
            self.subtask_dag = subtask_dag

            subtask_json = os.path.join(dag_output_dir, f"{task_name}_SUBTASK_DAG.json")
            dag_generator.save_subtask_dag_json(subtask_dag, subtask_json)

            subtask_png = os.path.join(dag_output_dir, f"{task_name}_SUBTASK_DAG.png")
            dag_generator.visualize_subtask_dag(subtask_dag, subtask_png)

            print(f"[SubtaskDAG] Saved: {subtask_json}")
            print(f"[SubtaskDAG] Saved: {subtask_png}")

            print(f"[DAG] Generated {len(self.plan_dags)} DAGs")

        except Exception as e:
            print(f"[DAG] Error generating DAG: {str(e)}")
            # DAG 생성 실패해도 전체 파이프라인은 계속 진행
            import traceback
            traceback.print_exc()

    def run_llmvalidator(self) -> None:
        """llm을 활용해서 pddl problem 파일 검증하기"""
        try:
            # subtask_pddl_problems_path (LLM이 만든 problem들)
            src_dir = self.file_processor.subtask_pddl_problems_path
            problem_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pddl')])

            src_dir2 = self.file_processor.precondition_subtasks_path
            precondition_files = sorted([f for f in os.listdir(src_dir2) if f.endswith('.txt')])

            # validated_subtask 폴더 비우기
            self.file_processor.clean_directory(self.file_processor.validated_subtask_path)

            self.validated_plan = []  # 매번 초기화

            for problem_file,precondition_file in zip(problem_files, precondition_files):
                problem_file_full = os.path.join(src_dir, problem_file)
                precondition_file_full = os.path.join(src_dir2, precondition_file)

                domain_name = self.file_processor.extract_domain_name(problem_file_full)
                if not domain_name:
                    print(f"[VALIDATOR] No domain specified in {problem_file}")
                    continue

                domain_file = self.file_processor.find_domain_file(domain_name)
                if not domain_file:
                    print(f"[VALIDATOR] No domain file found for domain {domain_name}")
                    continue

                domain_content = self.file_processor.read_file(domain_file)
                problem_content = self.file_processor.read_file(problem_file_full)
                precondition_content = self.file_processor.read_file(precondition_file_full)



                prompt = (
                    "You are a strict PDDL problem validator and repair system for Fast Downward.\n"
                    "The DOMAIN is the single source of truth.\n"
                    "Your job is to REWRITE the PROBLEM so that it is consistent with the DOMAIN "
                    "and solvable when the task intent is achievable.\n\n"

                    "CRITICAL RULES for receptacles:\n"
                    "OPENABLE objects (Drawer, Cabinet, Safe, Microwave, Dishwasher, Toilet, ShowerDoor, Box):\n"
                    "- Include (object-close robot1 <receptacle>) in :init (they start CLOSED)\n"
                    "- PutObject requires (not (object-close ?r ?loc)), so planner MUST OpenObject first\n"
                    "- If placing into them, :goal should include (object-close robot1 <receptacle>)\n"
                    "NON-OPENABLE objects (CounterTop, StoveBurner, CoffeeMachine, DiningTable, Shelf, SinkBasin, Plate, Bed, Sofa, Desk, GarbageCan, Bathtub):\n"
                    "- Do NOT include (object-close) for these. PutObject works directly.\n"
                    "FRIDGE: use (is-fridge fridge) and (not (fridge-open fridge)) in :init.\n\n"

                    f"precondition Description (to be check preconditions):\n{precondition_content}\n\n"
                    f"Domain Description (authoritative):\n{domain_content}\n\n"
                    f"Problem Description (to be repaired):\n{problem_content}\n\n"


                    "Output format must be:\n"
                    "(define (problem <name>)\n"
                    "  (:domain <domain>)\n"
                    "  (:objects ...)\n"
                    "  (:init ...)\n"
                    "  (:goal (and ...))\n"
                    ")\n"
                )
                if "gpt" not in self.gpt_version:
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1400, stop=["def"], frequency_penalty=0.0)
                else:
                    messages = [
                        {"role": "system", "content": "You are a PDDL validator. Output ONLY corrected PDDL."},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.0)

                # 정규화(주석/텍스트 제거 + (define부터)
                validated = self.file_processor.normalize_pddl(text)

                # 검증 결과 저장
                self.validated_plan.append(validated)

                out_path = os.path.join(self.file_processor.validated_subtask_path, problem_file)
                self.file_processor.write_file(out_path, validated)
                print(f"[VALIDATED WROTE] {out_path}")

        except Exception as e:
            print(f"Error in run_llmvalidator: {str(e)}")
            raise

    def extract_plan_actions(self, fd_stdout: str) -> list[str]:
        """
        'switchon robot1 faucet (1)'만 추출하는 함수
        """
        plan_actions = []
        action_pattern = re.compile(r'^[a-zA-Z_]+\s+.*\(\d+\)$')

        for line in fd_stdout.splitlines():
            line = line.strip()
            if action_pattern.match(line):
                plan_actions.append(line)

        return plan_actions

    def _inject_metric(self, problem_path: str) -> None:
        """problem PDDL에 action-cost metric이 없으면 자동 주입"""
        content = self.file_processor.read_file(problem_path)
        if '(total-cost)' in content:
            return  # 이미 있음

        # (:init ...) 안에 (= (total-cost) 0) 추가
        content = content.replace('(:init', '(:init\n    (= (total-cost) 0)')
        # 마지막 ) 앞에 (:metric minimize (total-cost)) 추가
        last_paren = content.rfind(')')
        content = content[:last_paren] + '\n  (:metric minimize (total-cost))\n)' + content[last_paren+1:]
        self.file_processor.write_file(problem_path, content)

    def run_planners(self) -> None:
        """Run PDDL planners on problem files."""
        try:
            planner_path = os.path.join(self.base_path, "downward", "fast-downward.py")
            problem_dir = self.file_processor.validated_subtask_path
            problem_files = [f for f in os.listdir(problem_dir) if f.endswith('.pddl')]  #나중에 검증기 넣을거면 여기 수정해야함
            for problem_file in problem_files:
                try:
                    problem_file_full = os.path.join(problem_dir, problem_file)
                    # action-cost metric 주입
                    self._inject_metric(problem_file_full)
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue

                    command = [
                        planner_path,
                        "--alias",
                        "seq-opt-lmcut",
                        domain_file,
                        problem_file_full
                    ]

                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    # 플랜 액션 추출
                    plan_actions = self.extract_plan_actions(result.stdout)
                    self.subtask_pddl_plans.append(plan_actions)

                    base_name = os.path.splitext(problem_file)[0]
                    actions_path = os.path.join(self.file_processor.subtask_pddl_plans_path, f"{base_name}_actions.txt")

                    with open(actions_path, "w") as f:
                        f.write("\n".join(plan_actions))

                    print(f"\n========== FAST-DOWNWARD OUTPUT ({problem_file}) ==========")
                    print(plan_actions)
                    print("===========================================================\n")
                    print(result.stdout)
                    print("===========================================================\n")


                    if result.stderr:
                        print(f"Warnings/Errors for {problem_file}:", result.stderr)
                        
                except subprocess.TimeoutExpired:
                    print(f"Planner timed out for {problem_file}")
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_planners: {str(e)}")
            raise

    def run_planner_for_subtask_id(self, subtask_id: int) -> Tuple[bool, List[str]]:
            """단일 서브태스크에 대해서만 Fast-Downward 실행 (피드백 루프에서 재계획 시 사용)."""
            return run_planner_for_one_subtask(
                base_path=self.base_path,
                file_processor=self.file_processor,
                extract_plan_actions_fn=self.extract_plan_actions,
                subtask_id=subtask_id,
                validated_subtask_path=self.file_processor.validated_subtask_path,
                subtask_pddl_plans_path=self.file_processor.subtask_pddl_plans_path,
            )

    def run_feedback_replan(
        self,
        task_idx: int,
        task_name: str,
        execution_results: Dict[int, Any],
        domain_content: str,
        objects_ai: str,
        state_store: Optional[Any] = None,
        task_robot_ids: Optional[List[int]] = None,
        floor_plan: Optional[int] = None,
        max_subtask_replan_attempts: int = 2,
    ) -> str:
        """
        재계획: decomposition → DAG 통합 → LP 재할당 → 재실행 준비
        
        Returns:
            "no_change" | "fully_replanned"
        """
        failed = [sid for sid, r in execution_results.items() if not r.success]
        if not failed:
            return "no_change"
        
        edges = load_subtask_dag_edges(self.base_path, task_name)
        parallel_groups = load_subtask_dag_parallel_groups(self.base_path, task_name)
        # dependency_groups: 의존성 간선으로 연결된 Subtask Connected Component
        # → LLM GroupAgent 할당 기준
        dependency_groups = load_dependency_groups_from_dag(self.base_path, task_name)
        
        # GroupAgent 생성
        group_agent = GroupAgent(self.llm, self.gpt_version)
        
        # decomposition_callback을 포함한 PartialReplanner 생성
        if state_store is not None:
            partial_replanner = self.create_partial_replanner(
                state_store=state_store,
                group_agent=group_agent
            )
        else:
            partial_replanner = None
        
        updated = False

        # dependency_groups별 재계획
        if partial_replanner is not None and dependency_groups:
            processed_groups = set()
            
            for failed_id in failed:
                # 실패한 서브태스크가 속한 dependency_group 찾기
                found_group = None
                for dgid, sids in dependency_groups.items():
                    if failed_id in sids:
                        found_group = dgid
                        break
                
                if found_group is None or found_group in processed_groups:
                    continue
                
                processed_groups.add(found_group)
                subtask_ids_in_group = dependency_groups[found_group]
                
                # ReplanContext 구성 - AI2Thor로 실제 로컬 환경 수집
                if floor_plan is not None:
                    try:
                        raw_local_env = build_local_env_for_group(
                            base_path=self.base_path,
                            group_subtask_ids=subtask_ids_in_group,
                            floor_plan=floor_plan,
                            task_name=task_name,
                        )
                        local_env_str = local_env_to_str(raw_local_env)
                    except Exception as _le:
                        print(f"[Feedback] local_env 수집 실패 (group {found_group}): {_le}")
                        local_env_str = "Kitchen environment with standard appliances"
                else:
                    local_env_str = "Kitchen environment with standard appliances"

                context = partial_replanner.build_context_for_replan(
                    dep_group_id=found_group,
                    subtask_ids_in_dep_group=subtask_ids_in_group,
                    local_env=local_env_str
                )
                
                if context is None:
                    continue
                
                print(f"\n[Feedback] Attempting group-level decomposition for group {found_group}")
                
                # 현재 문제/액션 정보 수집
                plans_dir = self.file_processor.subtask_pddl_plans_path
                problems_dir = self.file_processor.subtask_pddl_problems_path
                
                current_actions_by_id = {}
                problem_content_by_id = {}
                subtask_name_by_id = {}
                
                for sid in subtask_ids_in_group:
                    plan_actions = self._load_plan_actions_by_subtask_id().get(sid, [])
                    current_actions_by_id[sid] = plan_actions
                    
                    base_name = None
                    for fname in os.listdir(plans_dir):
                        m = re.match(rf"subtask_{sid:02d}_(.+)_actions\.txt$", fname)
                        if m:
                            base_name = f"subtask_{sid:02d}_{m.group(1)}"
                            break
                    
                    if base_name:
                        subtask_name_by_id[sid] = base_name
                        problem_path = os.path.join(problems_dir, f"{base_name}.pddl")
                        if os.path.exists(problem_path):
                            with open(problem_path, "r") as f:
                                problem_content_by_id[sid] = f.read()
                
                domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
                domain_content_actual = self.file_processor.read_file(domain_path)
                
                # 그룹 재계획
                success = partial_replanner.replan_group(
                    dep_group_id=found_group,
                    subtask_ids_in_dep_group=subtask_ids_in_group,
                    context=context,
                    domain_content=domain_content_actual,
                    problem_content_by_id=problem_content_by_id,
                    current_actions_by_id=current_actions_by_id,
                    subtask_name_by_id=subtask_name_by_id,
                )
                
                if success:
                    print(f"[Feedback] Group {found_group} successfully replanned via decomposition")
                    
                    replanned_ids = [context.failed_subtask_id] + context.remaining_pending_ids
                    
                    # DAG 통합
                    dag_integrated = self.integrate_replanned_subtasks_to_dag(
                        task_name=task_name,
                        task_idx=task_idx,
                        original_group_id=found_group,
                        replanned_subtask_ids=replanned_ids,
                        new_plans=current_actions_by_id,
                        state_store=state_store
                    )
                    
                    if not dag_integrated:
                        print("[Feedback] DAG integration failed, skipping LP reallocation")
                        continue
                    
                    # LP 재할당
                    if task_robot_ids is not None:
                        new_assignment = self.recompute_task_assignment_after_replan(
                            task_idx=task_idx,
                            task_name=task_name,
                            task_robot_ids=task_robot_ids,
                            floor_plan=floor_plan
                        )
                        
                        if new_assignment is None:
                            print("[Feedback] LP reallocation failed")
                            continue
                    else:
                        print("[Feedback] Warning: No task_robot_ids provided, skipping LP reallocation")
                    
                    updated = True
                    break
        
        # Fallback
        if not updated:
            print("[Feedback] Falling back to individual subtask replan")
            subtask_mgr = SubtaskManagerLLM(self.llm, self.gpt_version)
            success_effects_context = ""
            if state_store is not None:
                success_effects = state_store.get_success_effects_immutable()
                success_effects_context = format_success_effects_for_prompt(success_effects)
            
            plans_dir = self.file_processor.subtask_pddl_plans_path
            problems_dir = self.file_processor.subtask_pddl_problems_path
            precond_dir = self.file_processor.precondition_subtasks_path
            
            for sid in failed:
                if subtask_has_dependency(sid, edges):
                    print(f"[Feedback] Subtask {sid} has dependencies, skipping fallback replan (dependency_group replan should handle it)")
                    continue
                result = execution_results.get(sid)
                if not result or result.success:
                    continue
                err = result.error_message or "Unknown error"
                
                plan_actions = self._load_plan_actions_by_subtask_id().get(sid, [])
                base_name = None
                for fname in os.listdir(plans_dir):
                    m = re.match(rf"subtask_{sid:02d}_(.+)_actions\.txt$", fname)
                    if m:
                        base_name = f"subtask_{sid:02d}_{m.group(1)}"
                        break
                if not base_name:
                    continue
                
                problem_path = os.path.join(problems_dir, f"{base_name}.pddl")
                problem_content = ""
                if os.path.exists(problem_path):
                    with open(problem_path, "r") as f:
                        problem_content = f.read()
                
                allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
                domain_content_actual = self.file_processor.read_file(allaction_domain_path)
                
                for attempt in range(max_subtask_replan_attempts):
                    new_actions = subtask_mgr.replan_subtask(
                        subtask_id=sid,
                        subtask_name=base_name,
                        current_actions=plan_actions,
                        error_message=err,
                        domain_content=domain_content_actual,
                        problem_content=problem_content,
                        success_effects_context=success_effects_context or None,
                    )
                    if new_actions:
                        actions_path = os.path.join(plans_dir, f"{base_name}_actions.txt")
                        with open(actions_path, "w") as f:
                            f.write("\n".join(new_actions))
                        print(f"[Feedback] Subtask {sid} replanned by Subtask Manager LLM ({len(new_actions)} actions)")
                        updated = True
                        plan_actions = new_actions
                        break
                    
                    ok, plan_actions = self.run_planner_for_subtask_id(sid)
                    if ok and plan_actions:
                        updated = True
                        break
        
        return "fully_replanned" if updated else "no_change"

    def create_decomposition_callback(self):
        """
        이 TaskManager 인스턴스를 위한 decomposition_callback 생성
        
        Returns:
            Callable: 실패한 그룹을 재분해하는 콜백 함수
        """
        def decomposition_callback(
            group_id: int,
            subtask_ids_in_group: List[int],
            context,  # ReplanContext
        ) -> Optional[Dict[int, List[str]]]:
            """
            실패한 서브태스크 그룹을 재분해하여 새로운 서브태스크 생성 후 계획 수립
            
            Args:
                group_id: 병렬 그룹 ID
                subtask_ids_in_group: 그룹 내 서브태스크 ID 리스트
                context: ReplanContext (success_effects, failed_subtask_id, 등)
            
            Returns:
                Dict[subtask_id, action_list] or None if failed
            """
            print(f"\n{'='*60}")
            print(f"[DecompCallback] Redecomposing Group {group_id}")
            print(f"{'='*60}")
            print(f"  Subtasks in group: {subtask_ids_in_group}")
            print(f"  Failed subtask: {context.failed_subtask_id}")
            print(f"  Remaining pending: {context.remaining_pending_ids}")
            
            # 1. 재계획 대상 서브태스크 리스트 (실패 + 미수행)
            tasks_to_replan = [context.failed_subtask_id] + context.remaining_pending_ids
            print(f"  Tasks to redecompose: {tasks_to_replan}")
            
            if not tasks_to_replan:
                print("  No tasks to replan")
                return None
            
            # 2. 원래 서브태스크 목표 정보 수집
            precond_dir = self.file_processor.precondition_subtasks_path
            problems_dir = self.file_processor.subtask_pddl_problems_path
            
            subtask_goals = {}
            for sid in tasks_to_replan:
                # pre_XX_*.txt 파일 찾기
                found = False
                if os.path.exists(precond_dir):
                    for fname in os.listdir(precond_dir):
                        if fname.startswith(f"pre_{sid:02d}_"):
                            precond_path = os.path.join(precond_dir, fname)
                            try:
                                with open(precond_path, "r") as f:
                                    subtask_goals[sid] = f.read()
                                found = True
                                break
                            except Exception as e:
                                print(f"  Warning: Could not read {fname}: {e}")
                
                if not found:
                    print(f"  Warning: No precondition file found for subtask {sid}")
            
            if not subtask_goals:
                print("  ERROR: No subtask goals found, cannot redecompose")
                return None
            
            # 3. 성공한 서브태스크의 Effects 포맷
            success_effects_text = format_success_effects_for_prompt(
                context.success_effects,
                exclude_subtask_id=None
            )
            
            # 4. 통합 목표 텍스트 생성
            combined_goals = "\n\n".join([
                f"## Original Subtask {sid}:\n{goal}"
                for sid, goal in sorted(subtask_goals.items())
            ])
            
            # 5. 도메인 로드
            domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            try:
                domain_content = self.file_processor.read_file(domain_path)
            except Exception as e:
                print(f"  ERROR: Could not read domain: {e}")
                return None
            
            # 6. 재분해 프롬프트 생성
            redecompose_prompt = f"""You are redecomposing a failed group of subtasks in a multi-robot collaborative task.

    ## Context
    The following subtasks failed or were not executed. You need to redecompose them into NEW subtasks that will succeed.

    ## Already Achieved Effects (from successful subtasks - DO NOT REPEAT)
    {success_effects_text if success_effects_text else "(None - this is the first attempt)"}

    ## Local Environment
    {context.local_env if hasattr(context, 'local_env') and context.local_env else "(Standard kitchen environment)"}

    ## Failure Information
    - Failed Subtask ID: {context.failed_subtask_id}
    - Failure Reason: {context.failure_reason}
    - This suggests the approach needs to be changed

    ## Original Goals (to be re-achieved with new approach)
    {combined_goals}

    ## Available Robot Skills
    {self.available_robot_skills}

    ## Environment Objects
    {self.objects_ai}

    ## Domain Actions Reference
    {domain_content[:2500]}
    ...

    ## Your Task
    Redecompose the failed/pending goals into NEW subtasks that:
    1. Account for already achieved effects (don't duplicate successful work)
    2. Use a different approach to avoid the failure cause
    3. Maximize parallelism where dependencies allow
    4. Use ONLY available skills and existing objects

    ## Output Format
    For each new subtask:

    # SubTask <ID>: <Descriptive Title>
    - Skills Required: <skill1>, <skill2>, ...
    - Related Objects: <object1>, <object2>, ...
    - Description: <what this subtask accomplishes>
    # Initial condition analyze due to previous subtask:
    # 1. <condition>
    # 2. <condition>
    ...

    Start numbering from {min(tasks_to_replan)}.
    Output 1-3 subtasks maximum (prefer fewer, more robust subtasks).
    """
            
            # 7. LLM 호출
            print("  Calling LLM for redecomposition...")
            try:
                if "gpt" in self.gpt_version.lower():
                    messages = [{"role": "user", "content": redecompose_prompt}]
                    _, redecompose_text = self.llm.query_model(
                        messages,
                        self.gpt_version,
                        max_tokens=2500,
                        frequency_penalty=0.0
                    )
                else:
                    _, redecompose_text = self.llm.query_model(
                        redecompose_prompt,
                        self.gpt_version,
                        max_tokens=2500,
                        stop=None,
                        frequency_penalty=0.0
                    )
            except Exception as e:
                print(f"  ERROR: LLM call failed: {e}")
                return None
            
            if not redecompose_text:
                print("  ERROR: Empty response from LLM")
                return None
            
            # 8. 응답 파싱
            print("  Parsing LLM response...")
            redecomposed_subtasks = self._decomposed_plan_to_subtasks(redecompose_text)
            
            if not redecomposed_subtasks:
                print("  ERROR: Could not parse redecomposed subtasks")
                return None
            
            print(f"  ✓ Redecomposed into {len(redecomposed_subtasks)} new subtasks")
            
            # 9. Precondition 및 PDDL Problem 생성
            print("  Generating preconditions and PDDL problems...")
            try:
                # Precondition 생성 (성공 effects를 초기 상태로 반영)
                precondition_subtasks = self._generate_precondition_subtasks(
                    redecomposed_subtasks,
                    domain_content,
                    self.available_robot_skills,
                    self.objects_ai
                )
                
                # 성공 effects를 초기 상태에 명시적으로 추가
                if success_effects_text:
                    for item in precondition_subtasks:
                        pre_text = item.get("pre_goal_text", "")
                        # precondition 텍스트에 성공 effects 주입
                        enhanced_pre = f"# Already Achieved (from successful subtasks):\n{success_effects_text}\n\n{pre_text}"
                        item["pre_goal_text"] = enhanced_pre
                
                # PDDL Problem 생성
                subtask_pddl_problems = self._generate_subtask_pddl_problems(
                    precondition_subtasks,
                    domain_content,
                    self.available_robot_skills,
                    self.objects_ai
                )
                
            except Exception as e:
                print(f"  ERROR: Failed to generate PDDL: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # 10. 파일 저장
            print("  Saving generated files...")
            try:
                for item in precondition_subtasks:
                    sid = item.get("subtask_id", -1)
                    title = item.get("subtask_title", "untitled")
                    text = item.get("pre_goal_text", "")
                    
                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"pre_{sid:02d}_{safe_title}_REPLAN.txt"
                    out_path = os.path.join(precond_dir, filename)
                    self.file_processor.write_file(out_path, text)
                
                for item in subtask_pddl_problems:
                    sid = item["subtask_id"]
                    title = item["subtask_title"]
                    pddl_text = item["problem_text"]
                    
                    safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_')
                    filename = f"subtask_{sid:02d}_{safe_title}_REPLAN.pddl"
                    out_path = os.path.join(problems_dir, filename)
                    self.file_processor.write_file(out_path, pddl_text)
            except Exception as e:
                print(f"  ERROR: Failed to save files: {e}")
                return None
            
            # 11. 플래너 실행하여 액션 생성
            print("  Running planner for new subtasks...")
            new_plans: Dict[int, List[str]] = {}
            
            for item in subtask_pddl_problems:
                sid = item["subtask_id"]
                
                ok, plan_actions = self.run_planner_for_subtask_id(sid)
                
                if ok and plan_actions:
                    new_plans[sid] = plan_actions
                    print(f"    Subtask {sid}: ✓ {len(plan_actions)} actions")
                else:
                    print(f"    Subtask {sid}: ✗ Planning failed")
            
            if not new_plans:
                print("  ERROR: No valid plans generated")
                return None
            
            print(f"\n  ✓ Successfully generated {len(new_plans)} new plans")
            print(f"{'='*60}\n")
            return new_plans
        
        return decomposition_callback
    
    def create_partial_replanner(
        self,
        state_store,
        group_agent
    ):
        """
        PartialReplanner 생성 시 decomposition_callback을 주입하는 헬퍼 메서드
        
        Args:
            state_store: SharedTaskStateStore 인스턴스
            group_agent: GroupAgent 인스턴스
        
        Returns:
            PartialReplanner (decomposition_callback 포함)
        """
        decomp_callback = self.create_decomposition_callback()
        
        replanner = PartialReplanner(
            store=state_store,
            group_agent=group_agent,
            decomposition_callback=decomp_callback
        )
        
        return replanner

    def integrate_replanned_subtasks_to_dag(
        self,
        task_name: str,
        task_idx: int,
        original_group_id: int,
        replanned_subtask_ids: List[int],
        new_plans: Dict[int, List[str]],
        state_store,
    ) -> bool:
        """
        재계획된 서브테스크들을 기존 DAG에 통합하고 새 DAG 생성
        
        플로우:
        1. 기존 DAG 로드
        2. 성공한 서브테스크 필터링
        3. 재계획된 서브테스크들에 대한 새 DAG 생성
        4. 성공 DAG + 새 DAG 병합
        5. 병렬 그룹 재계산
        6. 새 통합 DAG 저장
        
        Args:
            task_name: 작업 이름 (예: "task")
            task_idx: 작업 인덱스
            original_group_id: 실패가 발생한 원래 그룹 ID
            replanned_subtask_ids: 재계획된 서브테스크 ID 리스트
            new_plans: 새로운 액션 계획 (subtask_id -> actions)
            state_store: SharedTaskStateStore 인스턴스
        
        Returns:
            bool: 통합 성공 여부
        """
        try:
            print(f"\n{'='*60}")
            print(f"[DAG Integration] Integrating replanned subtasks into DAG")
            print(f"{'='*60}")
            
            dag_output_dir = os.path.join(self.resources_path, "dag_outputs")
            
            # 1. 기존 DAG 로드
            dag_path = os.path.join(dag_output_dir, f"{task_name}_SUBTASK_DAG.json")
            
            if not os.path.exists(dag_path):
                print(f"  ERROR: Original DAG not found at {dag_path}")
                return False
            
            with open(dag_path, "r") as f:
                original_dag = json.load(f)
            
            print(f"  ✓ Loaded original DAG with {len(original_dag['nodes'])} nodes")
            
            # 2. 성공한 서브테스크 노드/엣지 필터링
            success_ids = []
            failed_ids = []
            
            for sid, rec in state_store._store.items():
                sid_int = int(sid)
                if rec.get("state") == "SUCCESS":
                    success_ids.append(sid_int)
                elif rec.get("state") == "FAILED":
                    failed_ids.append(sid_int)
            
            print(f"  Success subtasks: {success_ids}")
            print(f"  Failed subtasks: {failed_ids}")
            print(f"  Replanned subtasks: {replanned_subtask_ids}")
            
            # 성공한 서브테스크 노드 유지
            success_ids_set = set(success_ids)
            success_nodes = [
                n for n in original_dag["nodes"]
                if int(n["id"]) in success_ids_set
            ]
            
            # 성공한 서브테스크 간 엣지만 유지
            success_edges = [
                e for e in original_dag["edges"]
                if int(e["from_id"]) in success_ids_set and int(e["to_id"]) in success_ids_set
            ]
            
            print(f"  ✓ Kept {len(success_nodes)} success nodes, {len(success_edges)} edges")
            
            # 3. 재계획된 서브테스크들에 대한 새 DAG 노드 생성
            print(f"\n  [Step 3] Generating new DAG for replanned subtasks...")
            
            plans_dir = self.file_processor.subtask_pddl_plans_path
            problems_dir = self.file_processor.subtask_pddl_problems_path
            precond_dir = self.file_processor.precondition_subtasks_path
            
            # DAGGenerator를 사용해 각 재계획 서브테스크의 DAG 생성
            from DAG_Module import DAGGenerator
            dag_generator = DAGGenerator(gpt_version=self.gpt_version)
            
            new_summaries = []
            
            for sid in sorted(replanned_subtask_ids):
                # REPLAN 파일들 찾기
                base_name = None
                for fname in os.listdir(plans_dir):
                    # subtask_XX_*_REPLAN_actions.txt 또는 subtask_XX_*_actions.txt
                    if re.match(rf"subtask_{sid:02d}_.*_actions\.txt$", fname):
                        base_name = fname.replace("_actions.txt", "")
                        break
                
                if not base_name:
                    print(f"    Warning: No plan file found for subtask {sid}")
                    continue
                
                plan_path = os.path.join(plans_dir, f"{base_name}_actions.txt")
                problem_path = os.path.join(problems_dir, f"{base_name}.pddl")
                precond_path = os.path.join(precond_dir, base_name.replace("subtask_", "pre_") + ".txt")
                
                plan_actions = []
                problem_content = ""
                precond_content = ""
                
                if os.path.exists(plan_path):
                    with open(plan_path, "r") as f:
                        plan_actions = [line.strip() for line in f.readlines() if line.strip()]
                
                if os.path.exists(problem_path):
                    with open(problem_path, "r") as f:
                        problem_content = f.read()
                
                if os.path.exists(precond_path):
                    with open(precond_path, "r") as f:
                        precond_content = f.read()
                
                # 서브테스크 요약 생성
                summary = dag_generator.build_subtask_summary(
                    subtask_id=sid,
                    subtask_name=base_name,
                    plan_actions=plan_actions,
                    problem_content=problem_content,
                    precondition_content=precond_content
                )
                new_summaries.append(summary)
                
                print(f"    Subtask {sid}: {len(plan_actions)} actions")
            
            # 재계획된 서브테스크들에 대한 DAG 생성
            if not new_summaries:
                print("  ERROR: No summaries generated for replanned subtasks, aborting integration")
                return False
            new_subtask_dag = dag_generator.build_subtask_dag(
                task_name=f"{task_name}_replanned",
                summaries=new_summaries
            )
            
            # SubtaskSummary 객체를 dict 변환
            from dataclasses import asdict
            new_nodes = [
                asdict(n) if hasattr(n, '__dataclass_fields__') else n
                for n in new_subtask_dag.nodes
            ]
            new_edges = [
                {"from_id": e.from_id, "to_id": e.to_id,
                 "dependency_type": getattr(e, "dependency_type", "causal"),
                 "reason": getattr(e, "reason", "")}
                if hasattr(e, 'from_id') else e
                for e in new_subtask_dag.edges
            ]
            
            print(f"  ✓ Generated new DAG: {len(new_nodes)} nodes, {len(new_edges)} edges")
            
            # 4. 성공 DAG + 새 DAG 병합
            print(f"\n  [Step 4] Merging success DAG and new DAG...")
            
            # 노드 병합
            integrated_nodes = success_nodes + new_nodes
            
            # 엣지 병합
            integrated_edges = success_edges + new_edges
            
            # 5. 성공한 서브테스크와 재계획 서브테스크 간 연결
            # (의존성 분석: 성공 서브테스크 중 어떤 것이 재계획 서브테스크의 선행자인지)
            
            # 원래 DAG에서 실패/재계획 서브테스크들이 의존하던 선행자 찾기
            replanned_ids_set = set(replanned_subtask_ids)
            for new_sid in replanned_subtask_ids:
                # 원래 DAG에서 이 서브테스크로 들어오는 엣지 찾기
                for orig_edge in original_dag["edges"]:
                    if int(orig_edge["to_id"]) == new_sid:
                        predecessor_id = int(orig_edge["from_id"])
                        # 선행자가 성공한 서브테스크라면 연결 유지
                        if predecessor_id in success_ids_set:
                            integrated_edges.append({
                                "from_id": predecessor_id,
                                "to_id": new_sid,
                                "dependency_type": orig_edge.get("dependency_type", "causal"),
                                "reason": orig_edge.get("reason", ""),
                            })
                            print(f"    Reconnected: {predecessor_id} → {new_sid}")
            
            # 중복 제거
            unique_edges = []
            seen = set()
            for e in integrated_edges:
                key = (e["from_id"], e["to_id"])
                if key not in seen:
                    unique_edges.append(e)
                    seen.add(key)
            
            integrated_edges = unique_edges
            
            print(f"  ✓ Merged DAG: {len(integrated_nodes)} nodes, {len(integrated_edges)} edges")
            
            # 6. 병렬 그룹 재계산
            print(f"\n  [Step 5] Recomputing parallel groups...")
            
            parallel_groups = self._compute_parallel_groups_from_dag(
                integrated_nodes,
                integrated_edges
            )
            
            print(f"  ✓ Computed {len(parallel_groups)} parallel groups")
            for gid, sids in sorted(parallel_groups.items()):
                print(f"    Group {gid}: {sids}")
            
            # 7. 통합 DAG 생성 및 저장
            integrated_dag = {
                "nodes": integrated_nodes,
                "edges": integrated_edges,
                "parallel_groups": parallel_groups
            }
            
            # 백업 (원본 보존)
            backup_path = os.path.join(dag_output_dir, f"{task_name}_SUBTASK_DAG_BACKUP.json")
            if not os.path.exists(backup_path):
                with open(backup_path, "w") as f:
                    json.dump(original_dag, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Backup saved: {backup_path}")
            
            # 새 DAG 저장 (원본 덮어쓰기)
            with open(dag_path, "w") as f:
                json.dump(integrated_dag, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Integrated DAG saved: {dag_path}")
            
            # 시각화
            try:
                img_path = os.path.join(dag_output_dir, f"{task_name}_SUBTASK_DAG_INTEGRATED.png")
                dag_generator.visualize_subtask_dag(new_subtask_dag, img_path)
                print(f"  ✓ Visualization saved: {img_path}")
            except Exception as e:
                print(f"  Warning: Visualization failed: {e}")
            
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"  ERROR: Failed to integrate DAG: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _compute_parallel_groups_from_dag(
        self,
        nodes: List[Dict],
        edges: List[Dict]
    ) -> Dict[int, List[int]]:
        """
        노드와 엣지로부터 parallel_groups 계산 — 위상 정렬(Kahn's algorithm) 기반.

        같은 레벨(level)에 속한 subtask들은 서로 의존성이 없으므로 병렬 실행 가능.
        실행기는 레벨 순서대로 그룹을 발사하고, 같은 레벨 내에서는 스레드로 동시 실행.

        [주의] Union-Find(Connected Component) 방식은 dependency_groups 계산용.
               이 함수는 반드시 위상 정렬을 사용해야 한다.

        Args:
            nodes: DAG 노드 리스트 ({"id": int, ...})
            edges: DAG 엣지 리스트 ({"from_id": int, "to_id": int})

        Returns:
            Dict[level, List[subtask_id]]: 레벨별 병렬 그룹
        """
        if not nodes:
            return {}

        node_ids = [n["id"] for n in nodes]
        id_set = set(node_ids)

        preds: Dict[int, set] = {i: set() for i in node_ids}
        succs: Dict[int, set] = {i: set() for i in node_ids}
        for e in edges:
            u, v = e["from_id"], e["to_id"]
            if u in id_set and v in id_set:
                preds[v].add(u)
                succs[u].add(v)

        from collections import deque
        indeg = {i: len(preds[i]) for i in node_ids}
        level = {i: 0 for i in node_ids}
        q = deque([i for i in node_ids if indeg[i] == 0])
        processed: set = set()

        while q:
            u = q.popleft()
            processed.add(u)
            for v in succs[u]:
                level[v] = max(level[v], level[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        # 사이클 감지 후 누락 노드 복구
        cycle_nodes = [i for i in node_ids if i not in processed]
        if cycle_nodes:
            print(f"[DAG] WARNING: Cycle detected {cycle_nodes}, placing at max_level+1")
            max_level = max(level[i] for i in processed) if processed else 0
            for i in cycle_nodes:
                level[i] = max_level + 1

        groups: Dict[int, List[int]] = {}
        for i in node_ids:
            groups.setdefault(level[i], []).append(i)
        return groups


    def recompute_task_assignment_after_replan(
        self,
        task_idx: int,
        task_name: str,
        task_robot_ids: List[int],
        floor_plan: Optional[int] = None
    ) -> Optional[Dict[int, int]]:
        """
        재계획 후 LP 작업 할당 재수행
        
        Args:
            task_idx: 작업 인덱스
            task_name: 작업 이름
            task_robot_ids: 사용 가능한 로봇 ID 리스트
            floor_plan: FloorPlan 번호 (거리 기반 최적화용)
        
        Returns:
            Dict[subtask_id, robot_id]: 새 작업 할당 또는 None
        """
        try:
            print(f"\n{'='*60}")
            print(f"[LP Reallocation] Recomputing task assignment")
            print(f"{'='*60}")
            
            # 1. 새 통합 DAG 로드
            dag_path = os.path.join(self.resources_path, "dag_outputs", f"{task_name}_SUBTASK_DAG.json")
            
            with open(dag_path, "r") as f:
                integrated_dag = json.load(f)
            
            # 2. 서브테스크 정보 수집
            # (parsed_subtasks 형태로 변환 필요)
            parsed_subtasks = []
            
            precond_dir = self.file_processor.precondition_subtasks_path
            
            for node in integrated_dag["nodes"]:
                sid = node["id"]
                
                # precondition 파일에서 skills/objects 정보 추출
                skills = []
                objects = []
                
                # pre_XX_*.txt 파일 찾기
                for fname in os.listdir(precond_dir):
                    if fname.startswith(f"pre_{sid:02d}_"):
                        precond_path = os.path.join(precond_dir, fname)
                        try:
                            with open(precond_path, "r") as f:
                                content = f.read()
                                
                                # Skills/Objects 간단 추출 (정규식)
                                skills_match = re.search(r"Skills Required:\s*(.+)", content, re.IGNORECASE)
                                if skills_match:
                                    skills = [s.strip() for s in skills_match.group(1).split(",")]
                                
                                objects_match = re.search(r"Related Objects?:\s*(.+)", content, re.IGNORECASE)
                                if objects_match:
                                    objects = [o.strip() for o in objects_match.group(1).split(",")]
                        except Exception as e:
                            print(f"  Warning: Could not parse {fname}: {e}")
                        break
                
                parsed_subtasks.append({
                    "id": sid,
                    "title": node.get("name", f"Subtask_{sid}"),
                    "skills": skills,
                    "objects": objects
                })
            
            print(f"  ✓ Collected {len(parsed_subtasks)} subtasks")
            
            # 3. Plan actions 로드
            plan_actions_by_sid = self._load_plan_actions_by_subtask_id()
            
            # 4. Binding pairs 계산
            from LP_Module import binding_pairs_from_subtask_dag
            
            # subtask_dag 객체 형태로 변환
            class SubtaskDAGWrapper:
                def __init__(self, dag_dict):
                    self.nodes = dag_dict["nodes"]
                    self.edges = dag_dict["edges"]
                    self.parallel_groups = dag_dict["parallel_groups"]
            
            subtask_dag_obj = SubtaskDAGWrapper(integrated_dag)
            binding_pairs = binding_pairs_from_subtask_dag(subtask_dag_obj)
            
            print(f"  ✓ Computed {len(binding_pairs)} binding pairs")
            
            # 5. 로봇/오브젝트 위치 가져오기
            robot_positions = None
            object_positions = None
            
            if floor_plan is not None:
                robot_positions, object_positions = MultiRobotExecutor.spawn_and_get_positions(
                    floor_plan, len(task_robot_ids)
                )
                print(f"  ✓ Robot positions: {robot_positions}")
            
            # 6. LP 작업 할당 실행
            from LP_Module import assign_subtasks_cp_sat
            import robots
            
            assignment = assign_subtasks_cp_sat(
                subtasks=parsed_subtasks,
                robot_ids=task_robot_ids,
                robots_db=robots.robots,
                plan_actions_by_subtask=plan_actions_by_sid,
                objects_ai=self.objects_ai,
                binding_pairs=binding_pairs,
                robot_positions=robot_positions,
                object_positions=object_positions,
            )
            
            # 7. 할당 결과 출력 및 저장
            print(f"\n  [LP Result]")
            for sid, rid in sorted(assignment.items()):
                subtask_title = next((st["title"] for st in parsed_subtasks if st["id"] == sid), f"Subtask {sid}")
                print(f"    Subtask {sid} ({subtask_title}) → Robot {rid}")
            
            assignment_output = {
                "task_idx": task_idx,
                "agent_count": len(task_robot_ids),
                "assignment": {str(k): v for k, v in assignment.items()},
                "subtasks": [{"id": st["id"], "title": st["title"], "robot": assignment.get(st["id"])} for st in parsed_subtasks],
                "replanned": True  # 재계획 표시
            }
            
            if robot_positions is not None:
                assignment_output["robot_spawn_positions"] = {
                    str(k): list(v) for k, v in robot_positions.items()
                }
            
            assignment_path = os.path.join(self.resources_path, "dag_outputs", f"task_{task_idx}_assignment_REPLANNED.json")
            with open(assignment_path, "w") as f:
                json.dump(assignment_output, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Assignment saved: {assignment_path}")
            print(f"{'='*60}\n")
            
            return assignment
            
        except Exception as e:
            print(f"  ERROR: Failed to recompute assignment: {e}")
            import traceback
            traceback.print_exc()
            return None


    def reload_executor_with_integrated_dag(
        self,
        executor: Any,  # MultiRobotExecutor
        task_idx: int,
        task_name: str = "task"
    ) -> bool:
        """
        통합된 DAG와 새 할당으로 실행기 재로드
        
        Args:
            executor: MultiRobotExecutor 인스턴스
            task_idx: 작업 인덱스
            task_name: 작업 이름
        
        Returns:
            bool: 재로드 성공 여부
        """
        try:
            print(f"\n[Executor Reload] Reloading executor with integrated DAG...")
            
            # 1. 새 할당 로드
            assignment_path = os.path.join(
                self.resources_path,
                "dag_outputs",
                f"task_{task_idx}_assignment_REPLANNED.json"
            )
            
            if os.path.exists(assignment_path):
                with open(assignment_path, "r") as f:
                    assignment_data = json.load(f)
                
                # assignment을 int key로 변환
                executor.assignment = {
                    int(k): v for k, v in assignment_data["assignment"].items()
                }
                print(f"  ✓ Loaded new assignment with {len(executor.assignment)} subtasks")
            else:
                # Fallback: 원본 할당 사용
                print(f"  Warning: No replanned assignment found, using original")
                executor.load_assignment(task_idx)
            
            # 2. 새 DAG 로드
            executor.load_subtask_dag(task_name)
            print(f"  ✓ Loaded integrated DAG")
            
            # 3. 새 Plan actions 로드
            executor.load_plan_actions()
            print(f"  ✓ Loaded {len(executor.subtask_plans)} subtask plans")
            
            return True
            
        except Exception as e:
            print(f"  ERROR: Failed to reload executor: {e}")
            import traceback
            traceback.print_exc()
            return False

# 커맨드 명령어로 받은 정보 저장하는 함수
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, help="Path to BDDL file")
    parser.add_argument(
        "--floor-plan",
        type=int,
        required=False,
        help="Required unless --bddl-file is provided",
    )
    # SmartLLM-style inputs
    parser.add_argument("--task", type=int, help="Task index (SmartLLM-style)")
    parser.add_argument("--floorplan", type=int, help="Floorplan index (SmartLLM-style)")
    parser.add_argument("--config-file", type=str, default="config.json")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-4o",
        choices=["gpt-3.5-turbo", "gpt-4o", "gpt-3.5-turbo-16k"],
    )
    parser.add_argument(
        "--prompt-decompse-set",
        type=str,
        default="pddl_train_task_decomposesep",
        choices=["pddl_train_task_decompose"],
    )
    parser.add_argument(
        "--prompt-allocation-set",
        type=str,
        default="pddl_train_task_allocationsep",
        choices=["pddl_train_task_allocation"],
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="final_test",
        choices=["final_test"],
    )
    parser.add_argument("--log-results", type=bool, default=True)

    parser.add_argument(
        "--run-with-feedback",
        action="store_true",
        help="Run execution in simulator and on failure replan via Subtask/Central LLM (requires --floor-plan)",
    )
    parser.add_argument("--max-replan-retries", type=int, default=2, help="Max replan attempts in feedback loop (default: 2)")

    args = parser.parse_args()

    # Validation
    smartllm_mode = (args.task is not None) or (args.floorplan is not None)
    legacy_mode = args.floor_plan is not None
    if not args.bddl_file and not smartllm_mode and not legacy_mode:
        parser.error(
            "Either --bddl-file or --floor-plan or (--task and --floorplan) must be provided"
        )
    if smartllm_mode and (args.task is None or args.floorplan is None):
        parser.error(
            "When using SmartLLM-style inputs, both --task and --floorplan must be provided"
        )

    return args

def _parse_floorplan_number(scene_name: str) -> int:
    match = re.search(r"FloorPlan(\d+)", scene_name)
    if not match:
        raise ValueError(f"Invalid scene name (expected FloorPlan<number>): {scene_name}")
    return int(match.group(1))

def main():
    """메인 실행 함수"""
    try:
        # 커맨드로 받은 정보들 저장하는 함수 실행(arg.floor_plan = 15, arg.gpt_version = gpt4 같이 저장됨)
        args = parse_arguments()
        
        pdl_root = str(SCRIPT_DIR.parent)

        # task manager 객체 생성 및 초기화 설정
        task_manager = TaskManager(
            base_path=pdl_root,
            gpt_version=args.gpt_version,
            api_key_file=args.openai_api_key_file
        )
        
        if args.bddl_file:
            print("\nBDDL Data:")
        elif args.task is not None or args.floorplan is not None:
            auto = AutoConfig(config_file=args.config_file)
            auto.set_task(args.task)
            auto.set_floorplan(args.floorplan)
            auto.set_agents(args.num_agents)

            task_str = auto.task_string()
            cfg = auto.config()
            floor_plan_num = _parse_floorplan_number(cfg.scene)

            test_tasks = [task_str]
            robots_test_tasks = [list(range(1, args.num_agents + 1))]
            available_robot_skills = [
                _available_robot_skills_from_ids(robots_test_tasks[0])
            ]
            gt_test_tasks = [None]
            trans_cnt_tasks = [None]
            min_trans_cnt_tasks = [None]

            print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

            objects_ai = f"\n\nobjects = {PDDLUtils.get_ai2_thor_objects(floor_plan_num)}"
            task_manager.process_tasks(test_tasks, robots_test_tasks, objects_ai, floor_plan=floor_plan_num,
                run_with_feedback=getattr(args, "run_with_feedback", False),
                max_replan_retries=getattr(args, "max_replan_retries", 2))

            if args.log_results:
                task_manager.log_results(
                    task=test_tasks[0],
                    idx=0,
                    available_robots=available_robot_skills,
                    gt_test_tasks=gt_test_tasks,
                    trans_cnt_tasks=trans_cnt_tasks,
                    min_trans_cnt_tasks=min_trans_cnt_tasks,
                    objects_ai=objects_ai,
                )
        else:
            test_file = os.path.join("data", args.test_set, f"FloorPlan{args.floor_plan}.json")
            test_tasks, robots_test_tasks, available_robot_skills, gt_test_tasks, trans_cnt_tasks, min_trans_cnt_tasks = \
                task_manager.load_dataset(test_file)
            
            print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
            
            objects_ai = f"\n\nobjects = {PDDLUtils.get_ai2_thor_objects(args.floor_plan)}"
            task_manager.process_tasks(test_tasks, robots_test_tasks, objects_ai, floor_plan=args.floor_plan,
                run_with_feedback=getattr(args, "run_with_feedback", False),
                max_replan_retries=getattr(args, "max_replan_retries", 2))
            
            if args.log_results:
                for idx, task in enumerate(test_tasks):
                    task_manager.log_results(
                        task=task,
                        idx=idx,
                        available_robots=available_robot_skills,
                        gt_test_tasks=gt_test_tasks,
                        trans_cnt_tasks=trans_cnt_tasks,
                        min_trans_cnt_tasks=min_trans_cnt_tasks,
                        objects_ai=objects_ai,
                    )
            
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"Full error: {str(e.__class__.__name__)}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()