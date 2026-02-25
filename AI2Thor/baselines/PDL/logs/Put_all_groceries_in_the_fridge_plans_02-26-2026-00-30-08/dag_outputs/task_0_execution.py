#!/usr/bin/env python3
"""
Auto-generated multi-robot execution code.
Run with: python <this_file> --floor-plan <N>
"""

import argparse
import sys
import os

import sys
from pathlib import Path
PDL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PDL_ROOT))
sys.path.insert(0, str(PDL_ROOT / "scripts"))
sys.path.insert(0, str(PDL_ROOT / "resources"))
# Add scripts folder to path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_path = os.path.join(base_path, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from MultiRobotExecutor import MultiRobotExecutor, SubtaskPlan

# --- Robot Assignment ---
ASSIGNMENT = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1}  # subtask_id -> robot_id
PARALLEL_GROUPS = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5]}  # group_id -> [subtask_ids]
AGENT_COUNT = 4
SPAWN_POSITIONS = {1: (-1.0, 0.900999128818512, 0.25), 2: (1.5, 0.900999128818512, 0.25), 3: (0.25, 0.900999128818512, -1.5), 4: (1.0, 0.900999128818512, 2.0)}  # LP에서 결정된 스폰 좌표

# --- Subtask Plans ---
SUBTASK_PLANS = {
    1: {
        'name': 'subtask_01_Put_the_Apple_in_the_Fridge',
        'robot_id': 4,
        'actions': ['gotoobject robot1 apple (1)', 'pickupobject robot1 apple countertop (1)', 'gotoobject robot1 fridge (1)', 'openfridge robot1 fridge (1)', 'putobjectinfridge robot1 apple fridge (1)', 'closeobject robot1 fridge (1)'],
        'parallel_group': 0,
    },
    2: {
        'name': 'subtask_02_Put_the_Bottle_in_the_Fridge',
        'robot_id': 3,
        'actions': ['gotoobject robot1 bottle (1)', 'pickupobject robot1 bottle shelf (1)', 'gotoobject robot1 fridge (1)', 'openfridge robot1 fridge (1)', 'putobjectinfridge robot1 bottle fridge (1)', 'closeobject robot1 fridge (1)'],
        'parallel_group': 1,
    },
    3: {
        'name': 'subtask_03_Put_the_Bread_in_the_Fridge',
        'robot_id': 2,
        'actions': ['gotoobject robot1 bread (1)', 'pickupobject robot1 bread countertop (1)', 'gotoobject robot1 fridge (1)', 'openfridge robot1 fridge (1)', 'putobjectinfridge robot1 bread fridge (1)', 'closeobject robot1 fridge (1)'],
        'parallel_group': 2,
    },
    4: {
        'name': 'subtask_04_Put_the_Lettuce_in_the_Fridge',
        'robot_id': 1,
        'actions': ['gotoobject robot1 lettuce (1)', 'pickupobject robot1 lettuce countertop (1)', 'gotoobject robot1 fridge (1)', 'openfridge robot1 fridge (1)', 'putobjectinfridge robot1 lettuce fridge (1)', 'closeobject robot1 fridge (1)'],
        'parallel_group': 3,
    },
    5: {
        'name': 'subtask_05_Put_the_Egg_in_the_Fridge',
        'robot_id': 1,
        'actions': ['gotoobject robot1 egg (1)', 'pickupobject robot1 egg floor (1)', 'gotoobject robot1 fridge (1)', 'openfridge robot1 fridge (1)', 'putobjectinfridge robot1 egg fridge (1)', 'closeobject robot1 fridge (1)'],
        'parallel_group': 4,
    },
}

TASK_DESCRIPTION = 'Put all groceries in the fridge'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor-plan', type=int, default=1)
    args = parser.parse_args()
    
    executor = MultiRobotExecutor(base_path)
    executor.assignment = ASSIGNMENT
    executor.parallel_groups = PARALLEL_GROUPS
    
    # Reconstruct subtask_plans
    for sid, data in SUBTASK_PLANS.items():
        executor.subtask_plans[sid] = SubtaskPlan(
            subtask_id=sid,
            subtask_name=data['name'],
            robot_id=data['robot_id'],
            actions=data['actions'],
            parallel_group=data['parallel_group'],
        )
    
    # Execute in AI2-THOR
    executor.execute_in_ai2thor(
        floor_plan=args.floor_plan,
        task_description=TASK_DESCRIPTION if 'TASK_DESCRIPTION' in globals() else None,
        agent_count=AGENT_COUNT,
        spawn_positions=SPAWN_POSITIONS,
    )


if __name__ == '__main__':
    main()