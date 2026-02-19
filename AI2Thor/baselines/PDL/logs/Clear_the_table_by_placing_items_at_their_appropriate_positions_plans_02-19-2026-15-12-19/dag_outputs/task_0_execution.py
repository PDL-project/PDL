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
ASSIGNMENT = {1: 4, 2: 1, 3: 5, 4: 3, 5: 2, 6: 1}  # subtask_id -> robot_id
PARALLEL_GROUPS = {}  # group_id -> [subtask_ids]
AGENT_COUNT = 5
SPAWN_POSITIONS = {1: (-1.5, 0.9009997844696045, 3.0), 2: (-1.5, 0.9009996652603149, 1.0), 3: (-3.0, 0.9009997844696045, 2.0), 4: (-0.25, 0.9009997248649597, 1.25), 5: (-2.75, 0.9009997844696045, 3.25)}  # LP에서 결정된 스폰 좌표

# --- Subtask Plans ---
SUBTASK_PLANS = {
    1: {
        'name': 'subtask_01_Move_the_Bowl_to_the_CounterTop',
        'robot_id': 4,
        'actions': [],
        'parallel_group': 0,
    },
    2: {
        'name': 'subtask_02_Move_the_Bread_to_the_CounterTop',
        'robot_id': 1,
        'actions': [],
        'parallel_group': 0,
    },
    3: {
        'name': 'subtask_03_Move_the_Fork_to_the_Drawer',
        'robot_id': 5,
        'actions': [],
        'parallel_group': 0,
    },
    4: {
        'name': 'subtask_04_Move_the_Spoon_to_the_Drawer',
        'robot_id': 3,
        'actions': [],
        'parallel_group': 0,
    },
    5: {
        'name': 'subtask_05_Move_the_Pan_to_the_StoveBurner',
        'robot_id': 2,
        'actions': [],
        'parallel_group': 0,
    },
    6: {
        'name': 'subtask_06_Move_the_Tomato_to_the_Fridge',
        'robot_id': 1,
        'actions': [],
        'parallel_group': 0,
    },
}

TASK_DESCRIPTION = 'Clear the table by placing items at their appropriate positions'


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