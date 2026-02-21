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
ASSIGNMENT = {1: 5, 2: 1, 3: 1, 4: 2, 5: 4, 6: 5, 7: 2, 8: 3, 9: 4}  # subtask_id -> robot_id
PARALLEL_GROUPS = {0: [5], 1: [7], 2: [9]}  # group_id -> [subtask_ids]
AGENT_COUNT = 5
SPAWN_POSITIONS = {1: (0.5, 0.900999128818512, -1.25), 2: (-0.25, 0.900999128818512, 1.75), 3: (1.25, 0.900999128818512, 1.25), 4: (-1.25, 0.900999128818512, -1.25), 5: (2.0, 0.900999128818512, -0.25)}  # LP에서 결정된 스폰 좌표

# --- Subtask Plans ---
SUBTASK_PLANS = {
    1: {
        'name': 'subtask_01_Open_the_first_Drawer',
        'robot_id': 5,
        'actions': [],
        'parallel_group': 0,
    },
    2: {
        'name': 'subtask_02_Open_the_second_Drawer',
        'robot_id': 1,
        'actions': [],
        'parallel_group': 0,
    },
    3: {
        'name': 'subtask_03_Open_the_third_Drawer',
        'robot_id': 1,
        'actions': [],
        'parallel_group': 0,
    },
    4: {
        'name': 'subtask_04_Open_the_fourth_Drawer',
        'robot_id': 2,
        'actions': [],
        'parallel_group': 0,
    },
    5: {
        'name': 'subtask_05_Open_the_fifth_Drawer',
        'robot_id': 4,
        'actions': ['gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)'],
        'parallel_group': 0,
    },
    6: {
        'name': 'subtask_06_Open_the_sixth_Drawer',
        'robot_id': 5,
        'actions': [],
        'parallel_group': 0,
    },
    7: {
        'name': 'subtask_07_Open_the_seventh_Drawer',
        'robot_id': 2,
        'actions': ['gotoobject robot1 drawer7 (1)', 'openobject robot1 drawer7 (1)'],
        'parallel_group': 1,
    },
    8: {
        'name': 'subtask_08_Open_the_eighth_Drawer',
        'robot_id': 3,
        'actions': [],
        'parallel_group': 0,
    },
    9: {
        'name': 'subtask_09_Open_the_ninth_Drawer',
        'robot_id': 4,
        'actions': ['gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)'],
        'parallel_group': 2,
    },
}

TASK_DESCRIPTION = 'Open all the drawers'


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