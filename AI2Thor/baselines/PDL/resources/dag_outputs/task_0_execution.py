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
ASSIGNMENT = {1: 5, 2: 3, 3: 1, 4: 1, 5: 4, 6: 4, 7: 2, 8: 5}  # subtask_id -> robot_id
PARALLEL_GROUPS = {0: [0, 1, 2, 6, 7, 8], 1: [3, 4, 5]}  # group_id -> [subtask_ids]
AGENT_COUNT = 5
SPAWN_POSITIONS = {1: (-0.25, 0.9009997248649597, 1.25), 2: (-2.5, 0.9009997844696045, 3.0), 3: (-3.5, 0.9009997844696045, 1.5), 4: (-1.75, 0.9009997248649597, 1.25), 5: (-1.25, 0.9009997844696045, 2.5)}  # LP에서 결정된 스폰 좌표

# --- Subtask Plans ---
SUBTASK_PLANS = {
    1: {
        'name': 'subtask_01_Move_the_Bowl_to_the_CounterTop',
        'robot_id': 5,
        'actions': ['gotoobject robot1 bowl (1)', 'pickupobject robot1 bowl diningtable (1)', 'gotoobject robot1 countertop (1)', 'drophandobject robot1 bowl countertop (1)'],
        'parallel_group': 0,
    },
    2: {
        'name': 'subtask_02_Move_the_Cup_to_the_Sink',
        'robot_id': 3,
        'actions': ['gotoobject robot1 cup (1)', 'pickupobject robot1 cup kitchen (1)', 'gotoobject robot1 sink (1)', 'drophandobject robot1 cup sink (1)'],
        'parallel_group': 0,
    },
    3: {
        'name': 'subtask_03_Move_the_Fork_to_the_Drawer',
        'robot_id': 1,
        'actions': ['gotoobject robot1 fork (1)', 'pickupobject robot1 fork diningtable (1)', 'gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)', 'drophandobject robot1 fork drawer (1)', 'closeobject robot1 drawer (1)'],
        'parallel_group': 1,
    },
    4: {
        'name': 'subtask_04_Move_the_Spoon_to_the_Drawer',
        'robot_id': 1,
        'actions': ['gotoobject robot1 spoon (1)', 'pickupobject robot1 spoon diningtable (1)', 'gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)', 'drophandobject robot1 spoon drawer (1)', 'closeobject robot1 drawer (1)'],
        'parallel_group': 1,
    },
    5: {
        'name': 'subtask_05_Move_the_Knife_to_the_Drawer',
        'robot_id': 4,
        'actions': ['gotoobject robot1 knife (1)', 'pickupobject robot1 knife sink (1)', 'gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)', 'drophandobject robot1 knife drawer (1)', 'closeobject robot1 drawer (1)'],
        'parallel_group': 1,
    },
    6: {
        'name': 'subtask_06_Move_the_Plate_to_the_Sink',
        'robot_id': 4,
        'actions': ['gotoobject robot1 plate (1)', 'pickupobject robot1 plate kitchen (1)', 'gotoobject robot1 sink (1)', 'drophandobject robot1 plate sink (1)'],
        'parallel_group': 0,
    },
    7: {
        'name': 'subtask_07_Move_the_Mug_to_the_CoffeeMachine',
        'robot_id': 2,
        'actions': ['gotoobject robot1 mug (1)', 'pickupobject robot1 mug diningtable (1)', 'gotoobject robot1 coffeemachine (1)', 'drophandobject robot1 mug coffeemachine (1)'],
        'parallel_group': 0,
    },
    8: {
        'name': 'subtask_08_Move_the_Pan_to_the_StoveBurner',
        'robot_id': 5,
        'actions': ['gotoobject robot1 pan (1)', 'pickupobject robot1 pan diningtable (1)', 'gotoobject robot1 stoveburner (1)', 'drophandobject robot1 pan stoveburner (1)'],
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