import os
import re
from pathlib import Path
import subprocess
import argparse

# MAP-THOR: task description → task folder mapping (mirrors PDL's _TASK_NAME_MAP)
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


def append_trans_ctr(allocated_plan):
    brk_ctr = 0
    code_segs = allocated_plan.split("\n\n")
    fn_calls = []
    for cd in code_segs:
        if "def" not in cd and "threading.Thread" not in cd and "join" not in cd and cd[-1] == ")":
            # fn_calls.append(cd)
            brk_ctr += 1
    print ("No Breaks: ", brk_ctr)
    return brk_ctr

def compile_aithor_exec_file(expt_name):
    log_path = os.getcwd() + "/logs/" + expt_name
    executable_plan = ""

    # append the imports to the file
    import_file = Path(os.getcwd() + "/data/aithor_connect/imports_aux_fn.py").read_text()
    executable_plan += (import_file + "\n")

    # parse log.txt
    log_file = open(log_path + "/log.txt")
    log_data = log_file.readlines()
    log_file.close()

    # ── robots ──────────────────────────────────────────────────────────────
    robots_line = None
    for line in log_data:
        if "robots = " in line:
            robots_line = line.strip()
            break

    if robots_line:
        executable_plan += (robots_line + "\n")
    else:
        executable_plan += (
            "robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', "
            "'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'CleanObject', 'PickupObject', "
            "'PutObject', 'PutObjectInFridge', 'DropHandObject', 'ThrowObject', 'PushObject', "
            "'PullObject'], 'mass': 100}]\n"
        )

    # ── floor_no ─────────────────────────────────────────────────────────────
    # pddlrun_llmseparate.py writes "floor_no = X" to log.txt.
    # Fallback: parse "Floor Plan: X" (legacy run_llm.py format), then default to 1.
    floor_no = 1
    for line in log_data:
        m = re.search(r"^floor_no\s*=\s*(\d+)", line)
        if m:
            floor_no = int(m.group(1))
            break
        m = re.search(r"Floor Plan:\s*(\d+)", line)
        if m:
            floor_no = int(m.group(1))
            break

    executable_plan += (f"floor_no = {floor_no}\n\n")

    # ── task_folder ──────────────────────────────────────────────────────────
    # pddlrun_llmseparate.py writes "task_folder = '...'" to log.txt.
    # Fallback: map the task description (first line) via _TASK_NAME_MAP.
    task_folder = None
    for line in log_data:
        m = re.search(r'^task_folder\s*=\s*[\'"](.+?)[\'"]', line)
        if m:
            task_folder = m.group(1)
            break

    if task_folder is None:
        # derive from first line (task description)
        task_desc = log_data[0].strip() if log_data else ""
        task_folder = (
            _TASK_NAME_MAP.get(task_desc)
            or _TASK_NAME_MAP_LOWER.get(task_desc.lower())
        )

    if task_folder:
        executable_plan += (f"task_folder = '{task_folder}'\n\n")
    else:
        executable_plan += ("task_folder = None  # task not found in MAP-THOR task map\n\n")

    # ── aithor connector and helper functions ────────────────────────────────
    connector_file = Path(os.getcwd() + "/data/aithor_connect/aithor_connect.py").read_text()
    executable_plan += (connector_file + "\n")

    # ── LLM-generated code plan ──────────────────────────────────────────────
    # Execution must use only validated code_plan.py generated by plantocode.py.
    _code_plan_path = Path(log_path) / "code_plan.py"
    if not _code_plan_path.exists():
        raise FileNotFoundError(
            f"Missing validated plan file: {_code_plan_path}. "
            "Run scripts/plantocode.py first and ensure validation passes."
        )
    allocated_plan = _code_plan_path.read_text()

    # Strip conflicting initialization lines injected by plantocode.py.
    # These variables are already defined by aithor_connect.py earlier in the file.
    _conflict_re = re.compile(
        r'^(?:import\s+(?:threading|time)\b'       # already imported
        r'|action_queue\s*=\s*\[\]'                # already defined
        r'|task_over\s*=\s*False'                  # already defined
        r'|robots\s*=\s*\[robot\w'                 # robots = [robot1, ...] (no quotes)
        r"|robots\s*=\s*\['robot\w)"               # robots = ['robot1', ...] (with quotes)
    )
    allocated_plan = '\n'.join(
        line for line in allocated_plan.splitlines()
        if not _conflict_re.match(line.strip())
    )

    # Replace empty robots list with actual robots from log file
    if robots_line:
        allocated_plan = allocated_plan.replace("robots = []", robots_line.strip())
        allocated_plan = allocated_plan.replace("robots = ['robot1']", robots_line.strip())
        allocated_plan = allocated_plan.replace("robots = ['Robot2']", robots_line.strip())

    brks = append_trans_ctr(allocated_plan)
    executable_plan += (allocated_plan + "\n")
    executable_plan += ("no_trans = " + str(brks) + "\n")

    # ── task thread termination and evaluation ───────────────────────────────
    terminate_plan = Path(os.getcwd() + "/data/aithor_connect/end_thread.py").read_text()
    executable_plan += (terminate_plan + "\n")

    with open(f"{log_path}/executable_plan.py", 'w') as d:
        d.write(executable_plan)

    return (f"{log_path}/executable_plan.py")

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, required=True)
args = parser.parse_args()

expt_name = args.command
print (expt_name)
ai_exec_file = compile_aithor_exec_file(expt_name)

subprocess.run(["python", ai_exec_file])
