<div align="center">

# PDL: PDDL-DAG-LP Framework

**Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments**

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Based on LLaMAR](https://img.shields.io/badge/Based_on-LLaMAR_(NeurIPS_2024)-blue)](https://arxiv.org/abs/2407.10031)

</div>

A modular LLM-based framework for long-horizon multi-agent robot planning that combines **PDDL formal planning**, **DAG-based dependency analysis**, and **constraint programming (CP-SAT) for robot assignment**. Built on top of [LLaMAR](https://arxiv.org/abs/2407.10031), extended with explicit PDDL planning, parallelism detection, and feedback-driven replanning.

Supports two environments: **MAP-THOR** (AI2-THOR household tasks) and **Search & Rescue (SAR)**.

## Overview

<img width="16275" height="8520" alt="PDL Overview" src="https://github.com/user-attachments/assets/19130b55-428f-4be8-9239-6eda228e0b1e" />

## Architecture

PDL consists of five core modules that form a **Plan → Analyze → Assign → Execute → Feedback** pipeline:

### 1. `PDDL_Module` — Planner
- Decomposes natural language task instructions into subtasks via LLM
- Generates PDDL domain/problem files for each subtask
- Calls **FastDownward** (external PDDL planner) to produce action sequences
- Coordinates all downstream modules

### 2. `DAG_Module` — Dependency Analysis
- Analyzes PDDL action sequences to detect data/resource dependencies
- Builds a **SubtaskDAG** for inter-subtask ordering constraints
- Builds per-subtask **ActionDAGs** to identify intra-subtask parallelism
- Outputs dependency graphs as JSON (`task_SUBTASK_DAG.json`)

### 3. `LP_Module` — Robot Assignment (CP-SAT)
- Reads robot capabilities (skills, mass capacity) from `resources/robots.py`
- Solves a **constraint satisfaction problem** via Google OR-Tools (CP-SAT) to assign subtasks to robots
- Respects skill requirements, mass constraints, and temporal ordering from the DAG

### 4. `FeedbackLoopModule` — Replanner
- Maintains a **SharedTaskStateStore** with per-subtask state (`PENDING → RUNNING → SUCCESS/FAILED`)
- Executes subtask groups concurrently, respecting DAG precedence
- On failure: triggers **PartialReplanner** to redecompose and re-plan only the failed subtask
- Injects effects of successful subtasks as context for replanning

### 5. `MultiRobotExecutor` / `sar_executor` — Environment Interface
- Translates PDDL action sequences into environment API calls
- **AI2-THOR**: Generates and executes Python code for `PickupObject`, `PutObject`, etc.
- **SAR**: Maps PDDL actions to `NavigateTo`, `UseSupply`, `Carry`, `DropOff`, etc.
- Supports video recording and visualization

---

## Supported Environments

### MAP-THOR (AI2-THOR)
Household manipulation tasks across 30+ floorplans.

| Task Type | Description | Example |
|-----------|-------------|---------|
| Type 1 | Simple transport | "Put the bread, lettuce, and tomato in the fridge" |
| Type 2 | Bulk operations | "Open all the drawers" |
| Type 3 | Category operations | "Put all groceries in the fridge" |
| Type 4 | Complex organization | "Clear the table by placing items appropriately" |

### Search & Rescue (SAR)
Grid-world rescue simulation with fire suppression and person rescue.

- Agents: 1–10 robots with inventory constraints
- Actions: `NavigateTo`, `GetSupply`, `UseSupply`, `Carry`, `DropOff`, `StoreSupply`, `Explore`
- Scenes: 6+ predefined scenes with varying fire/rescue complexity

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `ai2thor==5.0.0` — AI2-THOR simulation
- `openai>=1.11.1` — LLM API (GPT-4o)
- `ortools` — CP-SAT constraint programming
- `torch==2.2.0`, `transformers==4.38.0` — ML utilities
- `opencv-python==4.9.0.80` — Vision utilities
- FastDownward PDDL planner (included in `AI2Thor/baselines/PDL/downward/`)

### 2. Set API Key

Save your OpenAI API key:

```bash
echo "your_openai_api_key" > AI2Thor/baselines/PDL/api_key.txt
```

---

## Usage

### MAP-THOR (AI2-THOR)

```bash
python AI2Thor/baselines/PDL/scripts/PDDL_Module.py \
    --task=0 \
    --floorplan=0 \
    --gpt-version=gpt-4o \
    --num-agents=2
```

With feedback-driven replanning:

```bash
python AI2Thor/baselines/PDL/scripts/PDDL_Module.py \
    --task=0 \
    --floorplan=0 \
    --num-agents=3 \
    --run-with-feedback \
    --max-replan-retries=2 \
    --record-video
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `0` | Task ID (see `configs/`) |
| `--floorplan` | `0` | FloorPlan number |
| `--gpt-version` | `gpt-4o` | LLM model (`gpt-4o`, `gpt-4o-mini`) |
| `--num-agents` | `2` | Number of robots (1–10) |
| `--run-with-feedback` | `False` | Enable feedback loop replanning |
| `--max-replan-retries` | `2` | Max replanning attempts per subtask |
| `--record-video` | `False` | Record execution video |
| `--config-file` | `config_type1.json` | Task configuration file |


### Search & Rescue (SAR)

```bash
python SAR/PDL_SAR/planning_sar.py \
    --scene=1 \
    --agents=3 \
    --gpt-version=gpt-4o \
    --with-feedback \
    --seed=42
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `1` | Scene ID (1–6+) |
| `--agents` | `2` | Number of agents (1–10) |
| `--gpt-version` | `gpt-4o` | LLM model |
| `--with-feedback` | `False` | Enable feedback loop |
| `--seed` | `42` | Random seed |

---

## Execution Flow

```
      Task + Environment State
             ↓
      [PDDL_Module]
      Decompose into subtasks via LLM
      Generate PDDL problem per subtask
             ↓
      [FastDownward]
      Solve PDDL → action sequences
             ↓
      [DAG_Module]
      Build SubtaskDAG (inter-subtask deps)
      Build ActionDAGs (intra-subtask parallelism)
             ↓
      [LP_Module (CP-SAT)]
      Assign subtasks to robots
      (respecting skills, mass, temporal order)
             ↓
      [FeedbackLoopModule]
      Execute subtask groups concurrently
             ↓
      [MultiRobotExecutor / sar_executor]
      Run actions in AI2-THOR / SAR
      Collect execution results
             ↓
      Failure detected?
         Yes → [PartialReplanner] → retry
         No  → Task Complete ✓
```

---

## Repo Structure

```
PDL/
├── AI2Thor/
│   ├── baselines/
│   │   └── PDL/
│   │       ├── scripts/
│   │       │   ├── PDDL_Module.py          # Main orchestrator
│   │       │   ├── DAG_Module.py           # Dependency analysis
│   │       │   ├── LP_Module.py            # Robot assignment (CP-SAT)
│   │       │   ├── FeedbackLoopModule.py   # Fault-tolerant execution
│   │       │   └── MultiRobotExecutor.py   # AI2-THOR interface
│   │       ├── resources/
│   │       │   ├── robots.py               # Robot capability definitions
│   │       │   └── actions.py              # AI2-THOR action definitions
│   │       ├── downward/                   # FastDownward PDDL planner
│   │       └── logs/                       # Execution logs & plans
│   ├── Tasks/                              # Task & scene definitions
│   └── summarisers/                        # Observation summarizers
├── SAR/
│   ├── PDL_SAR/
│   │   ├── planning_sar.py                 # SAR pipeline entry point
│   │   └── sar_executor.py                 # SAR environment interface
│   ├── Scenes/                             # SAR scene definitions
│   ├── core.py                             # SAR environment core
│   └── env.py                              # SAR environment interface
├── configs/                                # Task configuration files
├── thortils/                               # AI2-THOR utilities
├── vlms/                                   # Open-source VLM integrations
├── plots/                                  # Analysis & plotting scripts
├── results/                                # Experiment results
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Robot Capabilities

Robots are defined in `AI2Thor/baselines/PDL/resources/robots.py` with:
- **Skills**: subset of `[GoToObject, OpenObject, CloseObject, PickupObject, PutObject, DropHandObject, ...]`
- **Mass Capacity**: maximum object weight each robot can carry

The CP-SAT assignment ensures each subtask is assigned to a robot that has the required skills and sufficient capacity.

---

## Based On

This project extends [LLaMAR](https://arxiv.org/abs/2407.10031) (NeurIPS 2024):

```bibtex
@inproceedings{llamar,
  title={Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments},
  author={Nayak, Siddharth and Orozco, Adelmo Morrison and Ten Have, Marina and Zhang, Jackson
          and Thirumalai, Vittal and Chen, Darren and Kapoor, Aditya and Robinson, Eric
          and Gopalakrishnan, Karthik and Harrison, James and Ichter, Brian
          and Mahajan, Anuj and Balakrishnan, Hamsa},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

---

## License

MIT License
