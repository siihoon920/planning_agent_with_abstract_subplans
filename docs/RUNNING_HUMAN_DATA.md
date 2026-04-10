# Running Human Data Examples & Evaluation

This document explains how to run the human-data inference scripts and the
cross-model evaluation pipeline for the two experimental domains used in the
`"Modeling the Mistakes of Boundedly Rational Agents Within a Bayesian Theory
of Mind"` paper.

---

## Overview of Scripts

| Script | Domain | What it does |
|--------|--------|--------------|
| `examples/block-words/human_example.jl` | Block-Words | Runs the original **SIPS** model on a human trajectory, renders a storyboard, saves model goal-prob CSV |
| `examples/doors-keys-gems/human_example.jl` | Doors-Keys-Gems | Same as above for DKG domain |
| `examples/doors-keys-gems/human_example_abstract.jl` | Doors-Keys-Gems | Runs the **Hierarchical Abstract Planner** model on a human trajectory, produces identical outputs |
| `examples/evaluate.jl` | Either | Loads output CSVs from the above scripts, computes Pearson r / MSE / cross-entropy vs human data, prints a comparison table |

> [!NOTE]
> The hierarchical abstract planner (`src_new/`) is only applicable to the
> **Doors-Keys-Gems** domain because it relies on identifying `pickup` and
> `unlock` as natural abstract subgoal actions. It cannot be applied to the
> Block-Words domain.

---

## Environment Setup

Before running anything, activate the project and install dependencies:

```bash
# Activate and instantiate the Julia environment
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

All scripts must be run from the **repository root**.

---

## Experiment Selection & Format Conversion (Doors-Keys-Gems)

### Why only 8 of 16 experiments?

The original human experiment data covers 16 trials across 12 distinct map
problems (problems 1–12). However, the `master` branch of this codebase uses a
**new PDDL format** (bit-matrix walls, door objects) that is only compatible with
PDDL.jl ≥ 0.2.x. New-format problem files exist in `examples/doors-keys-gems/problems/`
only for problems 4–7. The remaining problems (8–12) do not have new-format
counterparts, so their experiments cannot be run with the current codebase.

**8 experiments kept** (problems 4–7, new-format problem files available):

| Experiment | Problem | True Goal |
|------------|---------|-----------|
| `1_1` | 6 | gem2 |
| `1_2` | 7 | gem1 |
| `1_3` | 4 | gem1 |
| `2_2` | 5 | gem1 |
| `3_1` | 6 | gem1 |
| `3_2` | 4 | gem2 |
| `4_1` | 7 | gem2 |
| `4_2` | 5 | gem3 |

**8 experiments dropped** (problems 8–12, no new-format problem files):
`1_4`, `2_1`, `2_3`, `2_4`, `3_3`, `3_4`, `4_3`, `4_4`

### What was converted and why?

The original human plan files in `domains/doors-keys-gems/plans/` were recorded
against the **old PDDL format** (from the `project-blocks` branch), which differs
from the `master` format in two ways:

**1. Coordinate system:**
- Old format: `y=1` is the bottom row, `y` increases upward
- New format: `y=1` is the top row, `y` increases downward
- Transform: `new_y = height + 1 - old_y` (height = 9 for all maps)
- `x` is unchanged

**2. Unlock action syntax:**
- Old format: `(unlock key direction)` — e.g., `(unlock key2 left)`
  meaning "unlock the door one step to the left of the agent"
- New format: `(unlock key doorN)` — e.g., `(unlock key2 door2)`
  meaning "unlock door2 (a named object)"

To convert, each `(unlock key dir)` action was traced through the trajectory
to determine which physical door the agent was adjacent to, and that door was
mapped to its name in the new problem file (`door1`, `door2`, etc.).

Converted plan files are stored in `examples/doors-keys-gems/plans/`.
Movement actions (`(up)`, `(down)`, etc.) and `(pickup ...)` actions are
identical between formats and required no changes.

**3. Gem naming in problem files:**

When the new-format problem files were originally created, gems were assigned
names in a different order than the old format for problems 4 and 6. Since the
plan files reference gems by name (e.g., `(pickup gem1)`), the gem coordinates
in the new problem files were corrected to match the old naming convention:

- `examples/doors-keys-gems/problems/problem-4.pddl`: gem2 and gem3 positions swapped
- `examples/doors-keys-gems/problems/problem-6.pddl`: gem1 and gem3 positions swapped

This does not affect experimental validity — only internal labels were changed,
not physical positions, trajectories, or human judgment data.

---

## Experiment IDs

Both domains have 16 experiments named `<category>_<variant>`:

| ID | Description |
|----|-------------|
| `1_1`, `1_2`, `1_3`, `1_4` | Experiment category 1 |
| `2_1`, `2_2`, `2_3`, `2_4` | Experiment category 2 |
| `3_1`, `3_2`, `3_3`, `3_4` | Experiment category 3 |
| `4_1`, `4_2`, `4_3`, `4_4` | Experiment category 4 |

If no experiment ID is supplied, scripts default to `1_1`.

---

## Script Details

### 1. `examples/block-words/human_example.jl`

**What it loads:**
- `domains/block-words/domain.pddl` — PDDL domain definition
- `domains/block-words/experiment-<X>-<Y>.pddl` — Initial state for that experiment (e.g., `experiment-1-1.pddl`)
- `domains/block-words/experiment-scenarios.jl` — Hardcoded action sequences (`get_action`) and goal word sets (`get_goal_space`) for every scenario
- `domains/block-words/average_human_results_arrays/<exp_id>.csv` — Human goal-probability data (each file contains a flat 1D array of `n_goals × T` floats)

**What it does:**
1. Loads the PDDL problem and compiles the domain.
2. Replays the action sequence from `experiment-scenarios.jl` to recover the observed trajectory.
3. Constructs a `WorldConfig` using the standard **SIPS** model (`ProbAStarPlanner` + particle filter).
4. Runs SIPS with 120 particles and logs goal probabilities at each timestep.
5. Saves the goal-probability output as `examples/block-words/model_sips_goal_probs_<exp_id>.csv`.
6. Overlays the human data on a visual storyboard and saves a PNG.

**Outputs:**
- `examples/block-words/human_trajectory_<exp_id>.gif`
- `examples/block-words/human_goal_inference_storyboard_<exp_id>.png`
- `examples/block-words/model_sips_goal_probs_<exp_id>.csv` ← used by `evaluate.jl`

**How to run:**
```bash
julia examples/block-words/human_example.jl 1_1
# Default (no argument) also runs 1_1
julia examples/block-words/human_example.jl
```

---

### 2. `examples/doors-keys-gems/human_example.jl` (SIPS model)

**What it loads:**
- `domains/doors-keys-gems/domain.pddl`
- `domains/doors-keys-gems/problem-<N>.pddl` — The problem number is embedded in the plan filename (e.g., `1_1_problem_6_goal1_0.dat` → `problem-6.pddl`)
- `domains/doors-keys-gems/plans/<exp_id>_*.dat` — Pre-recorded human action sequence, one action per line
- `domains/doors-keys-gems/average_human_results_arrays/<exp_id>.csv` — Human goal probabilities

**What it does:**
1. Scans the `plans/` directory for a `.dat` file whose name starts with `<exp_id>_`.
2. Parses the problem number from that filename and loads the matching PDDL problem.
3. Reads the plan (one PDDL action string per line) and simulates the trajectory.
4. Runs the standard **SIPS** particle filter (120 particles, `RelaxedMazeDist` heuristic).
5. Saves outputs including a model goal-probability CSV.

**Outputs:**
- `examples/doors-keys-gems/human_trajectory_<exp_id>.gif`
- `examples/doors-keys-gems/human_goal_inference_storyboard_<exp_id>.png`
- `examples/doors-keys-gems/model_sips_goal_probs_<exp_id>.csv` ← used by `evaluate.jl`

**How to run:**
```bash
julia examples/doors-keys-gems/human_example.jl 2_3
```

---

### 3. `examples/doors-keys-gems/human_example_abstract.jl` (Abstract Planner model)

**What it loads:** Same files as `human_example.jl` above.

**What it does differently:**
Instead of using the standard `ProbAStarPlanner` directly inside SIPS, this
script uses a **`HierarchicalPlanner`** wrapper (defined in the script itself)
that dispatches planning calls to `AbstractPlanner.solve(...)` from `src_new/`.

The hierarchical planning architecture works as follows:
- **Low-level (`src_new/PhysicalPlanner.jl`)**: From any state, runs a Dijkstra
  search over all physically reachable states, stopping at abstract subgoals
  (any `pickup` or `unlock` action). Returns a set of low-level plans &
  trajectories to each discovered subgoal.
- **High-level (`src_new/AbstractPlanner.jl`)**: Runs an A\* search over the
  abstract state space. Each node in this search is an abstract state (reachable
  via some chain of subgoal actions). When expanding a node, it calls the
  physical planner to discover candidate next abstract states. The heuristic used
  is `GoalManhattan`.

This means the agent reasons about goals at a hierarchical level (subgoal
sequences like `pickup_key → unlock_door → pickup_gem`) rather than at the level
of individual movement steps.

SIPS is still used for particle-filter goal inference — only the internal planner
changes.

**Outputs:**
- `examples/doors-keys-gems/human_trajectory_abstract_<exp_id>.gif`
- `examples/doors-keys-gems/human_goal_inference_storyboard_abstract_<exp_id>.png`
- `examples/doors-keys-gems/model_abstract_goal_probs_<exp_id>.csv` ← used by `evaluate.jl`

**How to run:**
```bash
julia examples/doors-keys-gems/human_example_abstract.jl 1_1
```

---

### 4. `examples/evaluate.jl` (Cross-Model Evaluation)

> [!IMPORTANT]
> **You must run the model scripts first.** `evaluate.jl` reads the CSV files
> saved by `human_example.jl` and `human_example_abstract.jl`. If those files
> don't exist, the script will warn and skip the missing model.

**What it does:**
1. Loads the model goal-probability CSVs from both the SIPS and (optionally)
   the abstract planner scripts.
2. Loads the corresponding human CSV from `domains/<domain>/average_human_results_arrays/`.
3. Aligns model and human data to the same number of timesteps (takes the minimum).
4. Computes three metrics for each model, both per-goal and overall:

| Metric | Formula | Better |
|--------|---------|--------|
| **Pearson r** | correlation(model, human) | Higher (max +1) |
| **MSE** | mean squared error | Lower (min 0) |
| **Cross-entropy** | −Σ human_i · log(model_i) | Lower (min 0) |

5. Prints a formatted comparison table.
6. Saves a results CSV to `examples/evaluation_results_<domain>_<exp_id|all>.csv`.

**How to run — single experiment:**
```bash
# SIPS vs Abstract, doors-keys-gems, experiment 1_1
julia examples/evaluate.jl doors-keys-gems 1_1

# SIPS only (no abstract), block-words, experiment 2_3
julia examples/evaluate.jl block-words 2_3
```

**How to run — all experiments:**
```bash
julia examples/evaluate.jl doors-keys-gems all
julia examples/evaluate.jl block-words     all
```

**Example output (doors-keys-gems):**
```
══════════════════════════════════════════════════════════════════════════════════
Exp        SIPS-r  SIPS-MSE   SIPS-CE    Abs-r   Abs-MSE    Abs-CE
──────────────────────────────────────────────────────────────────────────────────
1_1        0.8732    0.0123    1.4521   0.7901    0.0215    1.8902
1_2        0.7415    0.0201    1.5801   ...
...
──────────────────────────────────────────────────────────────────────────────────
MEAN       0.8100    0.0170    1.5300   0.7700    0.0220    1.7500
══════════════════════════════════════════════════════════════════════════════════

Legend:
  r   = Pearson correlation with human data (higher is better, max 1)
  MSE = Mean Squared Error                  (lower  is better, min 0)
  CE  = Cross-Entropy of human under model  (lower  is better, min 0)

  SIPS = Original SIPS particle-filter model
  Abs  = Hierarchical Abstract Planner model
```

---

## Recommended Workflow

Run the following in sequence for a full comparison on one experiment:

```bash
# Step 1 — Run the SIPS baseline
julia examples/doors-keys-gems/human_example.jl 1_1

# Step 2 — Run the Abstract Planner model
julia examples/doors-keys-gems/human_example_abstract.jl 1_1

# Step 3 — Compare both models against human data
julia examples/evaluate.jl doors-keys-gems 1_1
```

To run all 16 experiments for both models (long-running):
```bash
for exp in 1_1 1_2 1_3 1_4 2_1 2_2 2_3 2_4 3_1 3_2 3_3 3_4 4_1 4_2 4_3 4_4; do
    julia examples/doors-keys-gems/human_example.jl          $exp
    julia examples/doors-keys-gems/human_example_abstract.jl $exp
done
julia examples/evaluate.jl doors-keys-gems all
```

---

## Key Files at a Glance

```
domains/
  block-words/
    domain.pddl                         # PDDL domain definition
    experiment-{X}-{Y}.pddl             # Initial state per experiment
    experiment-scenarios.jl             # Hardcoded action lists + goal spaces
    average_human_results_arrays/       # Human goal-probability CSVs
      1_1.csv, 1_2.csv, ... 4_4.csv

  doors-keys-gems/
    domain.pddl
    problem-{N}.pddl                    # PDDL problems (referenced by plan files)
    plans/
      1_1_problem_6_goal1_0.dat         # Pre-recorded human action sequences
      ...
    average_human_results_arrays/       # Human goal-probability CSVs
      1_1.csv, ... 4_4.csv

src/                                    # Original SIPS model (Plinf.jl)
src_new/
  PhysicalPlanner.jl                    # Dijkstra subgoal discovery
  AbstractPlanner.jl                    # Hierarchical A* over abstract states
  abstract_example.jl                   # Standalone planning example (no inference)

examples/
  block-words/
    human_example.jl                    # SIPS on block-words human data
    model_sips_goal_probs_<id>.csv      # Model output (generated)
  doors-keys-gems/
    human_example.jl                    # SIPS on DKG human data
    human_example_abstract.jl           # Abstract Planner on DKG human data
    model_sips_goal_probs_<id>.csv      # SIPS output (generated)
    model_abstract_goal_probs_<id>.csv  # Abstract planner output (generated)
  evaluate.jl                           # Cross-model evaluation script
  evaluation_results_<domain>_<id>.csv  # Evaluation results (generated)
```
