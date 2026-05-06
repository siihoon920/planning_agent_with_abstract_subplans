# CLAUDE.md — planning_agent_with_abstract_subplans

## Project Overview

This project extends [Plinf.jl](https://github.com/ztangent/Plinf.jl) with a **hierarchical (abstract) planner** for Bayesian goal inference in the Doors-Keys-Gems gridworld. The core idea: rather than replanning with a flat A* planner, the agent reasons over abstract subgoals (key pickups, door unlocks, gem collection), and the particle filter infers human goals by weighing trajectories against this hierarchical planning model.

The main comparison is:
- **SIPS** (baseline): flat `ProbAStarPlanner` from Plinf
- **Hierarchical SIPS**: `AbstractPlanner` → `PhysicalPlanner` two-level search

---

## Julia Version Requirement

**Julia 1.6 only.** The PDDL/SymbolicPlanners/GLMakie versions used here are incompatible with newer Julia.

```bash
juliaup add 1.6
juliaup default 1.6
```

Activate the environment inside Julia REPL:
```
] activate .
] instantiate
```

---

## Directory Structure

```
src_new/
  AbstractPlanner.jl        # AbstractPlanner struct + SIPS-compatible solve()
  PhysicalPlanner.jl        # Dijkstra physical search, PHYSICAL_CACHE
  main_with_hierarchical.jl # Main script: runs Hierarchical SIPS for all 16 experiments
  main_multi_run.jl         # Runs both SIPS and Hierarchical SIPS for 5 seeded runs each
  main_sips_budget001.jl    # SIPS only with budget_dist_args=(2, 0.01, 1)
  make_storyboards.jl       # Regenerates storyboards from existing CSVs (no inference)
  debug_hierarchical.jl     # Debug/scratch script

example/doors-keys-gems/
  domain.pddl               # New-format PDDL domain (bit-mat walls, door objects)
  problems_modified/        # New-format problem files (problem-1..12.pddl) — ALL 12 NOW PRESENT
  utils.jl                  # Renderer + helper utilities
  solutions/
    trajectories/           # GIF animations per experiment
    storyboard_human/       # Human data storyboards
    storyboard_SIPS/        # SIPS model storyboards
    storyboard_hierarchical/# Hierarchical model storyboards
  goal_probs_SIPS/          # goal_probs_SIPS_<exp_id>.csv  +  _run_N variants
  goal_probs_hierarchical/  # goal_probs_hierarchical_<exp_id>.csv  +  _run_N variants
  goal_probs_SIPS_budget001/# goal_probs_SIPS_budget001_<exp_id>.csv (budget_p=0.01)

domains/doors-keys-gems/
  plans/                    # Human plan files (<exp_id>_problem_<N>_goal<K>_<P>.dat)
  problems/                 # OLD-format PDDL problem files (reference only, not loaded at runtime)
  average_human_results_arrays/  # Human judgment CSVs (<exp_id>.csv)
```

---

## Running Experiments

```bash
# Run all 16 experiments
julia --project=. src_new/main_with_hierarchical.jl

# Run specific experiments
julia --project=. src_new/main_with_hierarchical.jl 3_3 3_4 4_3 4_4

# 5 seeded runs (SIPS + Hierarchical)
julia --project=. src_new/main_multi_run.jl

# Regenerate storyboards only (reads existing CSVs, no inference)
julia --project=. src_new/make_storyboards.jl
```

The script outputs per experiment:
- Trajectory GIF → `example/doors-keys-gems/solutions/trajectories/plan_trajectory_<exp_id>.gif`
- SIPS CSV → `example/doors-keys-gems/goal_probs_SIPS/goal_probs_SIPS_<exp_id>.csv`
- Hierarchical CSV → `example/doors-keys-gems/goal_probs_hierarchical/goal_probs_hierarchical_<exp_id>.csv`
- Three storyboards (human / SIPS / hierarchical) → `example/doors-keys-gems/solutions/`

---

## All 16 Experiments

Plan files are named `<exp_id>_problem_<prob_id>_goal<k>_<participant>.dat`.
`goal<k>` is 0-indexed: goal0=gem1, goal1=gem2, goal2=gem3.

Set labels (from paper): Set 1 = optimal path, Set 2 = detour, Set 3 = backtracking, Set 4 = irreversible failure.

| Exp ID | Problem | True Goal | Notes |
|--------|---------|-----------|-------|
| 1_1    | 6       | gem2      | |
| 1_2    | 7       | gem1      | |
| 1_3    | 4       | gem1      | |
| 1_4    | 12      | gem3      | |
| 2_1    | 9       | gem3      | |
| 2_2    | 5       | gem1      | |
| 2_3    | 10      | gem3      | |
| 2_4    | 8       | gem2      | |
| 3_1    | 6       | gem1      | |
| 3_2    | 4       | gem2      | |
| 3_3    | 11      | gem2      | plan file converted from old format |
| 3_4    | 8       | gem3      | plan file converted from old format |
| 4_1    | 7       | gem2      | |
| 4_2    | 5       | gem3      | |
| 4_3    | 10      | gem1      | plan file converted from old format |
| 4_4    | 12      | gem3      | plan file converted from old format |

---

## Key Source Files

### `src_new/AbstractPlanner.jl`

- Defines `AbstractPlanner <: SymbolicPlanners.Planner` with the same fields as `ForwardPlanner` so SIPS can directly manipulate budget variables.
- `solve(planner, domain, state, spec)`: runs a probabilistic A* over abstract subgoal nodes. Each abstract node expansion calls `PhysicalPlanner.solve` to find neighboring abstract states and their physical path costs.
- Uses `prob_peek` / `prob_dequeue!` from SymbolicPlanners for stochastic node selection (search_noise = temperature ρ).
- At line 23: `include("../example/doors-keys-gems/utils.jl")` — path is relative to `src_new/`.

### `src_new/PhysicalPlanner.jl`

- Module `PhysicalPlanner` (nested inside `AbstractPlanners`).
- `abstract_actions(domain, state)`: returns ALL grounded `:pickup` and `:unlock` actions as `ActionGoal` subgoals — spec-independent (no filtering by goal gem).
- `solve(domain, state)`: Dijkstra over physical grid, stopping expansion when an abstract subgoal (pickup or unlock) is reached. Returns `MultiplePathsSearchSolution` with all reachable abstract subgoal states and their path costs.
- `PHYSICAL_CACHE`: module-level `Dict{UInt, MultiplePathsSearchSolution}` keyed by `hash(state)`. Avoids re-running Dijkstra for shared abstract states across particles.
- `clear_physical_cache!()`: called at the start of each experiment loop to prevent cross-experiment cache poisoning.

### `src_new/main_with_hierarchical.jl`

Key SIPS parameters:
```julia
# Both SIPS and Hierarchical use:
search_noise   = 0.1
prob_replan    = 0.1
budget_dist    = shifted_neg_binom
budget_dist_args = (2, 0.05, 1)   # SIPS
budget_dist_args = (2, 0.1, 1)    # Hierarchical
n_samples      = 120

# Rejuvenation kernels:
# SIPS:         ReplanKernel(2)
# Hierarchical: SequentialKernel(InitGoalKernel(), ReplanKernel(2))
```

`InitGoalKernel()` can resimulate the goal hypothesis; `ReplanKernel(n)` re-proposes the last n plan choices. Together they allow particles to escape incorrect goal hypotheses.

---

## PDDL Format Notes

### Coordinate System (new format)

- y=1 is the **top** row, y increases **downward** (opposite of old format).
- x is unchanged.
- Old → new conversion: `new_y = 9 - old_y` (for 8×8 grid, height=8).

### Wall Encoding

```pddl
(= (walls) (transpose (bit-mat
   (bit-vec b1 b2 b3 b4 b5 b6 b7 b8)  ; y=1, bits = x=1..8
   ...                                  ; y=2..8
)))
```
Each row = one y-value; each bit = one x-value. After `transpose`, `walls[x][y]` gives wall status.

### Unlock Action Syntax

- Old: `(unlock key2 right)` — direction-based
- New: `(unlock key2 door1)` — named door object

Conversion: trace agent position at unlock time, identify which door is adjacent in the given direction, map to that door's name in the problem file.

### Problem Files Status

All 12 problem files exist in `example/doors-keys-gems/problems_modified/`. Problems 1–9 existed previously; problems 10, 11, 12 were converted/created from old-format files in `domains/doors-keys-gems/problems/`.

---

## Plan File Conversion (experiments 3_3, 3_4, 4_3, 4_4)

These four plan files were in old format and converted in this session:

| Plan file | Conversion |
|-----------|-----------|
| `3_3_problem_11_goal1_0.dat` | `(unlock key2 right)` → `door1`, `(unlock key1 down)` → `door2` |
| `3_4_problem_8_goal2_0.dat`  | `(unlock key1 up)` → `door1`, `(unlock key2 right)` → `door2`, `(unlock key3 up)` → `door3` |
| `4_3_problem_10_goal0_0.dat` | `(unlock key1 left)` → `door2` |
| `4_4_problem_12_goal2_1.dat` | `(unlock key1 up)` → `door2` |

Problem-11.pddl was also created from scratch by converting `domains/doors-keys-gems/problems/problem-11.pddl`.

---

## Known Behaviors / Gotchas

- **3_2 gem2 never recovers**: In problem 4, the paths to gem2 and gem3 share the same corridor until 1 step before divergence. With `search_noise=0.1`, the path cost difference compounds quickly, collapsing gem2 probability early. `InitGoalKernel` cannot rescue it because the full trajectory likelihood still favors gem3. * This issue was resolved actually, when replan probability was reduced from 0.5 to 0.1. 
- **`PHYSICAL_CACHE` must be cleared between experiments**: different problems have different domain states; hash collisions across problems would give wrong cached solutions.
- **Julia 1.6 only**: do not upgrade; newer Julia breaks PDDL/GLMakie dependencies.
- **`SequentialKernel` spelling**: it's `SequentialKernel(InitGoalKernel(), ReplanKernel(2))` — not `SequentialKernal` or `InitalReplanKernel`.
- **`include` path in AbstractPlanner.jl**: uses `../example/` (singular), not `../examples/`.
- **plan files are partial trajectories**: human participants did not always complete the task; SIPS infers goal probabilities from each step regardless.
