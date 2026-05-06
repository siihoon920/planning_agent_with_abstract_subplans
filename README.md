# planning_agent_with_abstract_subplans

Extends [Plinf.jl](https://github.com/ztangent/Plinf.jl) with a **hierarchical (abstract) planner** for Bayesian goal inference in the Doors-Keys-Gems gridworld. The agent reasons over abstract subgoal sequences (key pickups, door unlocks) rather than individual physical steps, and a Sequential Monte Carlo particle filter infers a human observer's goal by weighting trajectories against this hierarchical planning model.

The main comparison is:
- **SIPS** (baseline): flat `ProbAStarPlanner` from Plinf
- **Hierarchical SIPS**: two-level `AbstractPlanner` + `PhysicalPlanner`

---

## Environment Setup

**Julia 1.6 only.** Newer versions break PDDL/SymbolicPlanners/GLMakie dependencies.

```bash
# Install juliaup and pin to 1.6
brew install juliaup
juliaup add 1.6
juliaup default 1.6
```

Clone and activate the environment:

```bash
git clone git@github.com:siihoon920/planning_agent_with_abstract_subplans.git
cd planning_agent_with_abstract_subplans
```

Inside the Julia REPL (`julia`), enter package mode with `]`:

```
activate .
instantiate
```

---

## Hierarchical Planner

The hierarchical planner is a two-level search defined in `src_new/`.

### AbstractPlanner (`src_new/AbstractPlanner.jl`)

A proper subtype of `SymbolicPlanners.Planner` with the same fields as `ForwardPlanner`, so SIPS can manipulate budget variables directly. Its `solve()` runs a **probabilistic A\*** over abstract subgoal nodes:

- At each abstract node, it calls `PhysicalPlanner.solve` to find which abstract subgoals (key pickups, door unlocks) are reachable from the current physical state and at what cost.
- It selects the next abstract subgoal stochastically using `prob_dequeue!` with temperature `search_noise` — the probability of choosing a suboptimal next subgoal decays as `exp(-cost_diff / search_noise)`.
- The returned plan is a full physical action sequence spliced together from the physical sub-plans between each selected abstract subgoal.

### PhysicalPlanner (`src_new/PhysicalPlanner.jl`)

Runs a **Dijkstra search** over the physical grid state. Expansion stops when a `:pickup` or `:unlock` action is reached (these are the abstract subgoal boundaries). The search is spec-independent — it considers *all* pickup and unlock actions regardless of which gem is the true goal.

Results are cached in `PHYSICAL_CACHE` keyed by `hash(state)`, avoiding repeated Dijkstra runs when multiple particles share the same physical state. The cache must be cleared between experiments with `PhysicalPlanner.clear_physical_cache!()`.

### SIPS Integration (`src_new/main_with_hierarchical.jl`)

The hierarchical planner plugs into SIPS as a drop-in replacement for the flat planner:

```julia
abs_planner = AbstractPlanners.AbstractPlanner(
    RelaxedMazeDist();
    search_noise=0.3, save_search=true
)

abs_agent_config = AgentConfig(
    domain, abs_planner;
    goal_config      = StaticGoalConfig(goal_prior),
    replan_args      = (prob_replan=0.1, budget_dist=shifted_neg_binom, budget_dist_args=(2, 0.2, 1)),
    act_epsilon      = 0.05
)

abs_sips_runner = SIPS(
    abs_world_config,
    resample_cond = :ess,
    rejuv_cond    = :periodic,
    rejuv_kernel  = SequentialKernel(InitGoalKernel(), ReplanKernel(2)),
    period        = 2
)
```

`InitGoalKernel` can resimulate the goal hypothesis from scratch; `ReplanKernel(2)` re-proposes the last 2 planning choices. Together they allow particles to recover from incorrect goal hypotheses during inference.

---

## Running Experiments

```bash
# All 16 experiments
julia --project=. src_new/main_with_hierarchical.jl

# Specific experiments
julia --project=. src_new/main_with_hierarchical.jl 2_1 3_3 4_4

# 5 independent runs with fixed seeds (for multi-run averaging)
julia --project=. src_new/main_multi_run.jl

# SIPS only, with extended budget (budget_p = 0.01)
julia --project=. src_new/main_sips_budget001.jl
```

---

## Output Data

All inference outputs live under `example/doors-keys-gems/`.

### Goal probability CSVs

| Directory | Contents |
|---|---|
| `goal_probs_SIPS/` | Single-run SIPS results: `goal_probs_SIPS_<exp_id>.csv` |
| `goal_probs_SIPS/goal_probs_SIPS_<exp_id>_run_N.csv` | Per-run SIPS results (runs 1–5) |
| `goal_probs_hierarchical/` | Single-run hierarchical results: `goal_probs_hierarchical_<exp_id>.csv` |
| `goal_probs_hierarchical/goal_probs_hierarchical_<exp_id>_run_N.csv` | Per-run hierarchical results |
| `goal_probs_SIPS_budget001/` | SIPS with budget_p=0.01: `goal_probs_SIPS_budget001_<exp_id>.csv` |

All CSVs share the same format:
```
(has gem1),(has gem2),(has gem3)     ← header
0.333,0.333,0.333                    ← probabilities after action step 1
0.441,0.346,0.212                    ← probabilities after action step 2
...
```
Row `t` (after header) = model's goal distribution after `t` actions.

### Storyboards and animations

| Directory | Contents |
|---|---|
| `solutions/trajectories/` | GIF animation of the human trajectory per experiment |
| `solutions/storyboard_human/` | Human judgment probability plots |
| `solutions/storyboard_SIPS/` | SIPS model probability plots |
| `solutions/storyboard_hierarchical/` | Hierarchical model probability plots |

Storyboards can be regenerated from existing CSVs (without re-running inference):
```bash
julia --project=. src_new/make_storyboards.jl [exp_id ...]
```

---

## Evaluation

All evaluation scripts run from the **repository root**. Human judgment data is in `domains/doors-keys-gems/average_human_results_arrays/<exp_id>.csv`.

### `compute_pcc.jl` — Per-experiment Pearson correlation

Computes the Pearson Correlation Coefficient (PCC) between each model's predicted goal probabilities and the averaged human judgments, evaluated at the human survey timesteps (judgement points).

For each experiment: model predictions are extracted at the judgement-point action steps, flattened alongside the human judgment vectors, and correlated. Results are broken down by problem set:

- **Optimal (set 1)**: human follows the shortest path to the goal
- **Action mistakes (set 2)**: human takes a suboptimal action (e.g., picks up an unnecessary key)
- **Plan mistakes (set 3)**: human pursues a suboptimal plan
- **Short-sighted plan (set 4)**: human initially follows a locally sensible but globally suboptimal route

```bash
julia --project=. evaluation/compute_pcc.jl
# → evaluation/pcc_results.csv
```

### `compute_pcc_multi_run.jl` — Multi-run averaged PCC

Same as `compute_pcc.jl` but averages goal probabilities across the 5 independent runs before computing PCC, reducing run-to-run variance.

```bash
julia --project=. evaluation/compute_pcc_multi_run.jl
# → evaluation/pcc_results_multi_run.csv
```

### `compute_pcc_budget001.jl` — SIPS with extended budget

Same as `compute_pcc.jl` but reads from `goal_probs_SIPS_budget001/`. Compares SIPS with `budget_p=0.01` (mean budget ~198 steps) against the hierarchical model.

```bash
julia --project=. evaluation/compute_pcc_budget001.jl
# → evaluation/pcc_results_budget001.csv
```

### `compute_pcc_by_timeframe.jl` — PCC broken down by early / mid / late

Splits the judgement points into thirds and reports PCC separately for each third, revealing where in the trajectory each model gains or loses accuracy.

```bash
julia --project=. evaluation/compute_pcc_by_timeframe.jl
```

### `scatter_plot.jl` — Pooled model vs human scatter

Pools all `(model_prob, human_prob)` pairs across all experiments, goals, and judgement points into a single scatter plot. Reports one aggregate Pearson r per model. Note: this pooled r is inflated relative to the per-experiment averages in `compute_pcc.jl` because it includes between-experiment variance.

```bash
julia --project=. evaluation/scatter_plot.jl
# → evaluation/scatter_plot.png
```

### `analyze_abstract_structure.jl` — Goal ambiguity vs PCC gap

For experiments in problem sets 1 and 2, measures goal ambiguity in the human judgment data (mean Shannon entropy across judgement points) and correlates it with the Abstract − SIPS PCC gap. Tests whether sustained goal uncertainty predicts where the hierarchical model outperforms SIPS.

```bash
julia --project=. evaluation/analyze_abstract_structure.jl
# → evaluation/abstract_structure_analysis.csv
# → evaluation/entropy_vs_pcc.png
```

---

## Experiment Index

Plan files are named `<exp_id>_problem_<prob_id>_goal<k>_<participant>.dat`. `goal<k>` is 0-indexed: goal0=gem1, goal1=gem2, goal2=gem3.

| Exp ID | Problem | True Goal |
|--------|---------|-----------|
| 1_1 | 6 | gem2 |
| 1_2 | 7 | gem1 |
| 1_3 | 4 | gem1 |
| 1_4 | 12 | gem3 |
| 2_1 | 9 | gem3 |
| 2_2 | 5 | gem1 |
| 2_3 | 10 | gem3 |
| 2_4 | 8 | gem2 |
| 3_1 | 6 | gem1 |
| 3_2 | 4 | gem2 |
| 3_3 | 11 | gem2 |
| 3_4 | 8 | gem3 |
| 4_1 | 7 | gem2 |
| 4_2 | 5 | gem3 |
| 4_3 | 10 | gem1 |
| 4_4 | 12 | gem3 |
