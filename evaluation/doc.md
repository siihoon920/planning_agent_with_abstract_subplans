# Evaluation scripts

This folder contains scripts to evaluate the **SIPS** model and the **Abstract (Hierarchical)** model against human goal-inference data on the `doors-keys-gems` domain.

## What the CSV columns mean

Every model output CSV (`goal_probs_SIPS_X_Y.csv` and `goal_probs_hierarchical_X_Y.csv`) has three columns:

| Column | Meaning |
|--------|---------|
| `(has gem1)` | Inferred probability that the agent's goal is to collect gem 1 |
| `(has gem2)` | Inferred probability that the agent's goal is to collect gem 2 |
| `(has gem3)` | Inferred probability that the agent's goal is to collect gem 3 |

Each row is one timestep. The three values sum to 1 at every row — they form a probability distribution over the three possible goals. Row `t` (1-indexed, after the header) holds the model's prediction after observing the agent take `t` actions.

The human CSVs (`domains/doors-keys-gems/average_human_results_arrays/X_Y.csv`) contain the same three goal probabilities per survey point, stored as a flat single column in column-major (Julia-default) order:

```
goal1_t1
goal2_t1
goal3_t1
goal1_t2
goal2_t2
goal3_t2
...
```

Human data loading follows `main_with_hierarchical.jl` exactly:
```julia
human_data_1d    = vec(readdlm(path, ',', Float64))
n_time_steps     = div(length(human_data_1d), n_goals)
human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)
```

## Which stimuli.json to use

There are two `stimuli.json` files in the repository:

| File | Experiments where `length(times) == n_human_timesteps` |
|------|---------------------------------------------------------|
| `domains/doors-keys-gems/stimuli.json` (root) | 4 / 16 |
| `domains/doors-keys-gems/stimuli/stimuli.json` (subfolder) | 9 / 16 |

**The scripts use `domains/doors-keys-gems/stimuli/stimuli.json`** (the subfolder version). It matches significantly more experiments and appears to be the more recent, refined version.

The 7 remaining mismatches (where `times` length ≠ human data points) are off by only 1–2 entries, likely because a few extra observation points were added to the final human study after the stimuli.json was last updated. For those experiments the scripts emit a warning and use `min(len(times), n_human_timesteps)` data points.

## PCC computation

The `times` field in `stimuli.json` records the specific action-step indices (1-indexed) at which human participants were shown the agent's current state and asked for their goal judgment. The evaluation uses these exact timesteps, rather than evenly-spaced intervals, to sample model predictions.

For each experiment:

1. Load the model matrix `(n_goals × T_model)` — one column per action step.
2. Load the human matrix `(n_goals × n_human_ts)` — one column per survey point.
3. Extract model columns at `times[1], times[2], ...` to obtain `(n_goals × n)`.
4. Compare with human columns `1, 2, ..., n` to obtain `(n_goals × n)`.
5. Flatten both matrices to 1-D (all 3 gems × n timesteps concatenated).
6. Compute `pearsonr` on the two flat vectors.

All 3 gem probabilities are included in every PCC score — each (gem, timestep) pair contributes one (model_value, human_value) point.

## Experiment naming

Files are named `X_Y` where `X` is the problem number and `Y` is the set number.

| Problem | Category |
|---------|----------|
| 1 | Optimal |
| 2 | (neither) |
| 3 | Sub-optimal |
| 4 | Sub-optimal |

There are 4 sets per problem, giving 16 experiment sets in total.

## Running the scripts

All scripts must be run from the **repository root**.

### Compute PCC

```bash
julia --project=. evaluation/compute_pcc.jl
```

Prints a per-set table and three aggregate rows (overall mean, optimal, sub-optimal), then writes `evaluation/pcc_results.csv`.

### Scatter plot

```bash
julia --project=. evaluation/scatter_plot.jl
```

Produces `evaluation/scatter_plot.png` — SIPS predictions in blue, Abstract model in red, human values on y-axis. Each point is one (gem, timestep) pair from one experiment, sampled at the `stimuli.json` timesteps.

**Note on display:** `scatter_plot.jl` uses `GLMakie`, which requires a graphical display. If you are running on a headless server, replace `using GLMakie` at the top of the file with `using CairoMakie` (install once with `] add CairoMakie`).

## Output files

| File | Description |
|------|-------------|
| `pcc_results.csv` | Per-set PCC scores + aggregate summary (overall / optimal / sub-optimal) |
| `scatter_plot.png` | Scatter plot of model vs human goal probabilities |
