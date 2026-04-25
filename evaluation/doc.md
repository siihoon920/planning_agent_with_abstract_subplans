# Evaluation scripts

This folder evaluates the **SIPS** model and the **Abstract (Hierarchical)** model against human goal-inference data on the `doors-keys-gems` domain.

---

## Data structure

### Model output CSVs

Files in `example/doors-keys-gems/goal_probs_SIPS/` and `example/doors-keys-gems/goal_probs_hierarchical/` have the same layout:

```
(has gem1),(has gem2),(has gem3)   ← header row
0.333, 0.333, 0.333                ← prediction after action step 1
0.333, 0.333, 0.333                ← prediction after action step 2
0.441, 0.346, 0.212                ← prediction after action step 3
...
```

Row `t` (1-indexed, after the header) = the model's goal-probability distribution after observing the agent take `t` actions. The three values sum to 1 — they are probabilities over the three possible goals (collect gem1, gem2, or gem3).

### Human data CSVs

Files in `domains/doors-keys-gems/average_human_results_arrays/` store averaged human judgments. **There is no header.** The file is a flat single column stored in **column-major order** (matching Julia's `reshape` default):

```
gem1_at_times[1]
gem2_at_times[1]
gem3_at_times[1]
gem1_at_times[2]
gem2_at_times[2]
gem3_at_times[2]
...
```

Each group of 3 consecutive rows corresponds to one survey point — the three gem probabilities assigned by humans at that moment. The survey points are the action-step indices listed in the `times` field of `stimuli/stimuli.json` for that experiment.

Human data is loaded identically to `main_with_hierarchical.jl`:
```julia
human_data_1d    = vec(readdlm(path, ',', Float64))
n_time_steps     = div(length(human_data_1d), n_goals)
human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)
```

---

## Which stimuli.json to use

There are two stimuli files:

| File | Match rate (`len(times)*3 == CSV rows`) |
|------|-----------------------------------------|
| `domains/doors-keys-gems/stimuli.json` (root) | 4 / 16 |
| `domains/doors-keys-gems/stimuli/stimuli.json` (subfolder) | 9 / 16 |

**All scripts use `domains/doors-keys-gems/stimuli/stimuli.json`** (the subfolder). It matches significantly more experiments and is the refined version used in the final study.

### Mismatch handling

For the 7 experiments where `len(times)*3 < actual CSV rows`, the human CSV contains **extra survey points appended at the end** — additional observation moments that were collected but are not documented in `stimuli.json`.

**Rule:** always use the first `len(times) * 3` rows of the human CSV and discard the rest. This guarantees that human column `i` always corresponds to `times[i]` from `stimuli.json`.

For any experiment where the CSV has *fewer* rows than `len(times)*3` (not currently observed, but handled defensively), the code warns and uses only the available data.

---

## PCC computation (`compute_pcc.jl`)

For each of the 16 experiments:

1. Read `times` from `stimuli/stimuli.json` — a list of action-step indices (1-based) at which human judgments were collected.

2. **Load human data** — take the first `len(times) * 3` rows of the CSV, reshape to `(3, len(times))`. Column `i` = human judgment vector at `times[i]`.

3. **Load model data** — parse the model CSV (skip header), transpose to `(3, T_model)`. Column `t` = model prediction at action step `t`.

4. **Extract model predictions at survey times** — for each `times[i]`, take `model[:, times[i]]`. Skip any `times[i]` that exceed the model's output length (warn if any are dropped).

5. **Flatten and correlate** — flatten both `(3, n_valid)` matrices to 1-D vectors of length `n_valid * 3`, then compute `pearsonr`. This means each (gem, survey-point) pair contributes one data point to the correlation.

### Aggregate scores

- **Overall mean** — average PCC across all 16 experiments.
- **Optimal** — average PCC for problem 1 (4 sets).
- **Sub-optimal** — average PCC for problems 3 and 4 (8 sets).

---

## Scatter plot (`scatter_plot.jl`)

Each dot represents one **gem probability** at one **survey timestep** in one **experiment**:

- **x-axis** — model predicted probability (SIPS in blue, Abstract in red).
- **y-axis** — human inferred probability.

Only the timesteps listed in `stimuli.json` are used (extra human rows are discarded, as above).

**Total dots per model** = Σ `len(times_i) * 3` over all 16 experiments.

From `stimuli/stimuli.json` the times lengths are:

| Experiment | len(times) | Dots |
|------------|-----------|------|
| 1_1 | 4 | 12 |
| 1_2 | 4 | 12 |
| 1_3 | 4 | 12 |
| 1_4 | 5 | 15 |
| 2_1 | 4 | 12 |
| 2_2 | 3 | 9  |
| 2_3 | 4 | 12 |
| 2_4 | 5 | 15 |
| 3_1 | 4 | 12 |
| 3_2 | 4 | 12 |
| 3_3 | 4 | 12 |
| 3_4 | 4 | 12 |
| 4_1 | 4 | 12 |
| 4_2 | 3 | 9  |
| 4_3 | 3 | 9  |
| 4_4 | 4 | 12 |
| **Total** | **63** | **189** |

---

## Running the scripts

All scripts must be run from the **repository root**.

### Compute PCC

```bash
julia --project=. evaluation/compute_pcc.jl
```

Prints a per-experiment table and three aggregate rows, then writes `evaluation/pcc_results.csv`.

### Scatter plot

```bash
julia --project=. evaluation/scatter_plot.jl
```

Uses `PyPlot` (matplotlib via PyCall — no GPU required). Saves `evaluation/scatter_plot.png`.

### Check stimuli.json alignment

```bash
python evaluation/check_stimuli.py
```

Prints a table showing for each experiment whether `len(times)*3 == CSV rows` for both stimuli files.

---

## Output files

| File | Description |
|------|-------------|
| `pcc_results.csv` | Per-experiment PCC + aggregate summary |
| `scatter_plot.png` | Scatter plot of model vs human goal probabilities |
