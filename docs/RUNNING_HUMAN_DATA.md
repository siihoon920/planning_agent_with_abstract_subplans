# Running Human Data Examples

This document explains how to properly run and generate goal-inference visualisations for the human data provided in the repository for the `"Modeling the Mistakes of Boundedly Rational Agents Within a Bayesian Theory of Mind"` paper experiments.

## Overview

The codebase includes two primary experimental domains where the Bayesian goal inferences of human observers were recorded over time: **Block-Words** and **Doors-Keys-Gems**. The human data is provided as 1-dimensional float arrays in `.csv` files stored under the `average_human_results_arrays/` folder in each domain.

We have modified the `examples/block-words/human_example.jl` and `examples/doors-keys-gems/human_example.jl` scripts to accurately reflect this data by dynamically loading:
- The exact `.pddl` problem setup the humans were observing
- The exact action plan (trajectory) the boundedly rational agent took
- The dynamically generated goal spaces humans had to choose from
- The recorded goal-inference probabilities humans provided over time

## How the Code Works

The scripts have been modularized to accept an **experiment ID** (e.g., `1_1`, `2_3`, `4_4`) via the command line arguments. Based on this ID, each script correlates and parses the necessary background files from the `domains/` directory.

### Block-Words Domain

**Located in:** `examples/block-words/human_example.jl`

When provided an ID like `1_1`, the code securely maps to the following files:
1. **Problem:** It loads `domains/block-words/experiment-1-1.pddl`.
2. **Action Plan & Goals:** It leverages `domains/block-words/experiment-scenarios.jl` to fetch the specific deterministic sequence of moves via `get_action("1-1")` and the possible pool of word goals via `get_goal_space("1-1")`.
3. **Human Data:** It loads `domains/block-words/average_human_results_arrays/1_1.csv` and reshapes the 1D array into a 2D matrix matching the length of the goals and the length of the trajectory. 

### Doors-Keys-Gems Domain

**Located in:** `examples/doors-keys-gems/human_example.jl`

When provided an ID like `1_1`, the dataset mapping is slightly different as the PDDL problems aren't 1-to-1 named by the experiment.
1. **Action Plan & Environment Matching:** The script navigates to `domains/doors-keys-gems/plans/` and searches for a `.dat` file with the matching prefix (e.g., `1_1_problem_6_goal1_0.dat`). This single file yields: 
   - The sequence of actions the agent played out (parsed line-by-line).
   - The embedded problem ID (e.g., `problem_6` indicates it should load `problem-6.pddl`).
2. **Problem Definition:** It then loads `domains/doors-keys-gems/problem-6.pddl`.
3. **Human Data:** It loads `domains/doors-keys-gems/average_human_results_arrays/1_1.csv` and similarly manipulates the array to match the standard 3 gems present in the domain as candidate goals.

## How to Run the Scripts

The scripts run entirely headlessly and save the outputs to the directory you ran them from.

### 1. Activating the Environment
Before running these scripts, ensure the standard dependencies are active.
```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

### 2. Running Block-Words
Execute the Julia file and pass the experiment ID you wish to simulate.
```bash
julia examples/block-words/human_example.jl 1_1
```
*Outputs generated:*
- `examples/block-words/human_trajectory_1_1.gif` (Shows a top-down visual replay of the blocks moving logic)
- `examples/block-words/human_goal_inference_storyboard_1_1.png` (Creates a visual storyboard charting human probabilities mapping over the individual action frames)

### 3. Running Doors-Keys-Gems 
Execute the Julia file and pass the experiment ID.
```bash
julia examples/doors-keys-gems/human_example.jl 2_3
```
*Outputs generated:*
- `examples/doors-keys-gems/human_trajectory_2_3.gif` (Shows the gridworld playback of the agent retrieving items)
- `examples/doors-keys-gems/human_goal_inference_storyboard_2_3.png` (Creates the graphical probability trajectory synced with the gridworld states)

If you don't supply an argument, both scripts default to running `1_1` as a baseline.
