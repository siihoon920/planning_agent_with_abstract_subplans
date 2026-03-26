# Next Steps: What To Run And In What Order

This file gives you a practical sequence to run the repository after setup.

You already completed package setup correctly (`activate` + `instantiate`), so the next step is running examples and inspecting outputs.

---

## 1) Confirm Julia version and environment

From the repo root:

```bash
julia --version
```

Target version is Julia `1.6.x` (per `Project.toml` compat).

If needed:

```bash
juliaup add 1.6
juliaup default 1.6
```

---

## 2) Quick smoke test: load package only

```bash
julia --project=. -e 'using Plinf; println("Plinf loads successfully")'
```

If this fails, it is usually one of:

- wrong Julia version
- not running from project root
- incomplete package instantiate

---

## 3) Run the examples (recommended order)

Run from repo root:

### A) Gridworld (simplest conceptual start)

```bash
julia --project=. examples/gridworld/example.jl
```

What you should see:

- printed goal probabilities over time
- rendered/animated inference windows
- trajectory/planning behavior in a basic domain

### B) Doors-Keys-Gems (closest to paper-style compositional behavior)

```bash
julia --project=. examples/doors-keys-gems/example.jl
```

This script also saves files directly in `examples/doors-keys-gems/`:

- `plan.gif`
- `trajectory.gif`
- `inference.gif`
- `goal_inference_storyboard.png`

### C) Block-Words (symbolic compositional goal inference)

```bash
julia --project=. examples/block-words/example.jl
```

---

## 4) If rendering fails (GLMakie/display issues)

On some Linux/headless setups, GL rendering may fail.

Options:

- run in a desktop session with display support
- use X forwarding / proper OpenGL setup
- temporarily disable heavy rendering callbacks in example scripts and keep logging-only inference

Minimal non-visual strategy:

- keep `print_goal_probs = true`
- set rendering flags to `false` in combined callbacks (`render=false`, `record=false`, `plot_goal_* = false`)

---

## 5) First code modifications to try

After baseline runs, try controlled edits:

1. **Particle count sensitivity**
   - increase/decrease `n_samples` in each `example.jl`.
2. **Planning noise sensitivity**
   - change `search_noise` in `ProbAStarPlanner(...)`.
3. **Replanning assumptions**
   - adjust `prob_replan` and `budget_dist_args`.
4. **Inference maintenance**
   - compare `resample_cond=:none` vs `:ess`
   - test different rejuvenation kernels (`ReplanKernel(1|2|3)`).

These changes help you understand robustness/performance trade-offs directly.

---

## 6) How to run your own new scenario

Use any example as template:

1. add/choose PDDL domain + problem
2. define candidate goals + `goal_prior`
3. build:
   - `AgentConfig`
   - `ObsNoiseParams`
   - `WorldConfig`
4. create observed trajectory (`obs_traj`) from simulation or real observations
5. convert to choicemaps using `state_choicemap_pairs(...)`
6. run:
   - `sips = SIPS(world_config, ...)`
   - `pf_state = sips(n_particles, t_obs_iter; ...)`
7. inspect posterior over goal address (`goal_addr`) with logs/plots

---

## 7) Troubleshooting checklist

- `using Plinf` fails:
  - ensure `julia --project=.` and Julia 1.6.
- package resolution issues:
  - rerun `Pkg.instantiate()`.
- very slow runtime:
  - reduce `n_samples`, reduce visualization/recording, use smaller problem file.
- strange posterior behavior:
  - verify `goal_addr`, goal support ordering, and observation term selection.

---

## 8) Suggested immediate workflow for you

Given your goal (understand + run):

1. run `gridworld/example.jl` once
2. run `doors-keys-gems/example.jl` and inspect saved GIF/PNG outputs
3. read `CODEBASE_DETAILED_DOCUMENTATION.md` while comparing with one example file line-by-line
4. choose one parameter sweep (e.g., `n_samples`) and rerun same scenario to build intuition

This gets you from "setup complete" to "confidently interpreting results" quickly.
