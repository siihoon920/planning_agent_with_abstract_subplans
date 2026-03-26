# Codebase Documentation: `planning_agent_with_abstract_subplans`

This repository implements a Julia version of the ideas in the paper summary in `summary.md`: online Bayesian goal inference for boundedly rational planners, using a particle-filter style algorithm called Sequential Inverse Plan Search (SIPS).

The code is centered around:

- a **generative world model** in `src/modeling/`
- a **sequential inference engine** in `src/inference/`
- **worked domain examples** in `examples/`

---

## 1) High-level architecture

### Core idea

At each timestep, the model represents:

- an agent's latent internal state (belief, goal, plan),
- the environment state,
- noisy observations of that environment.

Inference maintains many weighted hypotheses ("particles") over these latent states and updates them as observations arrive.

### Main module

- `src/Plinf.jl`
  - defines module `Plinf`
  - loads modeling and inference subsystems
  - imports key dependencies (`PDDL`, `SymbolicPlanners`, `Gen`, `GenParticleFilters`, `PDDLViz`, `Makie`)

---

## 2) `src/modeling/`: generative model

The modeling side is split into agent/environment/observation components and composed into a world model.

### 2.1 World composition (`src/modeling/worlds.jl`)

Important types:

- `WorldState(agent_state, act_state, env_state, obs_state)`
- `WorldConfig(agent_config, env_config, obs_config)`

Important generative functions:

- `world_init(config)`
  - samples initial environment, observation, agent state, and action state
- `world_step(t, world_state, config)`
  - updates agent -> samples action -> transitions environment -> samples observation
- `world_model(n_steps, config)`
  - unfolds `world_step` over time using Gen's unfold combinator

Helpers:

- `get_agent_states`, `get_env_states`, `get_obs_states`, etc., extract trajectories from Gen traces.

### 2.2 Agent model (`src/modeling/agents/`)

`agents.jl` defines:

- `AgentState(belief_state, goal_state, plan_state)`
- `AgentConfig(belief_config, goal_config, plan_config, act_config)`
- `agent_init` and `agent_step`

The constructor `AgentConfig(domain, planner; ...)` auto-selects planning and action models from planner type + keyword args.

#### Goal dynamics (`goals.jl`)

- `StaticGoalConfig(goal_prior, ...)`: sample once, keep fixed.
- `ResamplingGoalConfig(goal_prior, prob_resample, ...)`: occasionally resample goal over time.

#### Plan dynamics (`plans.jl`)

- `PlanState(init_step, sol, spec)` holds current planner output and associated goal spec.
- `DetermReplanConfig`: deterministic replanning only when needed.
- `ReplanConfig`: stochastic replanning with probability + random planning budget.
- `ReplanPolicyConfig`: for policy-style planners, can replan from scratch or refine policy.

This is where bounded rationality enters strongly: random finite planning budgets (default via shifted negative binomial) and stochastic replanning.

#### Action model (`actions.jl`)

- `DetermActConfig`: deterministic action from plan/policy.
- `EpsilonGreedyActConfig`: with probability epsilon, pick a random/backup action.
- `BoltzmannActConfig`: soft action sampling from values.
- `CommunicativeActConfig`: optional joint action+utterance model.

### 2.3 Environment model (`environments.jl`)

- `StaticEnvConfig`: no dynamics.
- `PDDLEnvConfig(domain, init)`: deterministic PDDL transitions.

`pddl_env_step` safely handles unavailable/no-op actions and otherwise calls `transition`.

### 2.4 Observation model (`observations.jl`)

- `PerfectObsConfig`: noiseless observations.
- `MarkovObsConfig(domain, obs_params)`: observation depends on current state only.

`ObsNoiseParams` supports:

- manually specified term-wise noise models, or
- automatic noise setup for all non-static predicates/functions in a domain.

`ground_obs_params` and `observe_state` handle grounding quantified terms and sampling noisy observations.

---

## 3) `src/inference/`: SIPS particle filtering

### 3.1 Main interface (`src/inference/inference.jl`)

`SequentialInversePlanSearch` / `SIPS` config includes:

- `resample_cond` (`:none`, `:periodic`, `:always`, `:ess`)
- `resample_method` (`:multinomial`, `:residual`, `:stratified`)
- `rejuv_cond` and `rejuv_kernel`
- `ess_threshold`, `period`

Key functions:

- `sips_init(...)`
- `sips_step!(...)`
- `sips_run(...)`

You can call the object directly:

`pf_state = sips(n_particles, t_obs_iter; init_args=..., callback=...)`

### 3.2 Observation/action choicemaps (`choicemaps.jl`)

Utility functions convert trajectories into Gen choicemaps:

- `state_choicemap_pairs(states, obs_terms; ...)`
- `act_choicemap_pairs(actions; ...)`

These are what feed observations into SIPS over time.

### 3.3 Rejuvenation kernels (`rejuvenate.jl`)

MCMC rejuvenation after particle degeneracy:

- `NullKernel`
- `ReplanKernel(n)`: MH moves over recent plan latents
- `InitGoalKernel`, `RecentGoalKernel`, `ConsecutiveGoalKernel`
- compositional kernels: `SequentialKernel`, `MixtureKernel`

### 3.4 Callbacks (`callbacks.jl`)

Flexible instrumentation during inference:

- `PrintStatsCallback`
- `DataLoggerCallback`
- plotting callbacks (`BarPlotCallback`, `SeriesPlotCallback`)
- rendering/recording (`RenderCallback`, `RecordCallback`)
- `CombinedCallback` for composing multiple callbacks.

---

## 4) `examples/`: how the system is used in practice

Each example follows a common pattern:

1. load PDDL `domain.pddl` + one `problems/problem-*.pddl`
2. build initial state, compile domain
3. run a planner once for sanity visualization
4. define candidate goal set + prior
5. build `AgentConfig`, `ObsNoiseParams`, `WorldConfig`
6. produce an observed trajectory (`obs_traj`)
7. convert to timestep/choicemap stream
8. run `SIPS(...)`
9. visualize/save inference outputs

### 4.1 `examples/gridworld/example.jl`

- 2D navigation with candidate end positions as goals.
- Uses `ProbAStarPlanner` + stochastic replanning + epsilon-greedy actions.
- Uses `GridworldCombinedCallback` with optional overlays and recordings.
- Produces trajectory/inference visual assets in-memory (and can be saved manually).

### 4.2 `examples/doors-keys-gems/example.jl`

- compositional puzzle domain with keys, locked doors, and gems.
- Custom heuristic support in `utils.jl`:
  - `GoalManhattan`
  - `RelaxedMazeDist` (door-unlocked relaxed planning estimate)
- Script explicitly saves:
  - `plan.gif`
  - `trajectory.gif`
  - `inference.gif`
  - `goal_inference_storyboard.png`

### 4.3 `examples/block-words/example.jl`

- blocks world variant: infer intended word arrangement.
- Goal hypotheses are words mapped to stack constraints via `word_to_terms`.
- Uses `BlocksworldCombinedCallback` for probability logging and rendering.

---

## 5) Data flow and key addresses in traces

The Gen trace structure is hierarchical. Common addresses:

- initial goal index:
  - `:init => :agent => :goal => :goal`
- per-timestep latent blocks:
  - `:timestep => t => :agent => :belief`
  - `:timestep => t => :agent => :goal`
  - `:timestep => t => :agent => :plan`
  - `:timestep => t => :obs`

This is why examples construct `goal_addr` and `goal_strata` for stratified initialization.

---

## 6) Practical run notes

- `Project.toml` targets Julia `1.6`.
- You already did the right setup:
  - `Pkg.activate(".")`
  - `Pkg.instantiate()`

Potential runtime gotchas:

- GL-based rendering (`GLMakie`) may require a GUI/X11-capable session.
- first run may precompile and take time.
- examples can be compute-heavy depending on particle count and planner settings.

---

## 7) Suggested reading order in code

If you want to deeply understand implementation internals, read in this order:

1. `src/Plinf.jl`
2. `src/modeling/worlds.jl`
3. `src/modeling/agents/agents.jl`
4. `src/modeling/agents/plans.jl`
5. `src/modeling/observations.jl`
6. `src/inference/inference.jl`
7. `src/inference/rejuvenate.jl`
8. one full example script (`examples/doors-keys-gems/example.jl` is the richest)

---

## 8) Relationship to your paper summary

Your `summary.md` and this code align closely:

- bounded rationality through stochastic replanning + finite random budgets
- online inference via sequential particle updates
- rejuvenation via MH kernels to avoid collapse
- compositional symbolic domains via PDDL
- explicit handling of noisy observations

In short, this repository is a practical implementation scaffold of the SIPS-style inverse planning framework described in your summary.
