# GEMINI.md - Plinf.jl Project Overview

## Project Overview
**Plinf.jl** is a framework for agent-environment modeling and inverse planning (goal/state inference) in Julia. It integrates PDDL for domain representation, SymbolicPlanners.jl for planning, and Gen.jl for probabilistic inference.

## Core Architecture

### 1. Modeling (`src/modeling/`)
Defines the generative process for agent behavior and environment dynamics.
- **Agents (`src/modeling/agents/`)**: Components for agent beliefs, goals, and plans.
- **Worlds (`src/modeling/worlds.jl`)**: The primary Gen generative model (`world_model`) that simulates agent-environment interactions.
- **Observations (`src/modeling/observations.jl`)**: Logic for mapping state-action traces to observations.

### 2. Inference (`src/inference/`)
Implements algorithms for inverse planning (inferring agent states/goals from observations).
- **SIPS (`src/inference/inference.jl`)**: Sequential Inverse Plan Search, a particle filtering algorithm for online goal and plan inference.
- **Rejuvenation (`src/inference/rejuvenate.jl`)**: MCMC kernels to improve particle diversity in SIPS.

### 3. Hierarchical Planning (`src_new/`)
A specialized implementation for hierarchical reasoning, separating high-level strategic planning from low-level physical execution.

- **Physical Planner (`src_new/PhysicalPlanner.jl`)**:
    - Handles low-level navigation and atomic actions (e.g., `up`, `down`, `left`, `right`).
    - Uses Dijkstra-style search to identify all reachable "abstract subgoals" (e.g., picking up a key or reaching a door) from a given state.
    - Returns a `MultiplePathsSearchSolution` containing multiple physical plans and trajectories to these subgoals.

- **Abstract Planner (`src_new/AbstractPlanner.jl`)**:
    - Performs high-level planning over abstract states.
    - **Custom Search Nodes**: Uses `AbstractPathNode` and `MultipleLinkedNodesRef` to maintain the hierarchy. `MultipleLinkedNodesRef` acts as a bridge, storing the low-level physical plan and trajectory that connects one abstract state to another.
    - **State Expansion**: During `expand!`, it queries the `PhysicalPlanner` to discover candidate next abstract states and their associated costs.
    - **Hierarchical Reconstruction**: Features a specialized `reconstruct` function that traverses the chain of `MultipleLinkedNodesRef` to flatten them into a single, continuous physical plan and trajectory for the agent to follow.

- **Example (`src_new/abstract_example.jl`)**: 
    - Demonstrates the hierarchical planner on the `doors-keys-gems` domain, successfully finding multi-step plans (e.g., `pickup(key) -> unlock(door) -> pickup(gem)`) by bridging abstract intentions with physical execution.

### 4. Examples (`examples/`)
Reference implementations for standard domains:
- `block-words`: Blocks world reasoning.
- `doors-keys-gems`: Gridworld with navigation and object interaction.
- `gridworld`: Simple 2D navigation.

## Engineering Standards & Conventions

### Technology Stack
- **Julia 1.6**: **Strict requirement.** The codebase and its dependencies are optimized for this version.
- **Gen.jl**: Used for probabilistic modeling. Expect heavy use of `@gen`, `@trace`, and `@choicemap` macros.
- **PDDL.jl**: Used for state and action semantics. Domains are defined in `.pddl` files.
- **SymbolicPlanners.jl**: Core planning engine.

### Patterns
- **Generative Modeling**: The system is built around a "world model" that represents a prior over agent behavior.
- **Particle Filtering**: Inference is primarily online and uses Sequential Monte Carlo (SMC).
- **Modular Design**: Clear separation between the model (how agents act) and the inference (how we guess what they are doing).

## Development Lifecycle

### Setup
Ensure you are using Julia 1.6.
```bash
juliaup add 1.6
juliaup default 1.6
```
Inside the Julia REPL:
```julia
] activate .
] instantiate
```

### Running Tests
Tests are located in `test/runtests.jl`. Run them via the package manager:
```julia
] test
```

### Working with `src_new`
The files in `src_new/` are part of an ongoing transition towards abstract subplans. When modifying planning logic, consider both the legacy `src/modeling/agents/plans.jl` and the newer `src_new/` implementations.
