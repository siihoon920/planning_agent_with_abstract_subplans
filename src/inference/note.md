siihoon920: explain structure of this entire project

GitHub Copilot: Optimizing tool selection...I'll analyze the structure of this Plinf.jl project by examining key files to understand its purpose and organization. 

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/README.md#1-1), lines 1 to 50

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/README.md#51-51), lines 51 to 57

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/Project.toml#1-1), lines 1 to 30

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/src/Plinf.jl)

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/src/modeling/modeling.jl)

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/src/inference/inference.jl#1-1), lines 1 to 50

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/src/modeling/agents/agents.jl#1-1), lines 1 to 30

Read [](file:///Users/sihoonoh/Downloads/julia_test/Plinf.jl/examples/doors-keys-gems/example.jl#1-1), lines 1 to 30

Based on my analysis of the project structure, here's a comprehensive explanation of the Plinf.jl project:

## Project Overview

**Plinf.jl** is a Julia library that implements an architecture for **planning, inverse planning, and inference in planning**. It combines:
- **PDDL** (Planning Domain Definition Language) for representing planning domains
- **Gen** probabilistic programming language for Bayesian inference
- **SymbolicPlanners** for automated planning algorithms

The project focuses on modeling boundedly rational planning agents and performing goal inference from observed behavior.

## Core Architecture

### 1. **Main Module Structure** (src)

- **Plinf.jl** - Main module file that includes all submodules
- **`utils.jl`** - Utility functions used across the library
- **`modeling/`** - Agent and environment modeling components
- **`inference/`** - Bayesian inference algorithms

### 2. **Modeling Components** (modeling)

The modeling system represents planning agents and their environments:

- **`agents/`** - Agent modeling with multiple components:
  - agents.jl - Core agent state and configuration
  - `beliefs.jl` - Agent belief modeling
  - `goals.jl` - Goal representation and priors
  - `plans.jl` - Planning state and algorithms
  - `actions.jl` - Action selection and execution

- **`environments.jl`** - Environment dynamics and state transitions
- **`observations.jl`** - Observation models linking internal states to observable behavior
- **`worlds.jl`** - Complete world models combining agents and environments
- **`utils.jl`** - Modeling-specific utilities

### 3. **Inference Components** (inference)

Implements **Sequential Inverse Plan Search (SIPS)** for goal inference:

- **inference.jl** - Main SIPS particle filtering algorithm
- **`choicemaps.jl`** - Gen choicemap utilities for probabilistic inference
- **`rejuvenate.jl`** - Particle rejuvenation kernels
- **`callbacks.jl`** - Inference callbacks and monitoring
- **`utils.jl`** - Inference-specific utilities

### 4. **Examples** (examples)

Three complete domains demonstrating the library:

- **`gridworld/`** - Basic gridworld navigation
- **`doors-keys-gems/`** - Complex domain with locked doors, keys, and collectible gems
- **`block-words/`** - Classic blocks world planning domain

Each example contains:
- `domain.pddl` - PDDL domain definition
- example.jl - Complete runnable example
- `utils.jl` - Domain-specific utilities
- `problems/` - Multiple problem instances
- `ascii.jl` (some domains) - Text-based visualization

## Key Features

### 1. **Bounded Rationality Modeling**
The library models agents as boundedly rational - they don't always find optimal plans but use resource-bounded search with realistic limitations.

### 2. **Goal Inference**
Can infer agent goals from observed behavior, even when:
- Agents backtrack or make suboptimal decisions
- Plans fail to reach the intended goal
- Only partial trajectories are observed

### 3. **PDDL Integration**
Uses standard PDDL for domain representation, making it compatible with a wide range of planning domains and problems.

### 4. **Particle Filtering**
Implements sophisticated particle filtering with:
- Multiple resampling strategies
- Particle rejuvenation
- Effective sample size monitoring

### 5. **Visualization**
Integrates with PDDLViz and Makie for rich visualizations of:
- Agent trajectories
- Goal inference results
- Planning domains

## Dependencies

Key external libraries:
- **PDDL.jl** - Planning domain representation
- **SymbolicPlanners.jl** - Planning algorithms
- **Gen.jl** - Probabilistic programming
- **GenParticleFilters.jl** - Particle filtering
- **PDDLViz.jl** - Domain visualization
- **GLMakie.jl** - Graphics backend

## Use Cases

This library is designed for research in:
- **Theory of Mind** - Understanding agent intentions from behavior
- **Human-AI Interaction** - Modeling human planning behavior
- **Plan Recognition** - Inferring goals from partial observations
- **Robotics** - Understanding human intentions in shared environments

The project represents a sophisticated framework for Bayesian goal inference in planning domains, with applications in cognitive modeling, robotics, and AI safety.