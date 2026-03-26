# Comprehensive Summary: Online Bayesian Goal Inference for Boundedly-Rational Planning Agents

**Authors:** Tan Zhi-Xuan, Jordyn L. Mann, Tom Silver, Joshua B. Tenenbaum, Vikash K. Mansinghka
**Publication:** arXiv, October 2020

---

## 1. Introduction and Problem Statement

Humans possess a remarkable cognitive ability: we can easily infer the goals of others by observing their actions over time. Strikingly, we can accurately deduce these goals even when the observed actions are sub-optimal, or when the agent fails entirely due to mistakes, enabling us to step in and offer assistance.

Historically, AI approaches for inferring agent desires—such as inverse reinforcement learning (IRL) and plan recognition—have struggled to replicate this human capability. The primary limitation is that these models assume agents act optimally, or they model sub-optimality using highly simplified methods like Boltzmann-rational action noise.

These traditional approaches fail to account for the inherent difficulty of the *planning process itself*, which is often the true cause of sub-optimal behavior or failed plans. Furthermore, traditional IRL saddles the observer with a massive computational burden: to infer an agent's goal, the system must compute the optimal policy for *all* possible goals in advance by solving Markov Decision Processes (MDPs), which is deeply intractable in complex domains.

---

## 2. The Core Contribution

To address these limitations, the authors introduce a unified modeling and inference architecture that represents agents not as flawlessly optimal actors, but as **boundedly-rational planners**.

Instead of pre-computing full plans, this model assumes agents interleave resource-limited plan search with execution (replanning on the fly). By modeling this cognitive strategy as a probabilistic program, the architecture can perform efficient, online Bayesian inference over an agent's internal planning process and goals. To achieve this, the authors develop **Sequential Inverse Plan Search (SIPS)**, a sequential Monte Carlo (SMC) algorithm that incrementally extends hypothesized plans as new observations arrive.

---

## 3. Architecture and Generative Model

The architecture utilizes the probabilistic programming system **Gen** alongside the Planning Domain Definition Language (**PDDL**). PDDL acts as the environment model, structuring states, goals, and transitions using logical predicates and numeric fluents.

The agent is modeled as a generative process with the following mathematical structure:
- **Goal Prior:** $g \sim P(g)$
- **Plan Update:** $p_t \sim P(p_t|s_t, p_{t-1}, g)$
- **Action Selection:** $a_t \sim P(a_t|s_t, p_t)$
- **State Transition:** $s_{t+1} \sim P(s_{t+1}|s_t, a_t)$
- **Observation Noise:** $o_{t+1} \sim P(o_{t+1}|s_{t+1})$

### Modeling Boundedly-Rational Planning
To model the fact that full-horizon planning is computationally expensive, the architecture assumes the agent only searches up to a specific "budget" before executing a partial plan.
* This budget, denoted as $\eta$ (maximum nodes expanded), is sampled from a negative binomial distribution: $\eta \sim \text{NEGATIVE-BINOMIAL}(r, q)$.
* The underlying planning algorithm is a stochastic version of $A^*$ search. Instead of expanding the mathematically optimal state, it expands states based on a probability proportional to $exp(-f(s,g)/\gamma)$, where $\gamma$ is a noise parameter and $f(s,g)$ is the estimated total plan cost (path cost plus heuristic goal distance).

---

## 4. Sequential Inverse Plan Search (SIPS)

Computing the exact posterior over goals—$P(g|o_{1:t})$—is computationally intractable because it requires marginalizing over all unobserved states, actions, and internal plans.

SIPS solves this using an online sequential Monte Carlo procedure.
- **Particles and Extension:** SIPS maintains a set of weighted "particles," each representing a specific plan and goal hypothesis. Crucially, SIPS only extends a hypothesized plan if it does not already contain an action for the current time $t$ and state $s_t$, vastly reducing expensive planning calls.
- **Rejuvenation:** To prevent the particles from collapsing to a single incorrect hypothesis, SIPS utilizes a threshold to trigger resampling. It then applies Metropolis-Hastings rejuvenation kernels to maintain diversity.
* *Heuristic-driven goal proposal:* Proposes new goals that are heuristically close to the latest observed state.
* *Error-driven replanning proposal:* Finds the time when the hypothesized trajectory diverged from actual observations and proposes a new plan from that point.

---

## 5. Experimental Evaluation

The authors tested SIPS against Bayesian IRL (BIRL) baselines (both "unbiased" and "oracle" versions) across four complex domains characterized by compositional structure and sparse rewards:
1. **Taxi:** A standard gridworld task.
2. **Doors, Keys, & Gems:** Requires unlocking doors sequentially to reach a gem, allowing for dead-ends and failures.
3. **Block Words:** A Blocks World variant where the goal is to spell specific words.
4. **Intrusion Detection:** A massive cybersecurity state-space modeling server attacks.

### Key Results
- **Human-likeness:** In domains like *Doors, Keys, & Gems*, SIPS correctly adjusted its inferences sharply when an agent backtracked or failed myopically, mirroring human cognitive patterns. When compared to inferences made by human test subjects, SIPS achieved a remarkably high correlation ($r=0.89$), far outperforming the BIRL baseline ($r=0.51$).
- **Accuracy and Speed:** SIPS outperformed unbiased BIRL in both accuracy and speed in three of the four domains. Because traditional VI failed to converge in large state spaces, SIPS's online partial-planning approach resulted in average runtimes that were often orders of magnitude smaller.
- **Robustness:** SIPS demonstrated high resilience against "model mismatch"—meaning it successfully inferred goals even when the observed agents used different planning heuristics, varying levels of rationality, or were actual human pilots rather than simulated algorithms.

---

## 6. Limitations and Broader Impact

### Limitations
* The current model assumes a finite set of final goals, whereas humans navigate an infinite space of hierarchical and instrumental sub-goals.
* The environments tested are largely deterministic; expanding this to stochastic dynamics and continuous action spaces would require integrating sample-based motion planners or Monte Carlo Tree Search.

### Broader Societal Impact
The authors explicitly recognize the ethical weight of their research.
- **Positive Applications:** Understanding human failure is critical for "value alignment" in AI. Applications include smart user interfaces, intelligent assistants, and collaborative robotics that aid users who are struggling with a task.
- **Risks:** The capacity to infer intent from partial actions can easily be co-opted for surveillance, offensive military purposes, or manipulation. The authors caution that inferring "suspicious intent" could exacerbate systemic issues, such as the over-policing of marginalized communities.