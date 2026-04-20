## Identical to example.jl except the trajectory comes from the 1_1 human plan
## (problem-6, goal=gem2).  Use this to verify that human_example_abstract.jl
## runs exactly the same SIPS as the reference example.jl.

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie

include("utils.jl")

PDDL.Arrays.register!()

# ── Domain / State (same as example.jl) ───────────────────────────────────────
domain  = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem = load_problem(joinpath(@__DIR__, "problems", "problem-6.pddl"))

state = initstate(domain, problem)
domain, state = PDDL.compiled(domain, state)

# ── Load 1_1 human trajectory ─────────────────────────────────────────────────
plans_dir = joinpath(@__DIR__, "../../domains/doors-keys-gems/plans")
plan_file = "1_1_problem_6_goal1_0.dat"
plan_strings = filter(l -> !isempty(strip(l)),
                      readlines(joinpath(plans_dir, plan_file)))
plan = [parse_pddl(a) for a in plan_strings]
obs_traj = PDDL.simulate(domain, state, plan)
println("Plan length: $(length(plan)), trajectory length: $(length(obs_traj))")

# ── Model configuration (copy-pasted verbatim from example.jl) ────────────────
goals     = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_idxs = collect(1:length(goals))
goal_names = [write_pddl(g) for g in goals]

@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

goal_addr   = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

heuristic    = RelaxedMazeDist()
planner_sips = ProbAStarPlanner(heuristic, search_noise=0.1)

agent_config = AgentConfig(
    domain,
    planner_sips;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan      = 0.1,
        budget_dist      = shifted_neg_binom,
        budget_dist_args = (2, 0.05, 1)
    ),
    act_epsilon = 0.05
)

obs_params = ObsNoiseParams(
    (pddl"(xpos)",                                   normal, 1.0),
    (pddl"(ypos)",                                   normal, 1.0),
    (pddl"(forall (?d - door) (locked ?d))",         0.05),
    (pddl"(forall (?i - item) (has ?i))",            0.05),
    (pddl"(forall (?i - item) (offgrid ?i))",        0.05)
)
obs_params = ground_obs_params(obs_params, domain, state)
obs_terms  = collect(keys(obs_params))

world_config = WorldConfig(
    agent_config = agent_config,
    env_config   = PDDLEnvConfig(domain, state),
    obs_config   = MarkovObsConfig(domain, obs_params)
)

# ── SIPS (copy-pasted verbatim from example.jl) ───────────────────────────────
t_obs_iter = state_choicemap_pairs(obs_traj, obs_terms; batch_size=1)

logger_cb = DataLoggerCallback(
    t          = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
    lml_est    = pf -> log_ml_estimate(pf)::Float64,
)

sips = SIPS(
    world_config,
    resample_cond  = :ess,
    rejuv_cond     = :periodic,
    rejuv_kernel   = ReplanKernel(2),
    period         = 2
)

n_samples = 120

println("\nRunning SIPS (n=$n_samples)...")
pf_state = sips(
    n_samples,
    t_obs_iter;
    init_args = (init_strata=goal_strata,),
    callback  = logger_cb
)

# ── Print final goal probabilities ────────────────────────────────────────────
goal_probs = reduce(hcat, logger_cb.data[:goal_probs])

println("\nGoal probabilities over time:")
println(join(["t"; goal_names], "\t"))
for t in 1:size(goal_probs, 2)
    println(join([string(t-1); map(p -> @sprintf("%.3f", p), goal_probs[:, t])], "\t"))
end

println("\nFinal (t=$(length(plan))):")
for (name, p) in zip(goal_names, goal_probs[:, end])
    println("  $name : $(@sprintf("%.4f", p))")
end

# ── Diagnostic: particle state ────────────────────────────────────────────────
println("\n[DIAG] Weighted goal distribution from particle filter:")
traces  = GenParticleFilters.get_traces(pf_state)
weights = GenParticleFilters.get_norm_weights(pf_state)
goal_w  = Dict{Int, Float64}()
for (tr, w) in zip(traces, weights)
    g = tr[goal_addr]
    goal_w[g] = get(goal_w, g, 0.0) + w
end
for g in sort(collect(keys(goal_w)))
    println("  goal=$g  ($(goals[g]))  weighted_prob=$(round(goal_w[g]; digits=4))")
end

best_idx = argmax(weights)
best_tr  = traces[best_idx]
T_final  = length(plan)
env_final = best_tr[:timestep => T_final => :env]
xpos_t   = parse_pddl("(xpos)")
ypos_t   = parse_pddl("(ypos)")
has2_t   = parse_pddl("(has gem2)")
println("[DIAG] Heaviest particle: goal=$(best_tr[goal_addr])  xpos=$(env_final[xpos_t])  ypos=$(env_final[ypos_t])  has_gem2=$(env_final[has2_t])")
println("[DIAG] Observed final:    xpos=$(obs_traj[end][xpos_t])  ypos=$(obs_traj[end][ypos_t])  has_gem2=$(obs_traj[end][has2_t])")

# ── Save CSV (same format as human_example_abstract.jl) ──────────────────────
csv_path = joinpath(@__DIR__, "example_1_1_test_sips_goal_probs.csv")
open(csv_path, "w") do io
    println(io, join(goal_names, ","))
    for t in 1:size(goal_probs, 2)
        println(io, join(goal_probs[:, t], ","))
    end
end
println("\nSaved → $csv_path")
