## human_example_abstract.jl with the generated trajectory from example.jl
## (problem-6, gem3 goal: key2 → door2 → key1 → gem3).
## Runs both SIPS and Abstract planner inference so we can compare directly.

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")
include("../../src_new/AbstractPlanner.jl")

PDDL.Arrays.register!()

# ── Shared Setup (identical to human_example_abstract.jl) ─────────────────────

domain_path = joinpath(@__DIR__, "domain.pddl")

gem_colors = PDDLViz.colorschemes[:vibrant]

renderer = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    agent_renderer = (d, s) -> HumanGraphic(color=:black),
    obj_renderers = Dict(
        :key  => (d, s, o) -> KeyGraphic(visible=!s[Compound(:has, [o])]),
        :door => (d, s, o) -> LockedDoorGraphic(visible=s[Compound(:locked, [o])]),
        :gem  => (d, s, o) -> GemGraphic(
            visible=!s[Compound(:has, [o])],
            color=gem_colors[parse(Int, string(o.name)[end])]
        )
    ),
    show_inventory   = true,
    inventory_fns    = [(d, s, o) -> s[Compound(:has, [o])]],
    inventory_types  = [:item]
)

goals       = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_idxs   = collect(1:length(goals))
goal_names  = [write_pddl(g) for g in goals]
goal_colors = gem_colors[goal_idxs]
goal_addr   = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

n_samples = 120

# ── Domain / State ─────────────────────────────────────────────────────────────

domain  = load_domain(domain_path)
problem = load_problem(joinpath(@__DIR__, "problems/problem-6.pddl"))

state = initstate(domain, problem)
domain, state = PDDL.compiled(domain, state)

# ── Generate trajectory exactly as in example.jl ──────────────────────────────

gen_planner = ProbAStarPlanner(RelaxedMazeDist(), search_noise=0.1)

sol1 = gen_planner(domain, state,              pddl"(has key2)")
sol2 = gen_planner(domain, sol1.trajectory[end], pddl"(not (locked door2))")
sol3 = gen_planner(domain, sol2.trajectory[end], pddl"(has key1)")
sol4 = gen_planner(domain, sol3.trajectory[end], pddl"(has gem3)")

plan     = [collect(sol1); collect(sol2); collect(sol3); collect(sol4)]
obs_traj = PDDL.simulate(domain, state, plan)

println("Generated plan length: $(length(plan)), trajectory: $(length(obs_traj)) states.")
println("Plan actions:")
for (i, a) in enumerate(plan)
    println("  $i: $a")
end

# ── Observation params (identical to human_example_abstract.jl) ───────────────

obs_params = ObsNoiseParams(
    (pddl"(xpos)",                                   normal, 1.0),
    (pddl"(ypos)",                                   normal, 1.0),
    (pddl"(forall (?d - door) (locked ?d))",         0.05),
    (pddl"(forall (?i - item) (has ?i))",            0.05),
    (pddl"(forall (?i - item) (offgrid ?i))",        0.05)
)
obs_params = ground_obs_params(obs_params, domain, state)
obs_terms  = collect(keys(obs_params))

# ── Trajectory animation / storyboard ─────────────────────────────────────────

anim_traj = anim_trajectory(
    renderer, domain, obs_traj;
    framerate=5, format="gif", trail_length=10
)
save(joinpath(@__DIR__, "orig_traj_trajectory.gif"), anim_traj)

frame_idxs = collect(1:max(1, div(length(plan), 3)):length(plan)+1)
storyboard = render_storyboard(
    anim_traj, frame_idxs;
    subtitles  = ["t = $(t-1)" for t in frame_idxs],
    xlabels    = ["t = $(t-1)" for t in frame_idxs],
    xlabelsize = 20, subtitlesize = 24
)

# ── Run 1: Standard SIPS (identical config to human_example_abstract.jl) ──────

sips_planner = ProbAStarPlanner(RelaxedMazeDist(), search_noise=0.1)

sips_agent_config = AgentConfig(
    domain,
    sips_planner;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan      = 0.1,
        budget_dist      = shifted_neg_binom,
        budget_dist_args = (2, 0.05, 1)
    ),
    act_epsilon = 0.05
)

sips_world_config = WorldConfig(
    agent_config = sips_agent_config,
    env_config   = PDDLEnvConfig(domain, state),
    obs_config   = MarkovObsConfig(domain, obs_params)
)

sips_logger_cb = DataLoggerCallback(
    t          = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
    lml_est    = pf -> log_ml_estimate(pf)::Float64,
)

sips_runner = SIPS(
    sips_world_config,
    resample_cond  = :ess,
    rejuv_cond     = :periodic,
    rejuv_kernel   = ReplanKernel(2),
    period         = 2
)

println("\nRunning SIPS ($n_samples particles)...")
sips_pf = sips_runner(
    n_samples,
    state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
    init_args = (init_strata=goal_strata,),
    callback  = sips_logger_cb
)

# Diagnostic
println("\n[DIAG SIPS] Weighted goal distribution:")
let traces = GenParticleFilters.get_traces(sips_pf),
    weights = GenParticleFilters.get_norm_weights(sips_pf),
    goal_w  = Dict{Int,Float64}()
    for (tr, w) in zip(traces, weights)
        g = tr[goal_addr]; goal_w[g] = get(goal_w, g, 0.0) + w
    end
    for g in sort(collect(keys(goal_w)))
        println("  goal=$g ($(goals[g])): $(round(goal_w[g]; digits=4))")
    end
    best = argmax(weights)
    ef   = traces[best][:timestep => length(plan) => :env]
    xt, yt, h3t = parse_pddl("(xpos)"), parse_pddl("(ypos)"), parse_pddl("(has gem3)")
    println("  heaviest particle: goal=$(traces[best][goal_addr])  pos=($(ef[xt]),$(ef[yt]))  has_gem3=$(ef[h3t])")
    println("  observed final:    pos=($(obs_traj[end][xt]),$(obs_traj[end][yt]))  has_gem3=$(obs_traj[end][h3t])")
end

sips_goal_probs = reduce(hcat, sips_logger_cb.data[:goal_probs])

sips_csv_path = joinpath(@__DIR__, "orig_traj_sips_goal_probs.csv")
open(sips_csv_path, "w") do io
    println(io, join(goal_names, ","))
    for t in 1:size(sips_goal_probs, 2)
        println(io, join(sips_goal_probs[:, t], ","))
    end
end
println("Saved SIPS CSV → $sips_csv_path")

# ── Run 2: Abstract Planner SIPS (identical config to human_example_abstract.jl)

abs_planner = AbstractPlanners.AbstractPlanner(
    GoalManhattan(); search_noise=0.01, save_search=true
)

abs_agent_config = AgentConfig(
    domain,
    abs_planner;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan      = 0.1,
        budget_dist      = shifted_neg_binom,
        budget_dist_args = (2, 0.5, 1)
    ),
    act_epsilon = 0.05
)

abs_world_config = WorldConfig(
    agent_config = abs_agent_config,
    env_config   = PDDLEnvConfig(domain, state),
    obs_config   = MarkovObsConfig(domain, obs_params)
)

abs_logger_cb = DataLoggerCallback(
    t          = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
    lml_est    = pf -> log_ml_estimate(pf)::Float64,
)

abs_sips_runner = SIPS(
    abs_world_config,
    resample_cond  = :ess,
    rejuv_cond     = :periodic,
    rejuv_kernel   = ReplanKernel(2),
    period         = 2
)

println("\nRunning Abstract SIPS ($n_samples particles)...")
abs_sips_runner(
    n_samples,
    state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
    init_args = (init_strata=goal_strata,),
    callback  = abs_logger_cb
)

abs_goal_probs = reduce(hcat, abs_logger_cb.data[:goal_probs])

abs_csv_path = joinpath(@__DIR__, "orig_traj_abstract_goal_probs.csv")
open(abs_csv_path, "w") do io
    println(io, join(goal_names, ","))
    for t in 1:size(abs_goal_probs, 2)
        println(io, join(abs_goal_probs[:, t], ","))
    end
end
println("Saved Abstract CSV → $abs_csv_path")

# ── Storyboards ────────────────────────────────────────────────────────────────

T        = length(plan)
model_xs = collect(0:T)
model_ts = collect(1:T)   # for vlines (not used here, just pass empty)

sips_storyboard = render_storyboard(
    anim_traj, frame_idxs;
    subtitles  = ["t = $(t-1)" for t in frame_idxs],
    xlabels    = ["t = $(t-1)" for t in frame_idxs],
    xlabelsize = 20, subtitlesize = 24
)
storyboard_goal_lines!(
    sips_storyboard, sips_goal_probs, Int[];
    xs = model_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
)
save(joinpath(@__DIR__, "orig_traj_storyboard_sips.png"), sips_storyboard)
println("Saved SIPS storyboard → orig_traj_storyboard_sips.png")

storyboard_goal_lines!(
    storyboard, abs_goal_probs, Int[];
    xs = model_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
)
save(joinpath(@__DIR__, "orig_traj_storyboard_abstract.png"), storyboard)
println("Saved Abstract storyboard → orig_traj_storyboard_abstract.png")

println("\nDone.")
