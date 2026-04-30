using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")
include("stimuli_times.jl")

# ──────────────────────────────────────────────────────────────────────────────
# AbstractPlanner wrapper
#
# AbstractPlanner.jl wraps a SymbolicPlanners.ForwardPlanner and calls
# PhysicalPlanner internally during expansion. It is self-contained as a module.
# We include it here and expose a thin callable struct so that SIPS / AgentConfig
# can use it as a drop-in planner.
# ──────────────────────────────────────────────────────────────────────────────
include("../../src_new/AbstractPlanner.jl")

"""
    HierarchicalPlanner

A thin wrapper around `AbstractPlanner.solve` that conforms to the
`SymbolicPlanners.Planner` interface used by `AgentConfig`.

Fields:
  - `inner` : a `ProbAStarPlanner` (or any `ForwardPlanner`) that the abstract
               planner uses internally for high-level search.
"""
struct HierarchicalPlanner <: SymbolicPlanners.Planner
    inner::SymbolicPlanners.ForwardPlanner
end

# Make HierarchicalPlanner callable: (planner)(domain, state, spec) -> solution
function (hp::HierarchicalPlanner)(
    domain::Domain, state::State, spec::Specification
)
    return AbstractPlanner.solve(hp.inner, domain, state, spec)
end

# Also support the two-arg signature used by some internal SIPS code paths
function SymbolicPlanners.solve(
    hp::HierarchicalPlanner, domain::Domain, state::State, spec::Specification
)
    return AbstractPlanner.solve(hp.inner, domain, state, spec)
end

Base.copy(hp::HierarchicalPlanner) = HierarchicalPlanner(copy(hp.inner))

function Base.getproperty(hp::HierarchicalPlanner, f::Symbol)
    f === :inner ? getfield(hp, :inner) : getproperty(hp.inner, f)
end

function Base.setproperty!(hp::HierarchicalPlanner, f::Symbol, v)
    setproperty!(hp.inner, f, v)
end

println("Saving outputs to: ", @__DIR__)

# ──────────────────────────────────────────────────────────────────────────────
# Experiment Setup
# ──────────────────────────────────────────────────────────────────────────────

# Accept experiment ID from command line, e.g.: julia human_example_abstract.jl 1_1
exp_id = get(ARGS, 1, "1_1")

# Locate the pre-recorded human plan file for this experiment ID.
# Files are named like: "1_1_problem_6_goal1_0.dat"
plans_dir = joinpath(@__DIR__, "plans")
plan_files = readdir(plans_dir)
matched_files = filter(f -> startswith(f, "$(exp_id)_"), plan_files)
if isempty(matched_files)
    error("No plan file found in $(plans_dir) with prefix '$(exp_id)_'.\n" *
          "Available experiments: $(join(first.(split.(plan_files, '_'), 2) .|> x -> join(x, "_"), ", "))")
end
plan_file = matched_files[1]
println("Using plan file: $plan_file")

# Extract embedded problem number from filename, e.g. "1_1_problem_6_goal1_0.dat" -> "6"
m = match(r"problem_(\d+)_", plan_file)
prob_id = m === nothing ? error("Cannot parse problem ID from $plan_file") : m.captures[1]
println("Loading problem-$(prob_id).pddl ...")

# ──────────────────────────────────────────────────────────────────────────────
# Domain / State Initialisation
# ──────────────────────────────────────────────────────────────────────────────

PDDL.Arrays.register!()

domain_path  = joinpath(@__DIR__, "domain.pddl")
problem_path = joinpath(@__DIR__, "problems/problem-$(prob_id).pddl")

domain  = load_domain(domain_path)
problem = load_problem(problem_path)

state = initstate(domain, problem)
spec  = Specification(problem)

domain, state = PDDL.compiled(domain, state)

# ──────────────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Goal Configuration
# ──────────────────────────────────────────────────────────────────────────────

goals      = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_idxs  = collect(1:length(goals))
goal_names = [write_pddl(g) for g in goals]
goal_colors = gem_colors[goal_idxs]

@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

goal_addr   = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical Planner + AgentConfig
#
# We pass ProbAStarPlanner(GoalManhattan()) as the inner forward planner that
# AbstractPlanner uses for its high-level A* search over abstract states.
# The physical planner (PhysicalPlanner.jl) is invoked automatically inside
# AbstractPlanner.expand! to discover candidate subgoals.
# ──────────────────────────────────────────────────────────────────────────────

inner_planner = ProbAStarPlanner(GoalManhattan(), search_noise=0.1, save_search=true)
hier_planner  = HierarchicalPlanner(inner_planner)

agent_config = AgentConfig(
    domain,
    hier_planner;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan        = 0.1,
        budget_dist        = shifted_neg_binom,
        budget_dist_args   = (2, 0.05, 1)
    ),
    act_epsilon = 0.05
)

# Observation noise model (identical to the SIPS baseline)
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

# ──────────────────────────────────────────────────────────────────────────────
# Load Human Trajectory (plan file)
# ──────────────────────────────────────────────────────────────────────────────

plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(plans_dir, plan_file)))
plan         = [parse_pddl(a) for a in plan_strings]
obs_traj     = PDDL.simulate(domain, state, plan)

println("Loaded plan with $(length(plan)) actions, trajectory has $(length(obs_traj)) states.")

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Animation
# ──────────────────────────────────────────────────────────────────────────────

anim_traj = anim_trajectory(
    renderer, domain, obs_traj;
    framerate=5, format="gif", trail_length=10
)

traj_gif_path = joinpath(@__DIR__, "human_trajectory_abstract_$(exp_id).gif")
save(traj_gif_path, anim_traj)
println("Saved trajectory animation → $traj_gif_path")

# Use the stimuli timepoints (1-indexed state numbers) as storyboard frames
frame_idxs = STIMULI_TIMES[exp_id]
storyboard = render_storyboard(
    anim_traj, frame_idxs;
    subtitles = ["t = $(t-1)" for t in frame_idxs],
    xlabels   = ["t = $(t-1)" for t in frame_idxs],
    xlabelsize = 20, subtitlesize = 24
)

# ──────────────────────────────────────────────────────────────────────────────
# SIPS Particle Filtering with the Abstract Planner
# ──────────────────────────────────────────────────────────────────────────────

t_obs_iter = state_choicemap_pairs(obs_traj, obs_terms; batch_size=1)

# DataLoggerCallback records goal probabilities at every timestep — we use
# these both for the storyboard overlay and for saving a model output CSV.
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

println("\nRunning SIPS with HierarchicalPlanner ($(n_samples) particles)...")
pf_state = sips(
    n_samples,
    t_obs_iter;
    init_args = (init_strata=goal_strata,),
    callback  = logger_cb
)

# ──────────────────────────────────────────────────────────────────────────────
# Extract and save model goal-probability output
# ──────────────────────────────────────────────────────────────────────────────

# goal_probs_matrix: shape (n_goals, T)
model_goal_probs = reduce(hcat, logger_cb.data[:goal_probs])

model_csv_path = joinpath(@__DIR__, "model_abstract_goal_probs_$(exp_id).csv")
open(model_csv_path, "w") do io
    # Header: goal names
    println(io, join(goal_names, ","))
    # Rows: one per timestep
    for t in 1:size(model_goal_probs, 2)
        println(io, join(model_goal_probs[:, t], ","))
    end
end
println("Saved abstract model goal probabilities → $model_csv_path")

# ──────────────────────────────────────────────────────────────────────────────
# Load Human Data + Build Comparison Storyboard
# ──────────────────────────────────────────────────────────────────────────────

csv_path      = joinpath(@__DIR__, "../../domains/doors-keys-gems/average_human_results_arrays/$(exp_id).csv")
human_data_1d = vec(readdlm(csv_path, ',', Float64))
n_goals       = length(goals)
n_time_steps  = div(length(human_data_1d), n_goals)

# Reshape: rows = goals, cols = timesteps
human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)

# Align model output length to human data length (truncate / pad if needed)
T_common = min(size(model_goal_probs, 2), n_time_steps)
model_clipped = model_goal_probs[:, 1:T_common]
human_clipped = human_goal_probs[:, 1:T_common]

# Add goal-probability subplot to the storyboard
storyboard_goal_lines!(
    storyboard,
    human_clipped,
    frame_idxs[1:T_common] .- 1;  # convert 1-indexed state numbers to action steps
    goal_names  = goal_names,
    goal_colors = goal_colors,
    show_legend = true
)

storyboard_path = joinpath(@__DIR__, "human_goal_inference_storyboard_abstract_$(exp_id).png")
save(storyboard_path, storyboard)
println("Saved storyboard → $storyboard_path")

println("\nDone. Outputs for experiment $(exp_id) (abstract planner):")
println("  Trajectory GIF : $traj_gif_path")
println("  Storyboard PNG : $storyboard_path")
println("  Model CSV      : $model_csv_path")
