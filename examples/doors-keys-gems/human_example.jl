using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")

println("Saving outputs to: ", @__DIR__)

# Allow passing experiment ID via command line, defaulting to 1_1
exp_id = get(ARGS, 1, "1_1")

# Find corresponding plan file and problem ID
plans_dir = joinpath(@__DIR__, "../../domains/doors-keys-gems/plans")
plan_files = readdir(plans_dir)
matched_files = filter(f -> startswith(f, "$(exp_id)_"), plan_files)
if isempty(matched_files)
    error("Could not find plan file for experiment $(exp_id)")
end
plan_file = matched_files[1]

# Extract problem number, e.g., "1_1_problem_6_goal1_0.dat" -> "6"
m = match(r"problem_(\d+)_", plan_file)
prob_id = m === nothing ? "6" : m.captures[1]

#--- Initial Setup ---#
PDDL.Arrays.register!()

domain_path = joinpath(@__DIR__, "../../domains/doors-keys-gems/domain.pddl")
problem_path = joinpath(@__DIR__, "../../domains/doors-keys-gems/problem-$(prob_id).pddl")

domain = load_domain(domain_path)
problem = load_problem(problem_path)

state = initstate(domain, problem)
spec = Specification(problem)

domain, state = PDDL.compiled(domain, state)

#--- Renderer ---#
gem_colors = PDDLViz.colorschemes[:vibrant]

renderer = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    agent_renderer = (d, s) -> HumanGraphic(color=:black),
    obj_renderers = Dict(
        :key => (d, s, o) -> KeyGraphic(
            visible=!s[Compound(:has, [o])]
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            visible=s[Compound(:locked, [o])]
        ),
        :gem => (d, s, o) -> GemGraphic(
            visible=!s[Compound(:has, [o])],
            color=gem_colors[parse(Int, string(o.name)[end])]
        )
    ),
    show_inventory = true,
    inventory_fns = [(d, s, o) -> s[Compound(:has, [o])]],
    inventory_types = [:item]
)

#--- Model Configuration ---#
goals = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_idxs = collect(1:length(goals))
goal_names = [write_pddl(g) for g in goals]
goal_colors = gem_colors[goal_idxs]

#--- Generate Trajectory ---#
# Load plan strings from matching human plan data
plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(plans_dir, plan_file)))
plan = [parse_pddl(a) for a in plan_strings]

obs_traj = PDDL.simulate(domain, state, plan)

anim_traj = anim_trajectory(
    renderer, domain, obs_traj;
    framerate=5, format="gif", trail_length=10
)

# Save the trajectory gif
trajectory_gif_path = joinpath(@__DIR__, "human_trajectory_$exp_id.gif")
save(trajectory_gif_path, anim_traj)
println("Saved trajectory animation to $trajectory_gif_path")

# Create a storyboard with a few key frames dynamically based on plan length
frame_idxs = collect(1:max(1, div(length(plan), 3)):length(plan)+1)
storyboard = render_storyboard(
    anim_traj,
    frame_idxs;
    subtitles = ["t = $(t-1)" for t in frame_idxs],
    xlabels = ["t = $(t-1)" for t in frame_idxs],
    xlabelsize = 20,
    subtitlesize = 24
)

#--- Load Human Data ---#
csv_path = joinpath(@__DIR__, "../../domains/doors-keys-gems/average_human_results_arrays/$(exp_id).csv")
human_data_1d = vec(readdlm(csv_path, ',', Float64))
n_goals = length(goals)
n_time_steps = div(length(human_data_1d), n_goals)

# Reshape goal probs so time steps are columns
human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)

#--- Goal Probability Visualization ---#
# Apply lines using storyboard_goal_lines! defined in utils.jl
storyboard_goal_lines!(
    storyboard,
    human_goal_probs,
    collect(1:n_time_steps);
    goal_names = goal_names,
    goal_colors = goal_colors,
    show_legend=true
)

# Save human inference storyboard
storyboard_path = joinpath(@__DIR__, "human_goal_inference_storyboard_$exp_id.png")
save(storyboard_path, storyboard)
println("Saved human goal inference storyboard successfully to $storyboard_path!")
