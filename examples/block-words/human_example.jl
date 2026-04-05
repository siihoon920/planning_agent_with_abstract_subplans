using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")
include("../../domains/block-words/experiment-scenarios.jl")

println("Saving outputs to: ", @__DIR__)

#--- Experiment Setup ---#
# Allow passing experiment ID via command line, defaulting to 1_1
exp_id = get(ARGS, 1, "1_1")
exp_id_dash = replace(exp_id, "_" => "-") # e.g. "1-1"

# Load domain and problem
domain_path = joinpath(@__DIR__, "../../domains/block-words/domain.pddl")
problem_path = joinpath(@__DIR__, "../../domains/block-words/experiment-$exp_id_dash.pddl")
domain = load_domain(domain_path)
problem = load_problem(problem_path)

# Initialize state and construct goal specification
state = initstate(domain, problem)
spec = Specification(problem)

# Compile domain for faster performance
domain, state = PDDL.compiled(domain, state)

#--- Define Renderer ---#
renderer = BlocksworldRenderer(resolution=(800, 800))
canvas = renderer(domain, state)

#--- Generate Trajectory and Storyboard ---#
# Load plan from experiment scenarios mapping
plan_strings = get_action(exp_id_dash)
plan = [parse_pddl(a) for a in plan_strings]
obs_traj = PDDL.simulate(domain, state, plan)

anim = anim_plan(renderer, domain, state, plan;
                 format="gif", transition=PDDLViz.StepTransition(),
                 framerate=2)

# Save the trajectory gif
trajectory_gif_path = joinpath(@__DIR__, "human_trajectory_$exp_id.gif")
save(trajectory_gif_path, anim)
println("Saved trajectory animation to $trajectory_gif_path")

# Generate storyboard frames (pick a few evenly spaced timesteps)
ts = collect(1:max(1, div(length(plan), 3)):length(plan)+1)
storyboard = render_storyboard(
    anim, ts,
    subtitles = ["t = $(t-1)" for t in ts], # t=1 is state at step 0
    xlabels = ["t = $(t-1)" for t in ts],
    xlabelsize = 20, subtitlesize = 24,
    n_rows = 1
)

#--- Model Configuration and Human Data ---#
# Load goals specific to this experiment config
goal_words = sort(get_goal_space(exp_id_dash))
# Create distinct colors for the goals
goal_colors = Makie.colorschemes[:plasma][1:max(1, div(256, length(goal_words))):256]

# Load Human Data for this experiment
csv_path = joinpath(@__DIR__, "../../domains/block-words/average_human_results_arrays/$(exp_id).csv")
# Filter out empty lines to avoid parsing errors
lines = filter(l -> !isempty(strip(l)), readlines(csv_path))
human_data_1d = parse.(Float64, lines)

n_goals = length(goal_words)
n_time_steps = div(length(human_data_1d), n_goals)

# Reshape goal probabilities
human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)

#--- Goal Probability Visualization ---#
n_rows, n_cols = size(storyboard.layout)

# Add series plot at the bottom of the storyboard
ax, _ = series(
    storyboard[n_rows+1, 1:n_cols], human_goal_probs,
    color = goal_colors[1:length(goal_words)], labels=goal_words,
    axis = (xlabel="Time", ylabel = "Probability",
            limits=((1, size(human_goal_probs, 2)), (0, 1)))
)
axislegend(ax, ax, "Goals", framevisible=false)

# Add vertical lines at timesteps
ts_all = collect(1:size(human_goal_probs, 2))
vlines!(ax, ts_all, color=:black, linestyle=:dash)
positions = [(t + 0.1, 0.85) for t in ts_all]
text_labels = ["t = $t" for t in ts_all]
text!(ax, positions; text=text_labels, color = :black, fontsize=14)

# Resize to accommodate the new subplot
rowsize!(storyboard.layout, n_rows+1, Auto(0.5))
resize!(storyboard, 1600, 800)

# Save human inference storyboard
storyboard_path = joinpath(@__DIR__, "human_goal_inference_storyboard_$exp_id.png")
save(storyboard_path, storyboard)
println("Saved human goal inference storyboard successfully to $storyboard_path!")
