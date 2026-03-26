using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")

println("Saving outputs to: ", @__DIR__)

#--- Initial Setup ---#
# Load domain and problem
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem = load_problem(joinpath(@__DIR__, "problems", "problem-1.pddl"))

# Initialize state and construct goal specification
state = initstate(domain, problem)
spec = Specification(problem)

# Compile domain for faster performance
domain, state = PDDL.compiled(domain, state)

#--- Define Renderer ---#
renderer = BlocksworldRenderer(resolution=(800, 800))
canvas = renderer(domain, state)

#--- Generate Trajectory and Storyboard ---#
# Use manually-specified trajectory from example.jl
plan = @pddl("(pick-up o)","(stack o w)","(unstack r p)","(stack r o)",
             "(unstack d a)","(put-down d)","(unstack a c)","(put-down a)",
             "(pick-up c)", "(stack c r)")
obs_traj = PDDL.simulate(domain, state, plan)

anim = anim_plan(renderer, domain, state, plan;
                 format="gif", transition=PDDLViz.StepTransition(),
                 framerate=2)

storyboard = render_storyboard(
    anim, [1, 3, 5, 7],
    subtitles = ["(i) Initial state",
                 "(ii) 'o' is stacked on 'w'",
                 "(iii) 'r' is stacked on 'o'",
                 "(iv) 'd' is unstacked from 'a'"],
    xlabels = ["t = 1", "t = 3", "t = 5", "t = 7"],
    xlabelsize = 20, subtitlesize = 24,
    n_rows = 2
)

#--- Model Configuration and Human Data ---#
goal_words = sort(["draw", "crow", "rope", "power", "wade"])
goal_colors = Makie.colorschemes[:plasma][1:32:32*length(goal_words)]

# Load Human Data
csv_path = joinpath(@__DIR__, "../../domains/block-words/average_human_results_arrays/1_1.csv")
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
    color = goal_colors, labels=goal_words,
    axis = (xlabel="Time", ylabel = "Probability",
            limits=((1, size(human_goal_probs, 2)), (0, 1)))
)
axislegend(ax, ax, "Goals", framevisible=false)

# Add vertical lines at timesteps
ts = 1:size(human_goal_probs, 2)
vlines!(ax, ts, color=:black, linestyle=:dash)
positions = [(t + 0.1, 0.85) for t in ts]
text_labels = ["t = $t" for t in ts]
text!(ax, positions; text=text_labels, color = :black, fontsize=14)

# Resize to accommodate the new subplot
rowsize!(storyboard.layout, n_rows+1, Auto(0.25))
resize!(storyboard, 2000, 1200)

# Save human inference storyboard
save(joinpath(@__DIR__, "human_goal_inference_storyboard.png"), storyboard)
println("Saved human goal inference storyboard successfully!")
