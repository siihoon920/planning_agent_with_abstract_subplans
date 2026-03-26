using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")

println("Saving outputs to: ", @__DIR__)

#--- Initial Setup ---#
PDDL.Arrays.register!()

domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem = load_problem(joinpath(@__DIR__, "problems", "problem-6.pddl"))

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
heuristic = RelaxedMazeDist()
planner = ProbAStarPlanner(heuristic, search_noise=0.1)

sol1 = planner(domain, state, pddl"(has key2)")
sol2 = planner(domain, sol1.trajectory[end], pddl"(not (locked door2))")
sol3 = planner(domain, sol2.trajectory[end], pddl"(has key1)")
sol4 = planner(domain, sol3.trajectory[end], pddl"(has gem3)")

plan = [collect(sol1); collect(sol2); collect(sol3); collect(sol4)]

obs_traj = PDDL.simulate(domain, state, plan)

anim_traj = anim_trajectory(
    renderer, domain, obs_traj;
    framerate=5, format="gif", trail_length=10
)

storyboard = render_storyboard(
    anim_traj,
    [4, 9, 17, 21];
    subtitles = [
        "(i) Initially ambiguous goal",
        "(ii) Red eliminated upon key pickup",
        "(iii) Yellow most likely upon unlock",
        "(iv) Switch to blue upon backtracking"
    ],
    xlabels = ["t = 4", "t = 9", "t = 17", "t = 21"],
    xlabelsize = 20,
    subtitlesize = 24
)

#--- Load Human Data ---#
# By default, load 1_1.csv. This can be adapted to load any of the CSVs.
csv_path = joinpath(@__DIR__, "../../domains/doors-keys-gems/average_human_results_arrays/1_1.csv")
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
    collect(1:n_time_steps); # Overlaying vertical lines for each time step from the CSV
    goal_names = goal_names,
    goal_colors = goal_colors,
    show_legend=true
)

# Save human inference storyboard
save(joinpath(@__DIR__, "human_goal_inference_storyboard.png"), storyboard)
println("Saved human goal inference storyboard successfully!")
