using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie

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

canvas = renderer(domain, state)

#--- Visualize Plans ---#

planner = AStarPlanner(GoalManhattan(), save_search=true)
sol = planner(domain, state, spec)

plan = collect(sol)
canvas = renderer(canvas, domain, state, plan)

@assert satisfy(domain, sol.trajectory[end], problem.goal) == true

canvas = renderer(canvas, domain, state, sol, show_trajectory=false)

# PLAN ANIMATION
anim_plan_obj = anim_plan(
    renderer,
    domain,
    state,
    plan;
    format="gif",
    framerate=5,
    trail_length=10
)

save(joinpath(@__DIR__, "plan.gif"), anim_plan_obj)

#--- Model Configuration ---#

goals = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_idxs = collect(1:length(goals))
goal_names = [write_pddl(g) for g in goals]
goal_colors = gem_colors[goal_idxs]

@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

goal_addr = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

heuristic = RelaxedMazeDist()
planner = ProbAStarPlanner(heuristic, search_noise=0.1)

agent_config = AgentConfig(
    domain,
    planner;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan = 0.1,
        budget_dist = shifted_neg_binom,
        budget_dist_args = (2, 0.05, 1)
    ),
    act_epsilon = 0.05
)

obs_params = ObsNoiseParams(
    (pddl"(xpos)", normal, 1.0),
    (pddl"(ypos)", normal, 1.0),
    (pddl"(forall (?d - door) (locked ?d))", 0.05),
    (pddl"(forall (?i - item) (has ?i))", 0.05),
    (pddl"(forall (?i - item) (offgrid ?i))", 0.05)
)

obs_params = ground_obs_params(obs_params, domain, state)
obs_terms = collect(keys(obs_params))

world_config = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, state),
    obs_config = MarkovObsConfig(domain, obs_params)
)

#--- Generate Trajectory ---#

sol1 = planner(domain, state, pddl"(has key2)")
sol2 = planner(domain, sol1.trajectory[end], pddl"(not (locked door2))")
sol3 = planner(domain, sol2.trajectory[end], pddl"(has key1)")
sol4 = planner(domain, sol3.trajectory[end], pddl"(has gem3)")

plan = [collect(sol1); collect(sol2); collect(sol3); collect(sol4)]

obs_traj = PDDL.simulate(domain, state, plan)

# TRAJECTORY ANIMATION
anim_traj = anim_trajectory(
    renderer,
    domain,
    obs_traj;
    framerate=5,
    format="gif",
    trail_length=10
)

save(joinpath(@__DIR__, "trajectory.gif"), anim_traj)

# Storyboard from trajectory
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

#--- Online Goal Inference ---#

t_obs_iter = state_choicemap_pairs(obs_traj, obs_terms; batch_size=1)

callback = DKGCombinedCallback(
    renderer,
    domain;
    goal_addr = goal_addr,
    goal_names = ["red", "yellow", "blue"],
    goal_colors = goal_colors,
    obs_trajectory = obs_traj,
    print_goal_probs = true,
    plot_goal_bars = false,
    plot_goal_lines = false,
    render = true,
    inference_overlay = true,
    record = true
)

sips = SIPS(
    world_config,
    resample_cond=:ess,
    rejuv_cond=:periodic,
    rejuv_kernel=ReplanKernel(2),
    period=2
)

n_samples = 120

pf_state = sips(
    n_samples,
    t_obs_iter;
    init_args=(init_strata=goal_strata,),
    callback=callback
)

# INFERENCE ANIMATION
anim_inf = callback.record.animation
save(joinpath(@__DIR__, "inference.gif"), anim_inf)

#--- Goal Probability Visualization ---#

goal_probs = reduce(hcat, callback.logger.data[:goal_probs])[:, 1:25]

storyboard_goal_lines!(
    storyboard,
    goal_probs,
    [4, 9, 17, 21],
    show_legend=true
)

# SAVE STORYBOARD FIGURE
save(joinpath(@__DIR__, "goal_inference_storyboard.png"), storyboard)