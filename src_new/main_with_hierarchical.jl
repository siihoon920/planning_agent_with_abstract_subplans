using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("../example/doors-keys-gems/utils.jl")

include("AbstractPlanner.jl")

println("Saving outputs to: ", @__DIR__)

# ──────────────────────────────────────────────────────────────────────────────
# Shared Setup
# ──────────────────────────────────────────────────────────────────────────────

PDDL.Arrays.register!()

domain_path = joinpath(@__DIR__, "../example/doors-keys-gems/domain.pddl")
plans_dir   = joinpath(@__DIR__, "../domains/doors-keys-gems/plans")

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

n_samples = 200

const JUDGEMENT_POINTS = [
    [7, 17, 23],             # 1_1
    [9, 14, 17],             # 1_2
    [9, 17, 24],             # 1_3
    [7, 14, 23, 32],         # 1_4
    [6, 11, 24],             # 2_1
    [4, 6, 11],              # 2_2
    [5, 8, 13],              # 2_3
    [9, 12, 31, 44],         # 2_4
    [7, 22, 37, 50],         # 3_1
    [14, 24, 29, 40, 54],    # 3_2
    [7, 13, 20, 26],         # 3_3
    [6, 11, 26, 36, 49],     # 3_4
    [8, 14, 20],             # 4_1
    [4, 7, 10],              # 4_2
    [5, 8, 10],              # 4_3
    [7, 12, 18],             # 4_4
]

# Accept experiment IDs from command line, or run all 16 by default
exp_ids = length(ARGS) > 0 ?
collect(ARGS) :
    ["2_1"]
    #["1_1", "1_2", "1_3", "1_4","2_1", "2_2", "2_3", "2_4","3_1", "3_2", "3_3", "3_4","4_1", "4_2", "4_3", "4_4"]

# ──────────────────────────────────────────────────────────────────────────────
# Per-Experiment Loop
# ──────────────────────────────────────────────────────────────────────────────

for exp_id in exp_ids
    AbstractPlanners.PhysicalPlanner.clear_physical_cache!()
    println("\n" * "="^60)
    println("Running experiment: $exp_id")
    println("="^60)

    # ── Judgement points for this experiment ──────────────────────────────────

    exp_parts  = split(exp_id, "_")
    jp_problem = parse(Int, exp_parts[1])
    jp_set     = parse(Int, exp_parts[2])
    jp_times   = JUDGEMENT_POINTS[(jp_problem - 1) * 4 + jp_set]

    # ── Locate plan file ──────────────────────────────────────────────────────

    plan_files    = readdir(plans_dir)
    matched_files = filter(f -> startswith(f, "$(exp_id)_"), plan_files)
    if isempty(matched_files)
        @warn "No plan file found for $(exp_id), skipping."
        continue
    end
    plan_file = matched_files[1]
    println("Using plan file: $plan_file")

    m = match(r"problem_(\d+)_", plan_file)
    prob_id = m === nothing ?
        error("Cannot parse problem ID from $plan_file") : m.captures[1]
    println("Loading problem-$(prob_id).pddl ...")

    # ── Domain / State ────────────────────────────────────────────────────────

    problem_path = joinpath(@__DIR__, "../example/doors-keys-gems/problems_modified/problem-$(prob_id).pddl")

    domain  = load_domain(domain_path)
    problem = load_problem(problem_path)

    state = initstate(domain, problem)
    spec  = Specification(problem)

    domain, state = PDDL.compiled(domain, state)

    # ── Load Human Trajectory ─────────────────────────────────────────────────

    plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(plans_dir, plan_file)))
    plan         = [parse_pddl(a) for a in plan_strings]
    obs_traj     = PDDL.simulate(domain, state, plan)

    println("Loaded plan with $(length(plan)) actions, trajectory has $(length(obs_traj)) states.")

    # ── Observation Noise Params ──────────────────────────────────────────────

    obs_params = ObsNoiseParams(
        (pddl"(xpos)",                                   normal, 1.0),
        (pddl"(ypos)",                                   normal, 1.0),
        (pddl"(forall (?d - door) (locked ?d))",         0.05),
        (pddl"(forall (?i - item) (has ?i))",            0.05),
        (pddl"(forall (?i - item) (offgrid ?i))",        0.05)
    )
    obs_params = ground_obs_params(obs_params, domain, state)
    obs_terms  = collect(keys(obs_params))

    # ── Trajectory Animation ──────────────────────────────────────────────────
    
    anim_traj = anim_trajectory(
        renderer, domain, obs_traj;
        framerate=5, format="gif", trail_length=10
    )

    traj_gif_path = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/trajectories/plan_trajectory_$(exp_id).gif")
    save(traj_gif_path, anim_traj)
    println("Saved trajectory animation → $traj_gif_path")
    
    
    # Frame indices for storyboard: initial state + each judgement point state
    # jp_times are action-step indices (1-indexed); obs_traj frame = action_step + 1
    frame_idxs   = vcat([1], min.(jp_times .+ 1, length(obs_traj)))
    frame_titles = vcat(["t = 0"], ["t = $t" for t in jp_times])
    
    
    storyboard = render_storyboard(
        anim_traj, frame_idxs;
        subtitles  = frame_titles,
        xlabels    = frame_titles,
        xlabelsize = 20, subtitlesize = 24
    )

    #= ── Run 1: Standard SIPS (commented out) ───────────────────────────────

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

    println("\nRunning SIPS with standard planner ($(n_samples) particles)...")
    sips_pf_state = sips_runner(
        n_samples,
        state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
        init_args = (init_strata=goal_strata,),
        callback  = sips_logger_cb
    )

    sips_goal_probs = reduce(hcat, sips_logger_cb.data[:goal_probs])

    sips_csv_path = joinpath(@__DIR__, "../example/doors-keys-gems/goal_probs_SIPS/goal_probs_SIPS_$(exp_id).csv")
    open(sips_csv_path, "w") do io
        println(io, join(goal_names, ","))
        for t in 2:size(sips_goal_probs, 2)
            println(io, join(sips_goal_probs[:, t], ","))
        end
    end
    println("Saved SIPS model goal probabilities → $sips_csv_path")

    =# # ── end SIPS ──────────────────────────────────────────────────────────────

    # ── Run 2: Abstract Planner SIPS ──────────────────────────────────────────

    abs_planner = AbstractPlanners.AbstractPlanner(
        RelaxedMazeDist();
        search_noise=0.3, save_search=true
    )

    abs_agent_config = AgentConfig(
        domain,
        abs_planner;
        goal_config = StaticGoalConfig(goal_prior),
        replan_args = (
            prob_replan      = 0.1,
            budget_dist      = shifted_neg_binom,
            budget_dist_args = (2, 0.2, 1)
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
        rejuv_kernel   = SequentialKernel(InitGoalKernel(), ReplanKernel(2)),
        period         = 2
    )

    println("\nRunning SIPS with abstract planner ($(n_samples) particles)...")
    abs_sips_runner(
        n_samples,
        state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
        init_args = (init_strata=goal_strata,),
        callback  = abs_logger_cb
    )

    abs_goal_probs = reduce(hcat, abs_logger_cb.data[:goal_probs])

    abs_csv_path = joinpath(@__DIR__, "../example/doors-keys-gems/goal_probs_hierarchical/goal_probs_hierarchical_$(exp_id).csv")
    open(abs_csv_path, "w") do io
        println(io, join(goal_names, ","))
        for t in 2:size(abs_goal_probs, 2)
            println(io, join(abs_goal_probs[:, t], ","))
        end
    end
    println("Saved abstract model goal probabilities → $abs_csv_path")

    # ── Human Data ───────────────────────────────────────────────────────────

    csv_path         = joinpath(@__DIR__, "../domains/doors-keys-gems/average_human_results_arrays/$(exp_id).csv")
    human_data_1d    = vec(readdlm(csv_path, ',', Float64))
    n_time_steps     = div(length(human_data_1d), length(goals))
    human_goal_probs = reshape(human_data_1d, length(goals), n_time_steps)

    T = length(plan)

    # Trim human data to JP=1 (t=0) + actual JPs; use JP action-step numbers as x-coords
    n_jp    = length(jp_times)
    n_human = min(n_jp + 1, n_time_steps)
    human_goal_probs_plot = human_goal_probs[:, 1:n_human]
    human_xs = vcat([0], jp_times[1:n_human-1])
    model_xs = collect(0:T)   # full model x-axis (abs_goal_probs has T+1 columns)

    # ── Three Storyboards ─────────────────────────────────────────────────────

    # 1. Human storyboard
    human_storyboard = render_storyboard(
        anim_traj, frame_idxs;
        subtitles  = frame_titles,
        xlabels    = frame_titles,
        xlabelsize = 20, subtitlesize = 24
    )
    storyboard_goal_lines!(
        human_storyboard, human_goal_probs_plot, jp_times;
        xs = human_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
    )
    human_storyboard_path = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_human/storyboard_human_$(exp_id).png")
    save(human_storyboard_path, human_storyboard)
    println("Saved human storyboard → $human_storyboard_path")

    #= # 2. SIPS storyboard (commented out — SIPS runner is disabled above)
    sips_storyboard = render_storyboard(
        anim_traj, frame_idxs;
        subtitles  = frame_titles,
        xlabels    = frame_titles,
        xlabelsize = 20, subtitlesize = 24
    )
    storyboard_goal_lines!(
        sips_storyboard, sips_goal_probs, jp_times;
        xs = model_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
    )
    sips_storyboard_path = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_SIPS/storyboard_SIPS_$(exp_id).png")
    save(sips_storyboard_path, sips_storyboard)
    println("Saved SIPS storyboard → $sips_storyboard_path")
    =#

    # 3. Hierarchical storyboard
    storyboard_goal_lines!(
        storyboard, abs_goal_probs, jp_times;
        xs = model_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
    )
    abs_storyboard_path = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_hierarchical/storyboard_hierarchical_$(exp_id).png")
    save(abs_storyboard_path, storyboard)
    println("Saved hierarchical storyboard → $abs_storyboard_path")

    println("\nDone. Outputs for experiment $(exp_id):")
    # println("  Trajectory GIF          : $traj_gif_path")
    println("  Human storyboard        : $human_storyboard_path")
    println("  Hierarchical storyboard : $abs_storyboard_path")
    println("  Abstract CSV            : $abs_csv_path")

end

println("\nAll experiments complete.")
