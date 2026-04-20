using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DelimitedFiles

include("utils.jl")

# AbstractPlanner.jl defines module AbstractPlanners containing the AbstractPlanner
# struct (a proper Planner subtype) and its hierarchical search logic.
# PhysicalPlanner is called internally during expand!.
include("../../src_new/AbstractPlanner.jl")

println("Saving outputs to: ", @__DIR__)

# ──────────────────────────────────────────────────────────────────────────────
# Shared Setup
# ──────────────────────────────────────────────────────────────────────────────

PDDL.Arrays.register!()

domain_path = joinpath(@__DIR__, "domain.pddl")
plans_dir   = joinpath(@__DIR__, "../../domains/doors-keys-gems/plans")

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

# Accept experiment IDs from command line, or run all 8 by default
exp_ids = length(ARGS) > 0 ? collect(ARGS) :
    ["2_1"]
    #["2_1", "2_2", "2_3", "2_4"]
    #["1_1", "1_2", "1_3", "1_4", #

# ──────────────────────────────────────────────────────────────────────────────
# Per-Experiment Loop
# ──────────────────────────────────────────────────────────────────────────────

for exp_id in exp_ids
    println("\n" * "="^60)
    println("Running experiment: $exp_id")
    println("="^60)

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
    prob_id = m === nothing ? error("Cannot parse problem ID from $plan_file") : m.captures[1]
    println("Loading problem-$(prob_id).pddl ...")

    # ── Domain / State ────────────────────────────────────────────────────────

    problem_path = joinpath(@__DIR__, "problems/problem-$(prob_id).pddl")

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

    traj_gif_path = joinpath(@__DIR__, "human_trajectory_abstract_$(exp_id).gif")
    save(traj_gif_path, anim_traj)
    println("Saved trajectory animation → $traj_gif_path")

    frame_idxs = collect(1:max(1, div(length(plan), 3)):length(plan)+1)
    storyboard = render_storyboard(
        anim_traj, frame_idxs;
        subtitles = ["t = $(t-1)" for t in frame_idxs],
        xlabels   = ["t = $(t-1)" for t in frame_idxs],
        xlabelsize = 20, subtitlesize = 24
    )

    # ── Run 1: Standard SIPS (ProbAStarPlanner) ───────────────────────────────
#=
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

    # ── Diagnostic: final particle state ─────────────────────────────────────
    println("\n[DIAG] Final SIPS particle goals (goal index → goal term → weight):")
    traces  = GenParticleFilters.get_traces(sips_pf_state)
    weights = GenParticleFilters.get_norm_weights(sips_pf_state)
    goal_w  = Dict{Int, Float64}()
    for (tr, w) in zip(traces, weights)
        g = tr[goal_addr]
        goal_w[g] = get(goal_w, g, 0.0) + w
    end
    for g in sort(collect(keys(goal_w)))
        println("  goal=$g  ($(goals[g]))  weighted_prob=$(round(goal_w[g]; digits=4))")
    end
    # Also print position of the heaviest particle
    best_idx = argmax(weights)
    best_tr  = traces[best_idx]
    T_final  = length(plan)
    env_state_final = best_tr[:timestep => T_final => :env]
    xpos_term  = parse_pddl("(xpos)")
    ypos_term  = parse_pddl("(ypos)")
    has1_term  = parse_pddl("(has gem1)")
    has2_term  = parse_pddl("(has gem2)")
    has3_term  = parse_pddl("(has gem3)")
    println("[DIAG] Heaviest particle (w=$(round(weights[best_idx]; digits=6))):")
    println("  goal=$(best_tr[goal_addr])  xpos=$(env_state_final[xpos_term])  ypos=$(env_state_final[ypos_term])")
    println("  has gem1=$(env_state_final[has1_term])  has gem2=$(env_state_final[has2_term])  has gem3=$(env_state_final[has3_term])")
    obs_final = obs_traj[end]
    println("[DIAG] Observed final state: xpos=$(obs_final[xpos_term])  ypos=$(obs_final[ypos_term])  has gem2=$(obs_final[has2_term])")

    sips_goal_probs = reduce(hcat, sips_logger_cb.data[:goal_probs])

    sips_csv_path = joinpath(@__DIR__, "model_sips_goal_probs_$(exp_id).csv")
    open(sips_csv_path, "w") do io
        println(io, join(goal_names, ","))
        for t in 1:size(sips_goal_probs, 2)
            println(io, join(sips_goal_probs[:, t], ","))
        end
    end
    println("Saved SIPS model goal probabilities → $sips_csv_path")
=#
    # ── Run 2: Abstract Planner SIPS ──────────────────────────────────────────

    abs_planner = AbstractPlanners.AbstractPlanner(
        RelaxedMazeDist(); search_noise=0.5, save_search=true
    )

    abs_agent_config = AgentConfig(
        domain,
        abs_planner;
        goal_config = StaticGoalConfig(goal_prior),
        replan_args = (
            prob_replan      = 0.4,
            budget_dist      = shifted_neg_binom,
            budget_dist_args = (2, 0.1, 1)
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
        rejuv_kernel   =  SequentialKernel(InitGoalKernel(),ReplanKernel(2)),
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

    abs_csv_path = joinpath(@__DIR__, "model_abstract_goal_probs_$(exp_id).csv")
    open(abs_csv_path, "w") do io
        println(io, join(goal_names, ","))
        for t in 1:size(abs_goal_probs, 2)
            println(io, join(abs_goal_probs[:, t], ","))
        end
    end
    println("Saved abstract model goal probabilities → $abs_csv_path")

    # ── Critical Timesteps ────────────────────────────────────────────────────

    csv_path         = joinpath(@__DIR__, "../../domains/doors-keys-gems/average_human_results_arrays/$(exp_id).csv")
    human_data_1d    = vec(readdlm(csv_path, ',', Float64))
    n_time_steps     = div(length(human_data_1d), length(goals))
    human_goal_probs = reshape(human_data_1d, length(goals), n_time_steps)

    T        = length(plan)
    step     = div(T, n_time_steps + 1)
    human_ts = [step * i for i in 0:n_time_steps-1]       # e.g. [0,5,10,15]
    model_ts = [step * i for i in 1:div(T, step)]          # e.g. [5,10,15,20,25]
    model_xs = collect(0:T)

    # ── Three Storyboards ─────────────────────────────────────────────────────

    # 1. Human storyboard
    human_storyboard = render_storyboard(
        anim_traj, frame_idxs;
        subtitles = ["t = $(t-1)" for t in frame_idxs],
        xlabels   = ["t = $(t-1)" for t in frame_idxs],
        xlabelsize = 20, subtitlesize = 24
    )
    storyboard_goal_lines!(
        human_storyboard, human_goal_probs, human_ts;
        xs = human_ts, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
    )
    human_storyboard_path = joinpath(@__DIR__, "solutions/inference_human/storyboard_human_$(exp_id).png")
    save(human_storyboard_path, human_storyboard)
    println("Saved human storyboard → $human_storyboard_path")

    # 2. SIPS storyboard (skipped — SIPS run commented out)
    sips_storyboard_path = "(skipped)"

    # 3. Hierarchical storyboard
    storyboard_goal_lines!(
        storyboard, abs_goal_probs, model_ts;
        xs = model_xs, goal_names = goal_names, goal_colors = goal_colors, show_legend = true
    )
    abs_storyboard_path = joinpath(@__DIR__, "solutions/inference_hierarchical/storyboard_hierarchical_$(exp_id).png")
    save(abs_storyboard_path, storyboard)
    println("Saved hierarchical storyboard → $abs_storyboard_path")

    println("\nDone. Outputs for experiment $(exp_id):")
    println("  Trajectory GIF          : $traj_gif_path")
    println("  Human storyboard        : $human_storyboard_path")
    println("  SIPS storyboard         : $sips_storyboard_path")
    println("  Hierarchical storyboard : $abs_storyboard_path")
    println("  Abstract CSV            : $abs_csv_path")

end

println("\nAll experiments complete.")
