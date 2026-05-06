using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using DelimitedFiles

PDDL.Arrays.register!()

domain_path = joinpath(@__DIR__, "../example/doors-keys-gems/domain.pddl")
plans_dir   = joinpath(@__DIR__, "../domains/doors-keys-gems/plans")

goals       = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_names  = [write_pddl(g) for g in goals]
goal_addr   = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

n_samples = 200

const JUDGEMENT_POINTS_SIPS = [
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

exp_ids = length(ARGS) > 0 ?
    collect(ARGS) :
    ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4",
     "3_1", "3_2", "3_3", "3_4", "4_1", "4_2", "4_3", "4_4"]

for exp_id in exp_ids
    println("\n" * "="^60)
    println("Running SIPS (budget=0.01) for: $exp_id")
    println("="^60)

    exp_parts  = split(exp_id, "_")
    jp_problem = parse(Int, exp_parts[1])
    jp_set     = parse(Int, exp_parts[2])
    jp_times   = JUDGEMENT_POINTS_SIPS[(jp_problem - 1) * 4 + jp_set]

    plan_files    = readdir(plans_dir)
    matched_files = filter(f -> startswith(f, "$(exp_id)_"), plan_files)
    if isempty(matched_files)
        @warn "No plan file for $exp_id — skipping."
        continue
    end
    plan_file = matched_files[1]

    m = match(r"problem_(\d+)_", plan_file)
    prob_id = m === nothing ?
        error("Cannot parse problem ID from $plan_file") : m.captures[1]

    problem_path = joinpath(@__DIR__, "../example/doors-keys-gems/problems_modified/problem-$(prob_id).pddl")

    domain  = load_domain(domain_path)
    problem = load_problem(problem_path)
    state   = initstate(domain, problem)
    domain, state = PDDL.compiled(domain, state)

    plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(plans_dir, plan_file)))
    plan         = [parse_pddl(a) for a in plan_strings]
    obs_traj     = PDDL.simulate(domain, state, plan)

    obs_params = ObsNoiseParams(
        (pddl"(xpos)",                                   normal, 1.0),
        (pddl"(ypos)",                                   normal, 1.0),
        (pddl"(forall (?d - door) (locked ?d))",         0.05),
        (pddl"(forall (?i - item) (has ?i))",            0.05),
        (pddl"(forall (?i - item) (offgrid ?i))",        0.05)
    )
    obs_params = ground_obs_params(obs_params, domain, state)
    obs_terms  = collect(keys(obs_params))

    sips_planner = ProbAStarPlanner(RelaxedMazeDist(), search_noise=0.1)

    sips_agent_config = AgentConfig(
        domain,
        sips_planner;
        goal_config = StaticGoalConfig(goal_prior),
        replan_args = (
            prob_replan      = 0.1,
            budget_dist      = shifted_neg_binom,
            budget_dist_args = (2, 0.01, 1)
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

    println("Running SIPS ($(n_samples) particles, budget_p=0.01)...")
    sips_runner(
        n_samples,
        state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
        init_args = (init_strata=goal_strata,),
        callback  = sips_logger_cb
    )

    sips_goal_probs = reduce(hcat, sips_logger_cb.data[:goal_probs])

    out_dir  = joinpath(@__DIR__, "../example/doors-keys-gems/goal_probs_SIPS_budget001")
    mkpath(out_dir)
    csv_path = joinpath(out_dir, "goal_probs_SIPS_budget001_$(exp_id).csv")
    open(csv_path, "w") do io
        println(io, join(goal_names, ","))
        for t in 2:size(sips_goal_probs, 2)
            println(io, join(sips_goal_probs[:, t], ","))
        end
    end
    println("Saved → $csv_path")
end

println("\nAll experiments complete.")
