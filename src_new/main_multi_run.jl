"""
main_multi_run.jl  –  Multi-seed evaluation runner (no visualisation).

Runs SIPS and Abstract SIPS for all 16 experiments across N_RUNS independent
seeds, saving one CSV per model × experiment × run.  Storyboards and GIFs are
skipped entirely to keep runtime manageable.

Output files (same directories as main_with_hierarchical.jl):
  goal_probs_SIPS/goal_probs_SIPS_<exp_id>_run_<k>.csv
  goal_probs_hierarchical/goal_probs_hierarchical_<exp_id>_run_<k>.csv

Run from the repository root:
  julia --project=. src_new/main_multi_run.jl
  julia --project=. src_new/main_multi_run.jl 3_3 3_4   # specific experiments
"""

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using DelimitedFiles, Random

include("AbstractPlanner.jl")

PDDL.Arrays.register!()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

const N_RUNS    = 5
const N_SAMPLES = 200
const RUN_SEEDS = [111, 222, 333, 444, 555]

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

exp_ids = length(ARGS) > 0 ?
    collect(ARGS) :
    ["1_1","1_2","1_3","1_4","2_1","2_2","2_3","2_4",
     "3_1","3_2","3_3","3_4","4_1","4_2","4_3","4_4"]

# ──────────────────────────────────────────────────────────────────────────────
# Outer run loop
# ──────────────────────────────────────────────────────────────────────────────

for run_id in 1:N_RUNS
    Random.seed!(RUN_SEEDS[run_id])

    println("\n" * "█"^62)
    @printf "  RUN %d / %d   seed=%d   n_samples=%d\n" run_id N_RUNS RUN_SEEDS[run_id] N_SAMPLES
    println("█"^62)

    for exp_id in exp_ids
        AbstractPlanners.PhysicalPlanner.clear_physical_cache!()
        println("\n" * "="^60)
        @printf "Run %d | Experiment: %s\n" run_id exp_id
        println("="^60)

        # ── Locate plan file ─────────────────────────────────────────────────

        matched = filter(f -> startswith(f, "$(exp_id)_"), readdir(plans_dir))
        if isempty(matched)
            @warn "[$exp_id] No plan file found, skipping."
            continue
        end
        plan_file = matched[1]
        m = match(r"problem_(\d+)_", plan_file)
        m === nothing && error("Cannot parse problem ID from $plan_file")
        prob_id = m.captures[1]
        println("  Problem: $prob_id   Plan: $plan_file")

        # ── Domain / problem / trajectory ────────────────────────────────────

        problem_path = joinpath(@__DIR__,
            "../example/doors-keys-gems/problems_modified/problem-$(prob_id).pddl")
        domain  = load_domain(domain_path)
        problem = load_problem(problem_path)
        state   = initstate(domain, problem)
        domain, state = PDDL.compiled(domain, state)

        plan_strings = filter(l -> !isempty(strip(l)),
            readlines(joinpath(plans_dir, plan_file)))
        plan     = [parse_pddl(a) for a in plan_strings]
        obs_traj = PDDL.simulate(domain, state, plan)
        println("  Trajectory: $(length(plan)) actions")

        # ── Observation params ───────────────────────────────────────────────

        obs_params = ObsNoiseParams(
            (pddl"(xpos)",                               normal, 1.0),
            (pddl"(ypos)",                               normal, 1.0),
            (pddl"(forall (?d - door) (locked ?d))",     0.05),
            (pddl"(forall (?i - item) (has ?i))",        0.05),
            (pddl"(forall (?i - item) (offgrid ?i))",    0.05)
        )
        obs_params = ground_obs_params(obs_params, domain, state)
        obs_terms  = collect(keys(obs_params))

        # ── SIPS ─────────────────────────────────────────────────────────────

        sips_logger_cb = DataLoggerCallback(
            t          = (t, pf) -> t::Int,
            goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
            lml_est    = pf -> log_ml_estimate(pf)::Float64,
        )

        SIPS(
            WorldConfig(
                agent_config = AgentConfig(
                    domain,
                    ProbAStarPlanner(RelaxedMazeDist(), search_noise=0.1);
                    goal_config = StaticGoalConfig(goal_prior),
                    replan_args = (
                        prob_replan      = 0.1,
                        budget_dist      = shifted_neg_binom,
                        budget_dist_args = (2, 0.1, 1)
                    ),
                    act_epsilon = 0.05
                ),
                env_config = PDDLEnvConfig(domain, state),
                obs_config = MarkovObsConfig(domain, obs_params)
            ),
            resample_cond = :ess,
            rejuv_cond    = :periodic,
            rejuv_kernel  = ReplanKernel(2),
            period        = 2
        )(
            N_SAMPLES,
            state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
            init_args = (init_strata=goal_strata,),
            callback  = sips_logger_cb
        )

        sips_goal_probs = reduce(hcat, sips_logger_cb.data[:goal_probs])
        sips_csv = joinpath(@__DIR__,
            "../example/doors-keys-gems/goal_probs_SIPS/goal_probs_SIPS_$(exp_id)_run_$(run_id).csv")
        open(sips_csv, "w") do io
            println(io, join(goal_names, ","))
            for t in 2:size(sips_goal_probs, 2)
                println(io, join(sips_goal_probs[:, t], ","))
            end
        end
        println("  SIPS   → $sips_csv")

        # ── Abstract SIPS ────────────────────────────────────────────────────

        abs_logger_cb = DataLoggerCallback(
            t          = (t, pf) -> t::Int,
            goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
            lml_est    = pf -> log_ml_estimate(pf)::Float64,
        )

        SIPS(
            WorldConfig(
                agent_config = AgentConfig(
                    domain,
                    AbstractPlanners.AbstractPlanner(
                        RelaxedMazeDist(); search_noise=0.1, save_search=true
                    );
                    goal_config = StaticGoalConfig(goal_prior),
                    replan_args = (
                        prob_replan      = 0.1,
                        budget_dist      = shifted_neg_binom,
                        budget_dist_args = (2, 0.1, 1)
                    ),
                    act_epsilon = 0.05
                ),
                env_config = PDDLEnvConfig(domain, state),
                obs_config = MarkovObsConfig(domain, obs_params)
            ),
            resample_cond = :ess,
            rejuv_cond    = :periodic,
            rejuv_kernel  = SequentialKernel(InitGoalKernel(), ReplanKernel(2)),
            period        = 2
        )(
            N_SAMPLES,
            state_choicemap_pairs(obs_traj, obs_terms; batch_size=1);
            init_args = (init_strata=goal_strata,),
            callback  = abs_logger_cb
        )

        abs_goal_probs = reduce(hcat, abs_logger_cb.data[:goal_probs])
        abs_csv = joinpath(@__DIR__,
            "../example/doors-keys-gems/goal_probs_hierarchical/goal_probs_hierarchical_$(exp_id)_run_$(run_id).csv")
        open(abs_csv, "w") do io
            println(io, join(goal_names, ","))
            for t in 2:size(abs_goal_probs, 2)
                println(io, join(abs_goal_probs[:, t], ","))
            end
        end
        println("  Abstract → $abs_csv")
    end

    println("\nRun $run_id complete.")
end

println("\n" * "="^62)
println("All $N_RUNS runs complete.")
println("CSVs saved to:")
println("  example/doors-keys-gems/goal_probs_SIPS/")
println("  example/doors-keys-gems/goal_probs_hierarchical/")
println("="^62)
