"""
make_storyboards.jl

Regenerates human / SIPS / hierarchical storyboards from pre-computed CSVs.
Does NOT re-run inference — only loads existing goal_probs CSVs and human data.

Run from repo root:
    julia --project=. src_new/make_storyboards.jl [exp_id ...]

If no arguments are given, all 16 experiments are processed.
Skips silently if a required CSV is missing.
"""

using PDDL, Printf
using SymbolicPlanners, Plinf
using PDDLViz, GLMakie
using DelimitedFiles

include("../example/doors-keys-gems/utils.jl")

PDDL.Arrays.register!()

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

const DOMAIN_PATH = joinpath(@__DIR__, "../example/doors-keys-gems/domain.pddl")
const PLANS_DIR   = joinpath(@__DIR__, "../domains/doors-keys-gems/plans")
const PROBS_DIR   = joinpath(@__DIR__, "../example/doors-keys-gems/problems_modified")
const SIPS_DIR    = joinpath(@__DIR__, "../example/doors-keys-gems/goal_probs_SIPS")
const ABS_DIR     = joinpath(@__DIR__, "../example/doors-keys-gems/goal_probs_hierarchical")
const HUMAN_DIR   = joinpath(@__DIR__, "../domains/doors-keys-gems/average_human_results_arrays")
const OUT_HUMAN   = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_human")
const OUT_SIPS    = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_SIPS")
const OUT_ABS     = joinpath(@__DIR__, "../example/doors-keys-gems/solutions/storyboard_hierarchical")

# ─────────────────────────────────────────────────────────────
# Renderer + goal setup
# ─────────────────────────────────────────────────────────────

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
goal_names  = [write_pddl(g) for g in goals]
goal_colors = gem_colors[1:length(goals)]

# ─────────────────────────────────────────────────────────────
# Judgement points
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# Helper: load pre-computed model CSV → (n_goals × T) matrix
# ─────────────────────────────────────────────────────────────

function load_model_csv(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])   # skip header row
    return collect(transpose(data))          # (n_goals × T)
end

# ─────────────────────────────────────────────────────────────
# Experiment list
# ─────────────────────────────────────────────────────────────

ALL_EXP_IDS = ["1_1","1_2","1_3","1_4","2_1","2_2","2_3","2_4",
               "3_1","3_2","3_3","3_4","4_1","4_2","4_3","4_4"]

exp_ids = length(ARGS) > 0 ? collect(ARGS) : ALL_EXP_IDS

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

for exp_id in exp_ids
    println("\n" * "="^60)
    println("Storyboards for: $exp_id")
    println("="^60)

    # ── Judgement points ──────────────────────────────────────
    exp_parts  = split(exp_id, "_")
    jp_problem = parse(Int, exp_parts[1])
    jp_set     = parse(Int, exp_parts[2])
    jp_times   = JUDGEMENT_POINTS[(jp_problem - 1) * 4 + jp_set]

    # ── Locate plan file ──────────────────────────────────────
    matched = filter(f -> startswith(f, "$(exp_id)_"), readdir(PLANS_DIR))
    if isempty(matched)
        @warn "No plan file for $exp_id — skipping."
        continue
    end
    plan_file = matched[1]

    m = match(r"problem_(\d+)_", plan_file)
    if m === nothing
        @warn "Cannot parse problem ID from $plan_file — skipping."
        continue
    end
    prob_id = m.captures[1]

    # ── Domain / state / trajectory ──────────────────────────
    domain  = load_domain(DOMAIN_PATH)
    problem = load_problem(joinpath(PROBS_DIR, "problem-$(prob_id).pddl"))
    state   = initstate(domain, problem)
    domain, state = PDDL.compiled(domain, state)

    plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(PLANS_DIR, plan_file)))
    plan         = [parse_pddl(a) for a in plan_strings]
    obs_traj     = PDDL.simulate(domain, state, plan)

    T = length(plan)
    println("Plan length: $T")

    anim_traj = anim_trajectory(renderer, domain, obs_traj;
        framerate=5, format="gif", trail_length=10)

    frame_idxs   = vcat([1], min.(jp_times .+ 1, length(obs_traj)))
    frame_titles = vcat(["t = 0"], ["t = $t" for t in jp_times])

    # ── Human data ───────────────────────────────────────────
    human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    if !isfile(human_path)
        @warn "No human data for $exp_id — skipping."
        continue
    end
    human_data_1d    = vec(readdlm(human_path, ',', Float64))
    n_time_steps     = div(length(human_data_1d), length(goals))
    human_goal_probs = reshape(human_data_1d, length(goals), n_time_steps)

    n_jp    = length(jp_times)
    n_human = min(n_jp + 1, n_time_steps)
    human_goal_probs_plot = human_goal_probs[:, 1:n_human]
    human_xs = vcat([0], jp_times[1:n_human-1])

    # ── Human storyboard ─────────────────────────────────────
    sb = render_storyboard(anim_traj, frame_idxs;
        subtitles=frame_titles, xlabels=frame_titles, xlabelsize=20, subtitlesize=24)
    storyboard_goal_lines!(sb, human_goal_probs_plot, jp_times;
        xs=human_xs, goal_names=goal_names, goal_colors=goal_colors, show_legend=true)
    out = joinpath(OUT_HUMAN, "storyboard_human_$(exp_id).png")
    save(out, sb)
    println("  Human       → $out")

    # ── SIPS storyboard ──────────────────────────────────────
    sips_path = joinpath(SIPS_DIR, "goal_probs_SIPS_$(exp_id).csv")
    if isfile(sips_path)
        sips_probs = load_model_csv(sips_path)
        sb = render_storyboard(anim_traj, frame_idxs;
            subtitles=frame_titles, xlabels=frame_titles, xlabelsize=20, subtitlesize=24)
        storyboard_goal_lines!(sb, sips_probs, jp_times;
            xs=collect(1:size(sips_probs,2)), goal_names=goal_names, goal_colors=goal_colors, show_legend=true)
        out = joinpath(OUT_SIPS, "storyboard_SIPS_$(exp_id).png")
        save(out, sb)
        println("  SIPS        → $out")
    else
        println("  SIPS CSV missing — skipped")
    end

    # ── Hierarchical storyboard ───────────────────────────────
    abs_path = joinpath(ABS_DIR, "goal_probs_hierarchical_$(exp_id).csv")
    if isfile(abs_path)
        abs_probs = load_model_csv(abs_path)
        sb = render_storyboard(anim_traj, frame_idxs;
            subtitles=frame_titles, xlabels=frame_titles, xlabelsize=20, subtitlesize=24)
        storyboard_goal_lines!(sb, abs_probs, jp_times;
            xs=collect(1:size(abs_probs,2)), goal_names=goal_names, goal_colors=goal_colors, show_legend=true)
        out = joinpath(OUT_ABS, "storyboard_hierarchical_$(exp_id).png")
        save(out, sb)
        println("  Hierarchical → $out")
    else
        println("  Hierarchical CSV missing — skipped")
    end
end

println("\nDone.")
