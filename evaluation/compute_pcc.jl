"""
compute_pcc.jl  –  Pearson Correlation Coefficient evaluation for
                   SIPS model and Abstract (Hierarchical) model vs human data.

Human data loading mirrors main_with_hierarchical.jl exactly.
Model predictions are sampled at the specific action timesteps recorded in
domains/doors-keys-gems/stimuli/stimuli.json (the subfolder version),
rather than at evenly-spaced intervals.

Run from the repository root:
    julia --project=. evaluation/compute_pcc.jl

See evaluation/doc.md for full usage notes.
"""

using DelimitedFiles, Statistics, Printf

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

const REPO_ROOT    = dirname(@__DIR__)
const SIPS_DIR     = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_SIPS")
const ABS_DIR      = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_hierarchical")
const HUMAN_DIR    = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")
const STIMULI_PATH = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "stimuli", "stimuli.json")

const N_GOALS  = 3
const PROBLEMS = 1:4
const SETS     = 1:4

const OPTIMAL_PROBLEMS    = Set([1])
const SUBOPTIMAL_PROBLEMS = Set([3, 4])

# ─────────────────────────────────────────────────────────────
# Lightweight stimuli.json parser (no external deps)
#
# Extracts exp_id → Vector{Int} of action timesteps from the
# domains/doors-keys-gems/stimuli/stimuli.json file.
# ─────────────────────────────────────────────────────────────

function load_stimuli_times(path::String)
    times_map = Dict{String, Vector{Int}}()
    text = read(path, String)

    # Split into per-object blocks (objects are flat – no nested {})
    for obj in eachmatch(r"\{[^{}]+\}"s, text)
        obj_text = obj.match

        name_m  = match(r"\"name\"\s*:\s*\"scenario_(\d+_\d+)\"", obj_text)
        times_m = match(r"\"times\"\s*:\s*\[([0-9,\s]+)\]",       obj_text)
        (name_m === nothing || times_m === nothing) && continue

        exp_id = name_m.captures[1]
        times  = parse.(Int, strip.(split(times_m.captures[1], ",")))
        times_map[exp_id] = times
    end

    return times_map
end

# ─────────────────────────────────────────────────────────────
# Loaders  (human loading matches main_with_hierarchical.jl)
# ─────────────────────────────────────────────────────────────

"""
Load model CSV (SIPS or Abstract).
Row 1 is a text header; remaining rows are T × n_goals floats.
Returns (n_goals × T) matrix where column t = prediction at action step t.
"""
function load_model(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])   # (T × n_goals)
    return collect(transpose(data))          # (n_goals × T)
end

"""
Load human CSV – identical to main_with_hierarchical.jl:
    human_data_1d    = vec(readdlm(..., Float64))
    n_time_steps     = div(length(human_data_1d), n_goals)
    human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)

Julia's reshape is column-major, so the flat file stores values as
[goal1_t1, goal2_t1, goal3_t1, goal1_t2, ...].
Returns (n_goals × n_time_steps) matrix.
"""
function load_human(path::String, n_goals::Int = N_GOALS)
    human_data_1d    = vec(readdlm(path, ',', Float64))
    n_time_steps     = div(length(human_data_1d), n_goals)
    human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)
    return human_goal_probs
end

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

function pearson_r(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "Length mismatch: $(length(x)) vs $(length(y))"
    x̄, ȳ = mean(x), mean(y)
    num  = sum((x .- x̄) .* (y .- ȳ))
    den  = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

"""
Compute PCC between model and human by sampling the model at the specific
action timesteps listed in stimuli.json, then comparing to human data.

model  : (n_goals × T_model) — model output at every action step
human  : (n_goals × n_human) — human judgment at each survey point
times  : 1-indexed action steps at which human judgments were collected

When len(times) ≠ n_human (stimuli.json version mismatch), we take
min(len(times), n_human) and emit a warning.
"""
function pcc_at_times(model::Matrix, human::Matrix, times::Vector{Int}, exp_id::String)
    n_human = size(human, 2)
    n_times = length(times)

    if n_times != n_human
        @warn "[$exp_id] stimuli times ($n_times) ≠ human timesteps ($n_human); using min"
    end

    # Drop times that exceed the model's output length
    valid = filter(t -> 1 <= t <= size(model, 2), times)
    if length(valid) < n_times
        @warn "[$exp_id] $(n_times - length(valid)) time(s) exceed model length ($(size(model,2))), dropped"
    end

    n = min(length(valid), n_human)
    n < 2 && return NaN

    model_subset = model[:, valid[1:n]]
    human_subset = human[:, 1:n]

    return pearson_r(vec(model_subset), vec(human_subset))
end

# ─────────────────────────────────────────────────────────────
# Evaluate all 16 experiment sets
# ─────────────────────────────────────────────────────────────

struct Result
    exp_id   :: String
    problem  :: Int
    set      :: Int
    sips_pcc :: Float64
    abs_pcc  :: Float64
end

function evaluate_all()
    stimuli_times = load_stimuli_times(STIMULI_PATH)
    results = Result[]

    for problem in PROBLEMS, s in SETS
        exp_id     = "$(problem)_$(s)"
        human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
        sips_path  = joinpath(SIPS_DIR,  "goal_probs_SIPS_$(exp_id).csv")
        abs_path   = joinpath(ABS_DIR,   "goal_probs_hierarchical_$(exp_id).csv")

        if !isfile(human_path)
            @warn "[$exp_id] Human data missing, skipping."
            continue
        end

        if !haskey(stimuli_times, exp_id)
            @warn "[$exp_id] Not found in stimuli.json, skipping."
            continue
        end

        human_mat = load_human(human_path)
        times     = stimuli_times[exp_id]

        sips_r = if isfile(sips_path)
            pcc_at_times(load_model(sips_path), human_mat, times, exp_id)
        else
            @warn "[$exp_id] SIPS data missing."
            NaN
        end

        abs_r = if isfile(abs_path)
            pcc_at_times(load_model(abs_path), human_mat, times, exp_id)
        else
            @warn "[$exp_id] Abstract model data missing."
            NaN
        end

        push!(results, Result(exp_id, problem, s, sips_r, abs_r))
    end

    return results
end

# ─────────────────────────────────────────────────────────────
# Summary & printing
# ─────────────────────────────────────────────────────────────

nanmean(vals) = begin
    v = filter(!isnan, vals)
    isempty(v) ? NaN : mean(v)
end

function print_table(results::Vector{Result})
    sep = "─" ^ 44
    println("\n", sep)
    @printf "%-8s  %12s  %14s\n" "Exp" "SIPS PCC" "Abstract PCC"
    println(sep)
    for r in results
        @printf "%-8s  %12.4f  %14.4f\n" r.exp_id r.sips_pcc r.abs_pcc
    end
    println(sep)

    sips_all = [r.sips_pcc for r in results]
    abs_all  = [r.abs_pcc  for r in results]
    sips_opt = [r.sips_pcc for r in results if r.problem in OPTIMAL_PROBLEMS]
    abs_opt  = [r.abs_pcc  for r in results if r.problem in OPTIMAL_PROBLEMS]
    sips_sub = [r.sips_pcc for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
    abs_sub  = [r.abs_pcc  for r in results if r.problem in SUBOPTIMAL_PROBLEMS]

    println()
    sep2 = "─" ^ 54
    println(sep2)
    @printf "%-24s  %12s  %14s\n" "Category" "SIPS PCC" "Abstract PCC"
    println(sep2)
    @printf "%-24s  %12.4f  %14.4f\n" "Overall mean"        nanmean(sips_all) nanmean(abs_all)
    @printf "%-24s  %12.4f  %14.4f\n" "Optimal (prob 1)"    nanmean(sips_opt) nanmean(abs_opt)
    @printf "%-24s  %12.4f  %14.4f\n" "Sub-optimal (3 & 4)" nanmean(sips_sub) nanmean(abs_sub)
    println(sep2, "\n")
end

# ─────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────

function save_csv(results::Vector{Result})
    out = joinpath(@__DIR__, "pcc_results.csv")
    open(out, "w") do io
        println(io, "exp_id,problem,set,sips_pcc,abstract_pcc")
        for r in results
            @printf io "%s,%d,%d,%.6f,%.6f\n" r.exp_id r.problem r.set r.sips_pcc r.abs_pcc
        end

        sips_all = [r.sips_pcc for r in results]
        abs_all  = [r.abs_pcc  for r in results]
        sips_opt = [r.sips_pcc for r in results if r.problem in OPTIMAL_PROBLEMS]
        abs_opt  = [r.abs_pcc  for r in results if r.problem in OPTIMAL_PROBLEMS]
        sips_sub = [r.sips_pcc for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
        abs_sub  = [r.abs_pcc  for r in results if r.problem in SUBOPTIMAL_PROBLEMS]

        println(io, "\ncategory,sips_pcc,abstract_pcc")
        @printf io "overall,%.6f,%.6f\n"    nanmean(sips_all) nanmean(abs_all)
        @printf io "optimal,%.6f,%.6f\n"    nanmean(sips_opt) nanmean(abs_opt)
        @printf io "suboptimal,%.6f,%.6f\n" nanmean(sips_sub) nanmean(abs_sub)
    end
    println("Saved → $out")
end

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

results = evaluate_all()
print_table(results)
save_csv(results)
