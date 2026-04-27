"""
compute_pcc.jl  –  Pearson Correlation Coefficient evaluation for
                   SIPS model and Abstract (Hierarchical) model vs human data.

Human data loading mirrors main_with_hierarchical.jl exactly.
Model predictions are sampled at the specific action timesteps recorded in
domains/doors-keys-gems/stimuli/stimuli.json.

Run from the repository root:
    julia --project=. evaluation/compute_pcc.jl

See evaluation/doc.md for full usage notes.
"""

using DelimitedFiles, Statistics, Printf

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

const REPO_ROOT = dirname(@__DIR__)
const SIPS_DIR  = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_SIPS")
const ABS_DIR   = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_hierarchical")
const HUMAN_DIR = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")

const N_GOALS  = 3
const PROBLEMS = 1:4
const SETS     = 1:4

const OPTIMAL_PROBLEMS    = Set([1])
const SUBOPTIMAL_PROBLEMS = Set([3, 4])

# Survey judgement-point timestamps, ordered: 1_1, 1_2, 1_3, 1_4,
#                                              2_1, 2_2, 2_3, 2_4,
#                                              3_1, 3_2, 3_3, 3_4,
#                                              4_1, 4_2, 4_3, 4_4
const JUDGEMENT_POINTS = [
    [1, 7, 17, 23],        # 1_1
    [1, 9, 14, 17],        # 1_2
    [1, 9, 17, 24],        # 1_3
    [1, 7, 14, 23, 32],    # 1_4
    [1, 6, 11, 24],        # 2_1
    [1, 4, 6, 11],         # 2_2
    [1, 5, 8, 13],         # 2_3
    [1, 9, 12, 31, 44],    # 2_4
    [1, 7, 22, 37, 50],    # 3_1
    [1, 14, 24, 29, 40, 54], # 3_2
    [1, 7, 13, 20, 26],    # 3_3
    [1, 6, 11, 26, 36, 49], # 3_4
    [1, 8, 14, 20],        # 4_1
    [1, 4, 7, 10],         # 4_2
    [1, 5, 8, 10],         # 4_3
    [1, 7, 12, 18],        # 4_4
]

# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

"""
Load a model CSV (SIPS or Abstract).
Row 1 is a text header; rows 2..end are one row per action step.
Returns (n_goals × T_model): column t = model prediction at action step t.
"""
function load_model(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])   # (T × n_goals)
    return collect(transpose(data))          # (n_goals × T)
end

"""
Load human CSV for a specific experiment.

The human CSV stores data ONLY for the timesteps listed in stimuli.json.
Layout (column-major, matching Julia's reshape default):
    row 1 : gem1 at times[1]
    row 2 : gem2 at times[1]
    row 3 : gem3 at times[1]
    row 4 : gem1 at times[2]
    ...

n_times is taken from stimuli.json for this experiment.

For 9/16 experiments the file has exactly n_times*3 rows.
For the remaining 7 the file has extra rows (additional undocumented survey
points appended at the end). We always take only the first n_times*3 rows so
that column i of the returned matrix corresponds to times[i].

Returns (n_goals × n_times) matrix.
"""
function load_human(path::String, n_times::Int, n_goals::Int = N_GOALS)
    human_data_1d = vec(readdlm(path, ',', Float64))
    n_available   = div(length(human_data_1d), n_goals)

    if n_available < n_times
        @warn "Human CSV has $n_available timesteps but stimuli.json lists $n_times; using $n_available"
        n_times = n_available
    end
    # Extra rows (n_available > n_times) are silently dropped — they are
    # undocumented survey points not listed in stimuli.json.

    trimmed = human_data_1d[1 : n_times * n_goals]
    return reshape(trimmed, n_goals, n_times)   # (n_goals × n_times)
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
Compute PCC by comparing:
    model[:, times[i]]  vs  human[:, i]   for i = 1 .. n_times

human is already shaped (n_goals × n_times) by load_human, so columns
align 1:1 with the times array.

Any times[i] that exceed the model's output length are dropped (with a
warning), and the corresponding human columns are dropped too.
"""
function pcc_at_times(model::Matrix, human::Matrix, times::Vector{Int}, exp_id::String)
    n_times = size(human, 2)   # == length(times) after load_human trimming

    # Identify valid times (within model's output range)
    valid_mask  = [1 <= t <= size(model, 2) for t in times[1:n_times]]
    n_dropped   = count(!, valid_mask)
    if n_dropped > 0
        @warn "[$exp_id] $n_dropped time(s) exceed model length ($(size(model,2))), dropped"
    end

    valid_times  = times[1:n_times][valid_mask]
    valid_cols   = (1:n_times)[valid_mask]

    length(valid_times) < 2 && return NaN

    model_subset = model[:, valid_times]      # (n_goals × n_valid)
    human_subset = human[:, valid_cols]       # (n_goals × n_valid)

    # Flatten: [gem1_t1, gem2_t1, gem3_t1, gem1_t2, ...]
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

        times     = JUDGEMENT_POINTS[(problem - 1) * length(SETS) + s]
        human_mat = load_human(human_path, length(times))

        sips_r = if isfile(sips_path)
            pcc_at_times(load_model(sips_path), human_mat, times, exp_id)
        else
            @warn "[$exp_id] SIPS model data missing."
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

    sips_no2 = [r.sips_pcc for r in results if r.problem != 2]
    abs_no2  = [r.abs_pcc  for r in results if r.problem != 2]

    println()
    sep2 = "─" ^ 54
    println(sep2)
    @printf "%-24s  %12s  %14s\n" "Category" "SIPS PCC" "Abstract PCC"
    println(sep2)
    @printf "%-24s  %12.4f  %14.4f\n" "Overall mean"        nanmean(sips_all) nanmean(abs_all)
    @printf "%-24s  %12.4f  %14.4f\n" "Optimal (prob 1)"    nanmean(sips_opt) nanmean(abs_opt)
    @printf "%-24s  %12.4f  %14.4f\n" "Sub-optimal (3 & 4)" nanmean(sips_sub) nanmean(abs_sub)
    println(sep2)

    println()
    println("  (excluding problem 2)")
    println(sep2)
    @printf "%-24s  %12s  %14s\n" "Category" "SIPS PCC" "Abstract PCC"
    println(sep2)
    @printf "%-24s  %12.4f  %14.4f\n" "Overall (excl. prob 2)" nanmean(sips_no2) nanmean(abs_no2)
    @printf "%-24s  %12.4f  %14.4f\n" "Optimal (prob 1)"       nanmean(sips_opt) nanmean(abs_opt)
    @printf "%-24s  %12.4f  %14.4f\n" "Sub-optimal (3 & 4)"    nanmean(sips_sub) nanmean(abs_sub)
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
        sips_no2 = [r.sips_pcc for r in results if r.problem != 2]
        abs_no2  = [r.abs_pcc  for r in results if r.problem != 2]

        println(io, "\ncategory,sips_pcc,abstract_pcc")
        @printf io "overall,%.6f,%.6f\n"              nanmean(sips_all) nanmean(abs_all)
        @printf io "optimal,%.6f,%.6f\n"              nanmean(sips_opt) nanmean(abs_opt)
        @printf io "suboptimal,%.6f,%.6f\n"           nanmean(sips_sub) nanmean(abs_sub)
        @printf io "overall_excl_prob2,%.6f,%.6f\n"   nanmean(sips_no2) nanmean(abs_no2)
    end
    println("Saved → $out")
end

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

results = evaluate_all()
print_table(results)
save_csv(results)
