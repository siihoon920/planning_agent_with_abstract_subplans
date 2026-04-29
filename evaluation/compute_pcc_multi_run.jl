"""
compute_pcc_multi_run.jl  –  PCC evaluation across N independent runs.

Expects CSVs named:
  goal_probs_SIPS_<exp_id>_run_<k>.csv
  goal_probs_hierarchical_<exp_id>_run_<k>.csv

For each experiment, computes PCC(model, human) per run, then reports
mean ± std across runs in every table row.

Run from the repository root:
    julia --project=. evaluation/compute_pcc_multi_run.jl
"""

using DelimitedFiles, Statistics, Printf

# ─────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────

const REPO_ROOT = dirname(@__DIR__)
const SIPS_DIR  = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_SIPS")
const ABS_DIR   = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_hierarchical")
const HUMAN_DIR = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")

const N_RUNS   = 5
const N_GOALS  = 3
const PROBLEMS = 1:4
const SETS     = 1:4

const OPTIMAL_PROBLEMS    = Set([1])
const SUBOPTIMAL_PROBLEMS = Set([2, 3, 4])

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
# Loaders
# ─────────────────────────────────────────────────────────────

function load_model(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])
    return collect(transpose(data))   # (n_goals × T)
end

function load_human(path::String, n_times::Int, n_goals::Int = N_GOALS)
    human_data_1d = vec(readdlm(path, ',', Float64))
    n_available   = div(length(human_data_1d), n_goals)
    n_times       = min(n_times, n_available)
    return reshape(human_data_1d[1 : n_times * n_goals], n_goals, n_times)
end

# ─────────────────────────────────────────────────────────────
# PCC
# ─────────────────────────────────────────────────────────────

function pearson_r(x::AbstractVector, y::AbstractVector)
    (length(x) < 2 || std(x) ≈ 0 || std(y) ≈ 0) && return NaN
    x̄, ȳ = mean(x), mean(y)
    num  = sum((x .- x̄) .* (y .- ȳ))
    den  = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

function pcc_at_times(model::Matrix, human::Matrix, times::Vector{Int})
    n_times    = size(human, 2)
    shifted    = times[1:n_times] .- 1   # JPs are 1-indexed from t=0; model CSV starts at t=1
    valid_mask = [1 <= t <= size(model, 2) for t in shifted]
    valid_times = shifted[valid_mask]
    valid_cols  = (1:n_times)[valid_mask]
    length(valid_times) < 2 && return NaN
    return pearson_r(vec(model[:, valid_times]), vec(human[:, valid_cols]))
end

nanmean(v) = (f = filter(!isnan, v); isempty(f) ? NaN : mean(f))
nanstd(v)  = (f = filter(!isnan, v); length(f) < 2 ? NaN : std(f))

# ─────────────────────────────────────────────────────────────
# Collect per-run PCCs for all experiments
# ─────────────────────────────────────────────────────────────

struct RunResult
    exp_id   :: String
    problem  :: Int
    set      :: Int
    sips_r   :: Vector{Float64}   # length N_RUNS (NaN if run missing)
    abs_r    :: Vector{Float64}
end

results = RunResult[]

for problem in PROBLEMS, s in SETS
    exp_id     = "$(problem)_$(s)"
    human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    !isfile(human_path) && continue

    times     = JUDGEMENT_POINTS[(problem - 1) * length(SETS) + s]
    human_mat = load_human(human_path, length(times) + 1)[:, 2:end]

    sips_rs = Float64[]
    abs_rs  = Float64[]

    for run_id in 1:N_RUNS
        sips_path = joinpath(SIPS_DIR,
            "goal_probs_SIPS_$(exp_id)_run_$(run_id).csv")
        abs_path  = joinpath(ABS_DIR,
            "goal_probs_hierarchical_$(exp_id)_run_$(run_id).csv")

        push!(sips_rs, isfile(sips_path) ?
            pcc_at_times(load_model(sips_path), human_mat, times) : NaN)
        push!(abs_rs,  isfile(abs_path)  ?
            pcc_at_times(load_model(abs_path),  human_mat, times) : NaN)
    end

    push!(results, RunResult(exp_id, problem, s, sips_rs, abs_rs))
end

# ─────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────

fmt(μ, σ) = isnan(μ) ? "       N/A        " : @sprintf("%6.4f ± %6.4f", μ, σ)

# ─────────────────────────────────────────────────────────────
# Per-experiment table
# ─────────────────────────────────────────────────────────────

sep1 = "─" ^ 66
println("\n", sep1)
@printf "%-8s  %-20s  %-20s\n" "Exp" "SIPS PCC (mean±std)" "Abstract PCC (mean±std)"
println(sep1)
for r in results
    μs, σs = nanmean(r.sips_r), nanstd(r.sips_r)
    μa, σa = nanmean(r.abs_r),  nanstd(r.abs_r)
    @printf "%-8s  %-20s  %-20s\n" r.exp_id fmt(μs,σs) fmt(μa,σa)
end
println(sep1)

# ─────────────────────────────────────────────────────────────
# Category summary table
# ─────────────────────────────────────────────────────────────

function cat_stats(rs::Vector{RunResult}, pred)
    keep = filter(pred, rs)
    isempty(keep) && return NaN, NaN, NaN, NaN
    # Pool all per-run values for each selected experiment, then summarise
    sips_per_run = [nanmean([r.sips_r[k] for r in keep]) for k in 1:N_RUNS]
    abs_per_run  = [nanmean([r.abs_r[k]  for r in keep]) for k in 1:N_RUNS]
    nanmean(sips_per_run), nanstd(sips_per_run),
    nanmean(abs_per_run),  nanstd(abs_per_run)
end

categories = [
    ("Overall",                r -> true),
    ("Optimal (prob 1)",       r -> r.problem == 1),
    ("Suboptimal (2, 3, 4)",   r -> r.problem in SUBOPTIMAL_PROBLEMS),
    ("  Action mistakes (2)",  r -> r.problem == 2),
    ("  Plan mistakes (3)",    r -> r.problem == 3),
    ("  Short-sighted (4)",    r -> r.problem == 4),
]

sep2 = "─" ^ 70
println()
println(sep2)
@printf "%-22s  %-20s  %-20s\n" "Category" "SIPS PCC (mean±std)" "Abstract PCC (mean±std)"
println(sep2)
for (label, pred) in categories
    μs, σs, μa, σa = cat_stats(results, pred)
    @printf "%-22s  %-20s  %-20s\n" label fmt(μs,σs) fmt(μa,σa)
end
println(sep2)

# ─────────────────────────────────────────────────────────────
# Per-run breakdown (so you can see variance)
# ─────────────────────────────────────────────────────────────

println()
sep3 = "─" ^ 56
println(sep3)
@printf "%-8s  %s\n" "Run" join([@sprintf("%6s", "SIPS") * "  " * @sprintf("%6s","Abs") for _ in 1:1], "  ")
println("  (per-run overall mean PCC across all experiments)")
println(sep3)
for k in 1:N_RUNS
    sips_k = nanmean([r.sips_r[k] for r in results])
    abs_k  = nanmean([r.abs_r[k]  for r in results])
    @printf "run %-4d  SIPS=%7.4f  Abstract=%7.4f\n" k sips_k abs_k
end
println(sep3)

# ─────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────

out_csv = joinpath(@__DIR__, "pcc_results_multi_run.csv")
open(out_csv, "w") do io
    println(io, "exp_id,problem,set,sips_mean,sips_std,abstract_mean,abstract_std")
    for r in results
        @printf io "%s,%d,%d,%.6f,%.6f,%.6f,%.6f\n" r.exp_id r.problem r.set \
            nanmean(r.sips_r) nanstd(r.sips_r) nanmean(r.abs_r) nanstd(r.abs_r)
    end

    println(io, "\ncategory,sips_mean,sips_std,abstract_mean,abstract_std")
    for (label, pred) in categories
        μs, σs, μa, σa = cat_stats(results, pred)
        @printf io "%s,%.6f,%.6f,%.6f,%.6f\n" label μs σs μa σa
    end

    println(io, "\nrun,sips_overall_mean,abstract_overall_mean")
    for k in 1:N_RUNS
        sips_k = nanmean([r.sips_r[k] for r in results])
        abs_k  = nanmean([r.abs_r[k]  for r in results])
        @printf io "%d,%.6f,%.6f\n" k sips_k abs_k
    end
end
println("\nSaved → $out_csv")
