"""
compute_pcc_by_timeframe.jl  –  Per-timeframe PCC breakdown.

For each judgement-point index k (1st, 2nd, 3rd, …) pools the k-th
survey point across all experiments that have at least k judgement
points, then computes PCC(model, human) using those 3*N data points
(3 gems × N experiments).

Also prints, for each individual experiment, the PCC at every one of
its own judgement-point timeframes.

Run from the repository root:
    julia --project=. evaluation/compute_pcc_by_timeframe.jl
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

const JUDGEMENT_POINTS = [
    [1, 7, 17, 23],          # 1_1
    [1, 9, 14, 17],          # 1_2
    [1, 9, 17, 24],          # 1_3
    [1, 7, 14, 23, 32],      # 1_4
    [1, 6, 11, 24],          # 2_1
    [1, 4, 6, 11],           # 2_2
    [1, 5, 8, 13],           # 2_3
    [1, 9, 12, 31, 44],      # 2_4
    [1, 7, 22, 37, 50],      # 3_1
    [1, 14, 24, 29, 40, 54], # 3_2
    [1, 7, 13, 20, 26],      # 3_3
    [1, 6, 11, 26, 36, 49],  # 3_4
    [1, 8, 14, 20],          # 4_1
    [1, 4, 7, 10],           # 4_2
    [1, 5, 8, 10],           # 4_3
    [1, 7, 12, 18],          # 4_4
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
    if n_available < n_times
        n_times = n_available
    end
    trimmed = human_data_1d[1 : n_times * n_goals]
    return reshape(trimmed, n_goals, n_times)   # (n_goals × n_times)
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

nanfmt(v) = isnan(v) ? "     N/A" : @sprintf("%8.4f", v)

# ─────────────────────────────────────────────────────────────
# Collect data
# ─────────────────────────────────────────────────────────────

struct ExpData
    exp_id    :: String
    problem   :: Int
    set       :: Int
    times     :: Vector{Int}       # judgement-point action steps
    sips_mat  :: Union{Matrix{Float64}, Nothing}
    abs_mat   :: Union{Matrix{Float64}, Nothing}
    human_mat :: Matrix{Float64}   # (n_goals × n_times_available)
end

all_exp = ExpData[]

for problem in PROBLEMS, s in SETS
    exp_id     = "$(problem)_$(s)"
    human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    !isfile(human_path) && continue

    sips_path = joinpath(SIPS_DIR, "goal_probs_SIPS_$(exp_id).csv")
    abs_path  = joinpath(ABS_DIR,  "goal_probs_hierarchical_$(exp_id).csv")

    times     = JUDGEMENT_POINTS[(problem - 1) * length(SETS) + s]
    human_mat = load_human(human_path, length(times))

    sips_mat = isfile(sips_path) ? load_model(sips_path) : nothing
    abs_mat  = isfile(abs_path)  ? load_model(abs_path)  : nothing

    push!(all_exp, ExpData(exp_id, problem, s, times, sips_mat, abs_mat, human_mat))
end

# ─────────────────────────────────────────────────────────────
# Part 1: PCC per experiment × per timeframe
# ─────────────────────────────────────────────────────────────

println("\n" * "═"^70)
println("  PCC per experiment × judgement-point timeframe")
println("  (each cell uses 3 data points — all 3 gems at that timestep)")
println("═"^70)
@printf "%-8s  %-6s  %-10s  %8s  %8s\n" "Exp" "tf idx" "action_t" "SIPS" "Abstract"
println("─"^70)

for ed in all_exp
    n_t = size(ed.human_mat, 2)
    for k in 1:n_t
        t = ed.times[k]

        sips_r = if ed.sips_mat !== nothing && t <= size(ed.sips_mat, 2)
            pearson_r(vec(ed.sips_mat[:, t]), vec(ed.human_mat[:, k]))
        else
            NaN
        end

        abs_r = if ed.abs_mat !== nothing && t <= size(ed.abs_mat, 2)
            pearson_r(vec(ed.abs_mat[:, t]), vec(ed.human_mat[:, k]))
        else
            NaN
        end

        @printf "%-8s  %6d  %10d  %s  %s\n" ed.exp_id k t nanfmt(sips_r) nanfmt(abs_r)
    end
    println("─"^70)
end

# ─────────────────────────────────────────────────────────────
# Part 2: Aggregate PCC pooled across experiments at each timeframe index
# ─────────────────────────────────────────────────────────────

max_tf = maximum(length(ed.times) for ed in all_exp)

# For each timeframe index k, accumulate (model, human) vectors across experiments
sips_pool = [Float64[] for _ in 1:max_tf]
abs_pool  = [Float64[] for _ in 1:max_tf]
hum_pool  = [Float64[] for _ in 1:max_tf]

for ed in all_exp
    n_t = size(ed.human_mat, 2)
    for k in 1:n_t
        t = ed.times[k]

        hv = vec(ed.human_mat[:, k])

        if ed.sips_mat !== nothing && t <= size(ed.sips_mat, 2)
            append!(sips_pool[k], vec(ed.sips_mat[:, t]))
            append!(hum_pool[k],  hv)   # parallel append for sips
        end
        if ed.abs_mat !== nothing && t <= size(ed.abs_mat, 2)
            append!(abs_pool[k], vec(ed.abs_mat[:, t]))
        end
    end
end

println("\n" * "═"^62)
println("  Pooled PCC at each timeframe index (across all experiments)")
println("  (N data points = 3 gems × number of experiments at that index)")
println("═"^62)
@printf "%-10s  %10s  %8s  %8s\n" "tf_index" "N (exps×3)" "SIPS_r" "Abstract_r"
println("─"^62)

for k in 1:max_tf
    n_pts   = length(hum_pool[k])
    sips_r  = n_pts >= 6 ? pearson_r(sips_pool[k], hum_pool[k]) : NaN

    # abs pool may differ if some files missing — use its own parallel human vec
    abs_hum = Float64[]
    for ed in all_exp
        k > length(ed.times) && continue
        t = ed.times[k]
        if ed.abs_mat !== nothing && t <= size(ed.abs_mat, 2)
            append!(abs_hum, vec(ed.human_mat[:, k]))
        end
    end
    abs_r = length(abs_hum) >= 6 ? pearson_r(abs_pool[k], abs_hum) : NaN

    n_exp = div(n_pts, N_GOALS)
    @printf "%-10d  %10d  %s  %s\n" k n_pts nanfmt(sips_r) nanfmt(abs_r)
end

println("═"^62, "\n")
