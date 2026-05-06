"""
compute_pcc_budget001.jl  –  Same as compute_pcc.jl but reads SIPS results
                              from goal_probs_SIPS_budget001/ (budget_p = 0.01).

Run from the repository root:
    julia --project=. evaluation/compute_pcc_budget001.jl
"""

using DelimitedFiles, Statistics, Printf

REPO_ROOT = dirname(@__DIR__)
SIPS_DIR  = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_SIPS_budget001")
ABS_DIR   = joinpath(REPO_ROOT, "example", "doors-keys-gems", "goal_probs_hierarchical")
HUMAN_DIR = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")

N_GOALS  = 3
PROBLEMS = 1:4
SETS     = 1:4

OPTIMAL_PROBLEMS    = Set([1])
SUBOPTIMAL_PROBLEMS = Set([2, 3, 4])

JUDGEMENT_POINTS = [
    [7, 17, 23],           # 1_1
    [9, 14, 17],           # 1_2
    [9, 17, 24],           # 1_3
    [7, 14, 23, 32],       # 1_4
    [6, 11, 24],           # 2_1
    [4, 6, 11],            # 2_2
    [5, 8, 13],            # 2_3
    [9, 12, 31, 44],       # 2_4
    [7, 22, 37, 50],       # 3_1
    [14, 24, 29, 40, 54],  # 3_2
    [7, 13, 20, 26],       # 3_3
    [6, 11, 26, 36, 49],   # 3_4
    [8, 14, 20],           # 4_1
    [4, 7, 10],            # 4_2
    [5, 8, 10],            # 4_3
    [7, 12, 18],           # 4_4
]

function load_model(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])
    return collect(transpose(data))
end

function load_human(path::String, n_times::Int, n_goals::Int = N_GOALS)
    human_data_1d = vec(readdlm(path, ',', Float64))
    n_available   = div(length(human_data_1d), n_goals)
    if n_available < n_times
        @warn "Human CSV has $n_available timesteps but expected $n_times; using $n_available"
        n_times = n_available
    end
    trimmed = human_data_1d[1 : n_times * n_goals]
    return reshape(trimmed, n_goals, n_times)
end

function pearson_r(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    x̄, ȳ = mean(x), mean(y)
    num  = sum((x .- x̄) .* (y .- ȳ))
    den  = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

function pcc_at_times(model::Matrix, human::Matrix, times::Vector{Int}, exp_id::String)
    n_times     = size(human, 2)
    shifted     = times[1:n_times] .- 1
    valid_mask  = [1 <= t <= size(model, 2) for t in shifted]
    n_dropped   = count(!, valid_mask)
    n_dropped > 0 && @warn "[$exp_id] $n_dropped time(s) out of model range, dropped"
    valid_times = shifted[valid_mask]
    valid_cols  = (1:n_times)[valid_mask]
    length(valid_times) < 2 && return NaN
    return pearson_r(vec(model[:, valid_times]), vec(human[:, valid_cols]))
end

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
        sips_path  = joinpath(SIPS_DIR,  "goal_probs_SIPS_budget001_$(exp_id).csv")
        abs_path   = joinpath(ABS_DIR,   "goal_probs_hierarchical_$(exp_id).csv")

        !isfile(human_path) && (@warn "[$exp_id] Human data missing, skipping."; continue)

        times     = JUDGEMENT_POINTS[(problem - 1) * length(SETS) + s]
        human_mat = load_human(human_path, length(times) + 1)[:, 2:end]

        sips_r = isfile(sips_path) ?
            pcc_at_times(load_model(sips_path), human_mat, times, exp_id) :
            (@warn("[$exp_id] SIPS budget001 CSV missing."); NaN)

        abs_r = isfile(abs_path) ?
            pcc_at_times(load_model(abs_path), human_mat, times, exp_id) :
            (@warn("[$exp_id] Abstract CSV missing."); NaN)

        push!(results, Result(exp_id, problem, s, sips_r, abs_r))
    end
    return results
end

nanmean(vals) = begin v = filter(!isnan, vals); isempty(v) ? NaN : mean(v) end

function print_table(results::Vector{Result})
    sep = "─" ^ 44
    println("\n", sep)
    @printf "%-8s  %12s  %14s\n" "Exp" "SIPS(p=0.01)" "Abstract PCC"
    println(sep)
    for r in results
        @printf "%-8s  %12.4f  %14.4f\n" r.exp_id r.sips_pcc r.abs_pcc
    end
    println(sep)

    sips_all = [r.sips_pcc for r in results]
    abs_all  = [r.abs_pcc  for r in results]
    sips_opt = [r.sips_pcc for r in results if r.problem == 1]
    abs_opt  = [r.abs_pcc  for r in results if r.problem == 1]
    sips_sub = [r.sips_pcc for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
    abs_sub  = [r.abs_pcc  for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
    sips_p2  = [r.sips_pcc for r in results if r.problem == 2]
    abs_p2   = [r.abs_pcc  for r in results if r.problem == 2]
    sips_p3  = [r.sips_pcc for r in results if r.problem == 3]
    abs_p3   = [r.abs_pcc  for r in results if r.problem == 3]
    sips_p4  = [r.sips_pcc for r in results if r.problem == 4]
    abs_p4   = [r.abs_pcc  for r in results if r.problem == 4]

    println()
    sep2 = "─" ^ 54
    println(sep2)
    @printf "%-28s  %8s  %12s\n" "Category" "SIPS(p=0.01)" "Abstract PCC"
    println(sep2)
    @printf "%-28s  %8.4f  %12.4f\n" "Overall"                  nanmean(sips_all) nanmean(abs_all)
    @printf "%-28s  %8.4f  %12.4f\n" "Optimal (prob 1)"         nanmean(sips_opt) nanmean(abs_opt)
    @printf "%-28s  %8.4f  %12.4f\n" "Suboptimal (2, 3, 4)"     nanmean(sips_sub) nanmean(abs_sub)
    @printf "%-28s  %8.4f  %12.4f\n" "  Action mistakes (2)"    nanmean(sips_p2)  nanmean(abs_p2)
    @printf "%-28s  %8.4f  %12.4f\n" "  Plan mistakes (3)"      nanmean(sips_p3)  nanmean(abs_p3)
    @printf "%-28s  %8.4f  %12.4f\n" "  Short-sighted plan (4)" nanmean(sips_p4)  nanmean(abs_p4)
    println(sep2, "\n")
end

function save_csv(results::Vector{Result})
    out = joinpath(@__DIR__, "pcc_results_budget001.csv")
    open(out, "w") do io
        println(io, "exp_id,problem,set,sips_pcc,abstract_pcc")
        for r in results
            @printf io "%s,%d,%d,%.6f,%.6f\n" r.exp_id r.problem r.set r.sips_pcc r.abs_pcc
        end

        sips_all = [r.sips_pcc for r in results]
        abs_all  = [r.abs_pcc  for r in results]
        sips_opt = [r.sips_pcc for r in results if r.problem == 1]
        abs_opt  = [r.abs_pcc  for r in results if r.problem == 1]
        sips_sub = [r.sips_pcc for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
        abs_sub  = [r.abs_pcc  for r in results if r.problem in SUBOPTIMAL_PROBLEMS]
        sips_p2  = [r.sips_pcc for r in results if r.problem == 2]
        abs_p2   = [r.abs_pcc  for r in results if r.problem == 2]
        sips_p3  = [r.sips_pcc for r in results if r.problem == 3]
        abs_p3   = [r.abs_pcc  for r in results if r.problem == 3]
        sips_p4  = [r.sips_pcc for r in results if r.problem == 4]
        abs_p4   = [r.abs_pcc  for r in results if r.problem == 4]

        println(io, "\ncategory,sips_pcc,abstract_pcc")
        @printf io "overall,%.6f,%.6f\n"             nanmean(sips_all) nanmean(abs_all)
        @printf io "optimal,%.6f,%.6f\n"             nanmean(sips_opt) nanmean(abs_opt)
        @printf io "suboptimal,%.6f,%.6f\n"          nanmean(sips_sub) nanmean(abs_sub)
        @printf io "action_mistakes,%.6f,%.6f\n"     nanmean(sips_p2)  nanmean(abs_p2)
        @printf io "plan_mistakes,%.6f,%.6f\n"       nanmean(sips_p3)  nanmean(abs_p3)
        @printf io "short_sighted_plan,%.6f,%.6f\n"  nanmean(sips_p4)  nanmean(abs_p4)
    end
    println("Saved → $out")
end

results = evaluate_all()
print_table(results)
save_csv(results)
