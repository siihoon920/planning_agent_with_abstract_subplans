"""
evaluate.jl  –  Compare the SIPS model and the Abstract (Hierarchical) Planner
               against ground-truth human goal-inference data.

Usage
-----
    # Compare both models for one experiment (doors-keys-gems)
    julia examples/evaluate.jl doors-keys-gems 1_1

    # Evaluate all 16 experiments in doors-keys-gems
    julia examples/evaluate.jl doors-keys-gems all

    # Evaluate all 16 experiments for block-words (SIPS model only)
    julia examples/evaluate.jl block-words all

The script expects that you have already run the corresponding human_example.jl
(and human_example_abstract.jl for the abstract model) so that the cached model
CSV files exist in the examples/<domain>/ folder.

Output
------
Prints a formatted table to stdout and writes:
    examples/evaluation_results_<domain>_<exp_id>.csv
"""

using DelimitedFiles, Printf, Statistics

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Pearson correlation between two equal-length vectors."""
function pearson_r(x::AbstractVector, y::AbstractVector)
    n = length(x)
    @assert n == length(y) "Length mismatch: $(n) vs $(length(y))"
    x̄, ȳ = mean(x), mean(y)
    num = sum((x .- x̄) .* (y .- ȳ))
    den = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

"""Mean Squared Error between two equal-length vectors."""
function mse(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "Length mismatch"
    return mean((x .- y).^2)
end

"""
Cross-entropy of human distribution q under model distribution p.
H(q, p) = -Σ q_i * log(p_i + ε).
Smaller is better. ε avoids log(0).
"""
function cross_entropy(q::AbstractVector, p::AbstractVector; ε=1e-9)
    @assert length(q) == length(p) "Length mismatch"
    return -sum(q .* log.(p .+ ε))
end

"""
Load a model CSV file saved by human_example.jl / human_example_abstract.jl.
Returns a (n_goals × T) Float64 matrix, or nothing if file not found.
"""
function load_model_csv(path::String)
    if !isfile(path)
        return nothing, nothing
    end
    raw = readdlm(path, ',', String)
    header = raw[1, :]   # goal names in first row
    data   = parse.(Float64, raw[2:end, :])   # (T × n_goals)
    return transpose(data), header    # return (n_goals × T)
end

"""
Load human CSV data.
Returns a (n_goals × T) Float64 matrix.
"""
function load_human_csv(path::String, n_goals::Int)
    if !isfile(path)
        error("Human data CSV not found: $path")
    end
    raw = vec(readdlm(path, ',', Float64))
    T   = div(length(raw), n_goals)
    return reshape(raw, n_goals, T)
end

"""
Compute per-goal and overall metrics between model and human matrices.
Both matrices are (n_goals × T).
Returns a NamedTuple with vectors of per-goal values + overall scalars.
"""
function compute_metrics(model::Matrix, human::Matrix)
    @assert size(model) == size(human) "Shape mismatch: $(size(model)) vs $(size(human))"
    n_goals, T = size(model)

    per_goal_r   = zeros(n_goals)
    per_goal_mse = zeros(n_goals)
    per_goal_ce  = zeros(n_goals)

    for g in 1:n_goals
        per_goal_r[g]   = pearson_r(model[g, :], human[g, :])
        per_goal_mse[g] = mse(model[g, :],      human[g, :])
        per_goal_ce[g]  = cross_entropy(human[g, :], model[g, :])
    end

    # Flatten for overall metrics
    model_flat = vec(model)
    human_flat = vec(human)

    overall_r   = pearson_r(model_flat, human_flat)
    overall_mse = mse(model_flat, human_flat)
    overall_ce  = cross_entropy(human_flat, model_flat)

    return (
        per_goal_r   = per_goal_r,
        per_goal_mse = per_goal_mse,
        per_goal_ce  = per_goal_ce,
        overall_r    = overall_r,
        overall_mse  = overall_mse,
        overall_ce   = overall_ce,
    )
end

"""Align two matrices to the same number of columns (min of the two T values)."""
function align_timesteps(a::Matrix, b::Matrix)
    T = min(size(a, 2), size(b, 2))
    return a[:, 1:T], b[:, 1:T]
end

# ──────────────────────────────────────────────────────────────────────────────
# Directory resolution
# ──────────────────────────────────────────────────────────────────────────────

const REPO_ROOT = dirname(@__DIR__)   # directory containing examples/

function examples_dir(domain::String)
    return joinpath(REPO_ROOT, "examples", domain)
end

function human_data_dir(domain::String)
    return joinpath(REPO_ROOT, "domains", domain, "average_human_results_arrays")
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-domain metadata
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_GOALS = Dict(
    "doors-keys-gems" => ["(has gem1)", "(has gem2)", "(has gem3)"],
    "block-words"     => String[]  # block-words goals are experiment-specific; loaded from CSV header
)

ALL_EXPERIMENTS = [
    "1_1", "1_2", "1_3", "1_4",
    "2_1", "2_2", "2_3", "2_4",
    "3_1", "3_2", "3_3", "3_4",
    "4_1", "4_2", "4_3", "4_4"
]

# ──────────────────────────────────────────────────────────────────────────────
# Single-experiment evaluation
# ──────────────────────────────────────────────────────────────────────────────

"""
Evaluate one experiment for one domain.
Returns a Dict with result rows ready for printing / saving.
"""
function evaluate_experiment(domain::String, exp_id::String)
    ex_dir   = examples_dir(domain)
    hd_dir   = human_data_dir(domain)
    hd_csv   = joinpath(hd_dir, "$(exp_id).csv")

    # ---- SIPS model ----
    sips_csv = joinpath(ex_dir, "model_sips_goal_probs_$(exp_id).csv")
    sips_mat, sips_header = load_model_csv(sips_csv)

    # ---- Abstract (hierarchical) model — only doors-keys-gems ----
    abs_csv  = joinpath(ex_dir, "model_abstract_goal_probs_$(exp_id).csv")
    abs_mat, abs_header = load_model_csv(abs_csv)

    # Infer n_goals from whichever model was found
    n_goals = if !isnothing(sips_mat)
        size(sips_mat, 1)
    elseif !isnothing(abs_mat)
        size(abs_mat, 1)
    else
        domain == "doors-keys-gems" ? 3 : 5   # fallback defaults
    end

    # ---- Human data ----
    if !isfile(hd_csv)
        @warn "Human data not found for $(domain)/$(exp_id), skipping."
        return nothing
    end
    human_mat = load_human_csv(hd_csv, n_goals)

    results = Dict{String, Any}("exp_id" => exp_id, "domain" => domain)

    # ---- Evaluate SIPS ----
    if !isnothing(sips_mat)
        sm, hm = align_timesteps(sips_mat, human_mat)
        m = compute_metrics(sm, hm)
        results["sips_r"]   = m.overall_r
        results["sips_mse"] = m.overall_mse
        results["sips_ce"]  = m.overall_ce
        results["sips_per_goal_r"]   = m.per_goal_r
        results["sips_per_goal_mse"] = m.per_goal_mse
    else
        results["sips_r"]   = NaN
        results["sips_mse"] = NaN
        results["sips_ce"]  = NaN
        @warn "SIPS model output not found for $(domain)/$(exp_id). " *
              "Run: julia examples/$(domain)/human_example.jl $(exp_id)"
    end

    # ---- Evaluate Abstract planner ----
    if !isnothing(abs_mat)
        am, hm = align_timesteps(abs_mat, human_mat)
        m = compute_metrics(am, hm)
        results["abs_r"]   = m.overall_r
        results["abs_mse"] = m.overall_mse
        results["abs_ce"]  = m.overall_ce
        results["abs_per_goal_r"]   = m.per_goal_r
        results["abs_per_goal_mse"] = m.per_goal_mse
    elseif domain == "doors-keys-gems"
        results["abs_r"]   = NaN
        results["abs_mse"] = NaN
        results["abs_ce"]  = NaN
        @warn "Abstract model output not found for $(domain)/$(exp_id). " *
              "Run: julia examples/$(domain)/human_example_abstract.jl $(exp_id)"
    else
        results["abs_r"]   = missing
        results["abs_mse"] = missing
        results["abs_ce"]  = missing
    end

    return results
end

# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print a results table
# ──────────────────────────────────────────────────────────────────────────────

function print_results_table(all_results::Vector)
    valid = filter(!isnothing, all_results)
    isempty(valid) && (println("No results to display."); return)

    domain = valid[1]["domain"]
    has_abstract = domain == "doors-keys-gems"

    sep = "─" ^ (has_abstract ? 82 : 52)
    println("\n", sep)
    @printf "%-8s  %8s  %8s  %8s" "Exp" "SIPS-r" "SIPS-MSE" "SIPS-CE"
    if has_abstract
        @printf "  %8s  %8s  %8s" "Abs-r" "Abs-MSE" "Abs-CE"
    end
    println()
    println(sep)

    for r in valid
        exp = r["exp_id"]
        sips_r   = get(r, "sips_r",   NaN)
        sips_mse = get(r, "sips_mse", NaN)
        sips_ce  = get(r, "sips_ce",  NaN)
        @printf "%-8s  %8.4f  %8.4f  %8.4f" exp sips_r sips_mse sips_ce
        if has_abstract
            abs_r   = get(r, "abs_r",   NaN)
            abs_mse = get(r, "abs_mse", NaN)
            abs_ce  = get(r, "abs_ce",  NaN)
            @printf "  %8.4f  %8.4f  %8.4f" abs_r abs_mse abs_ce
        end
        println()
    end

    println(sep)

    # Summary row (mean ± std over experiments)
    function safe_mean(vec)
        v = filter(!isnan, replace(vec, missing => NaN) .|> Float64)
        isempty(v) ? NaN : mean(v)
    end
    function safe_std(vec)
        v = filter(!isnan, replace(vec, missing => NaN) .|> Float64)
        length(v) < 2 ? NaN : std(v)
    end

    sips_rs   = [get(r, "sips_r",   NaN) for r in valid]
    sips_mses = [get(r, "sips_mse", NaN) for r in valid]
    sips_ces  = [get(r, "sips_ce",  NaN) for r in valid]

    @printf "%-8s  %8.4f  %8.4f  %8.4f" "MEAN" safe_mean(sips_rs) safe_mean(sips_mses) safe_mean(sips_ces)
    if has_abstract
        abs_rs   = [get(r, "abs_r",   NaN) for r in valid]
        abs_mses = [get(r, "abs_mse", NaN) for r in valid]
        abs_ces  = [get(r, "abs_ce",  NaN) for r in valid]
        @printf "  %8.4f  %8.4f  %8.4f" safe_mean(abs_rs) safe_mean(abs_mses) safe_mean(abs_ces)
    end
    println()
    println(sep, "\n")

    if has_abstract
        @printf "Legend:\n"
        @printf "  r    = Pearson correlation with human data  (higher is better, max 1)\n"
        @printf "  MSE  = Mean Squared Error                   (lower  is better, min 0)\n"
        @printf "  CE   = Cross-Entropy of human under model   (lower  is better, min 0)\n"
        @printf "\n  SIPS = Original SIPS particle-filter model\n"
        @printf "  Abs  = Hierarchical Abstract Planner model\n\n"
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Save results to CSV
# ──────────────────────────────────────────────────────────────────────────────

function save_results_csv(all_results::Vector, domain::String, tag::String)
    valid = filter(!isnothing, all_results)
    isempty(valid) && return
    has_abstract = domain == "doors-keys-gems"

    out_path = joinpath(REPO_ROOT, "examples", "evaluation_results_$(domain)_$(tag).csv")
    open(out_path, "w") do io
        # Header
        header = "exp_id,sips_r,sips_mse,sips_ce"
        if has_abstract
            header *= ",abstract_r,abstract_mse,abstract_ce"
        end
        println(io, header)
        # Rows
        for r in valid
            line = join([
                r["exp_id"],
                get(r, "sips_r",   NaN),
                get(r, "sips_mse", NaN),
                get(r, "sips_ce",  NaN),
            ], ",")
            if has_abstract
                line *= "," * join([
                    get(r, "abs_r",   NaN),
                    get(r, "abs_mse", NaN),
                    get(r, "abs_ce",  NaN),
                ], ",")
            end
            println(io, line)
        end
    end
    println("Saved evaluation CSV → $out_path")
end

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 2
        println("""
Usage:
  julia examples/evaluate.jl <domain> <exp_id|all>

Arguments:
  domain   : "doors-keys-gems" or "block-words"
  exp_id   : experiment ID like "1_1", "2_3", etc., OR "all" to run all 16

Examples:
  julia examples/evaluate.jl doors-keys-gems 1_1
  julia examples/evaluate.jl doors-keys-gems all
  julia examples/evaluate.jl block-words     all
""")
        exit(1)
    end

    domain = ARGS[1]
    exp_arg = ARGS[2]

    if !(domain in ["doors-keys-gems", "block-words"])
        error("Unknown domain '$(domain)'. Choose 'doors-keys-gems' or 'block-words'.")
    end

    experiments = exp_arg == "all" ? ALL_EXPERIMENTS : [exp_arg]

    println("\n=== Evaluating $(domain) | experiments: $(join(experiments, ", ")) ===\n")

    all_results = [evaluate_experiment(domain, eid) for eid in experiments]

    print_results_table(all_results)
    tag = exp_arg == "all" ? "all" : exp_arg
    save_results_csv(all_results, domain, tag)
end

main()
