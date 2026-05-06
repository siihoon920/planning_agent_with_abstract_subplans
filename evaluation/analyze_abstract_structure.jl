"""
analyze_abstract_structure.jl

For each experiment in problem sets 1 and 2, measures goal ambiguity in the human
judgment data and correlates it with the Abstract − SIPS PCC gap.

Two ambiguity metrics:
  mean_entropy   : mean Shannon entropy of human goal probs across all JPs (excl. t=0)
  disambig_step  : action step at which the leading goal first exceeds 0.7 probability
                   (= plan length if never reached within the trajectory)

Hypothesis: longer ambiguity → SIPS particle filter degenerates before the diagnostic
signal arrives → larger Abstract − SIPS gap.

Run from the repository root:
    julia --project=. evaluation/analyze_abstract_structure.jl
"""

using Statistics, DelimitedFiles, Printf
using PyCall
pyimport("matplotlib").use("Agg")
using PyPlot

# ─────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────

const REPO_ROOT  = dirname(@__DIR__)
const HUMAN_DIR  = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")
const PLANS_DIR  = joinpath(REPO_ROOT, "domains", "doors-keys-gems", "plans")
const PCC_CSV    = joinpath(@__DIR__, "pcc_results.csv")

const N_GOALS = 3

const EXP_IDS = ["1_1","1_2","1_3","1_4",
                  "2_1","2_2","2_3","2_4"]

const JUDGEMENT_POINTS = [
    [7, 17, 23],       # 1_1
    [9, 14, 17],       # 1_2
    [9, 17, 24],       # 1_3
    [7, 14, 23, 32],   # 1_4
    [6, 11, 24],       # 2_1
    [4, 6, 11],        # 2_2
    [5, 8, 13],        # 2_3
    [9, 12, 31, 44],   # 2_4
]

const DISAMBIG_THRESHOLD = 0.7

# ─────────────────────────────────────────────────────────────
# Ambiguity metrics from human data
# ─────────────────────────────────────────────────────────────

function shannon_entropy(probs::AbstractVector{Float64})
    return -sum(p > 0.0 ? p * log(p) : 0.0 for p in probs)
end

function ambiguity_metrics(exp_id::String, jp_times::Vector{Int})
    path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    data = vec(readdlm(path, ',', Float64))
    n_avail = div(length(data), N_GOALS)

    # Load JP=1 (t=0) + actual JPs
    n_use    = min(length(jp_times) + 1, n_avail)
    mat      = reshape(data[1 : n_use * N_GOALS], N_GOALS, n_use)  # (goals × n_use)
    jp_cols  = mat[:, 2:end]                                        # skip JP=1 column
    jp_steps = jp_times[1 : size(jp_cols, 2)]

    # Mean Shannon entropy across actual JPs (excl. t=0)
    entropies = [shannon_entropy(jp_cols[:, i]) for i in 1:size(jp_cols, 2)]
    mean_H    = mean(entropies)

    # First JP where one goal exceeds threshold
    disambig = jp_steps[end]
    for (i, step) in enumerate(jp_steps)
        if maximum(jp_cols[:, i]) > DISAMBIG_THRESHOLD
            disambig = step
            break
        end
    end

    return mean_H, disambig
end

# ─────────────────────────────────────────────────────────────
# Plan length per experiment
# ─────────────────────────────────────────────────────────────

function plan_length(exp_id::String)
    matched = filter(f -> startswith(f, "$(exp_id)_"), readdir(PLANS_DIR))
    isempty(matched) && error("No plan file for $exp_id")
    lines = filter(l -> !isempty(strip(l)), readlines(joinpath(PLANS_DIR, matched[1])))
    return length(lines)
end

# ─────────────────────────────────────────────────────────────
# Load PCC results
# ─────────────────────────────────────────────────────────────

function load_pcc(path::String)
    raw = readdlm(path, ',', String)
    pcc = Dict{String,Tuple{Float64,Float64}}()
    for row in eachrow(raw)
        exp_id = strip(row[1])
        match(r"^\d_\d$", exp_id) === nothing && continue
        s = tryparse(Float64, strip(row[4]))
        a = tryparse(Float64, strip(row[5]))
        (isnothing(s) || isnothing(a)) && continue
        pcc[exp_id] = (s, a)
    end
    return pcc
end

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

struct ExpResult
    exp_id       :: String
    plan_len     :: Int
    mean_entropy :: Float64
    disambig     :: Int
    sips_pcc     :: Float64
    abs_pcc      :: Float64
    pcc_diff     :: Float64
end

pcc_data = load_pcc(PCC_CSV)
results  = ExpResult[]

for (idx, exp_id) in enumerate(EXP_IDS)
    jp_times          = JUDGEMENT_POINTS[idx]
    mean_H, disambig  = ambiguity_metrics(exp_id, jp_times)
    plen              = plan_length(exp_id)
    sips_pcc, abs_pcc = get(pcc_data, exp_id, (NaN, NaN))
    pcc_diff          = abs_pcc - sips_pcc
    push!(results, ExpResult(exp_id, plen, mean_H, disambig, sips_pcc, abs_pcc, pcc_diff))
end

# ─────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────

sep = "─" ^ 82
println("\n", sep)
@printf "%-6s  %-9s  %-13s  %-12s  %-10s  %-12s  %-9s\n" "Exp" "PlanLen" "MeanEntropy" "DisambigStep" "SIPS_PCC" "Abstract_PCC" "PCC_Diff"
println(sep)
for r in results
    @printf "%-6s  %-9d  %-13.4f  %-12d  %-10.4f  %-12.4f  %-9.4f\n" r.exp_id r.plan_len r.mean_entropy r.disambig r.sips_pcc r.abs_pcc r.pcc_diff
end
println(sep)

# ─────────────────────────────────────────────────────────────
# Correlations
# ─────────────────────────────────────────────────────────────

mean_Hs   = [r.mean_entropy for r in results]
disambigs = Float64[r.disambig for r in results]
pcc_diffs = [r.pcc_diff     for r in results]

println()
@printf "cor(mean_entropy,  pcc_diff) = %7.4f  (n=%d, threshold=%.1f)\n" cor(mean_Hs,   pcc_diffs) length(results) DISAMBIG_THRESHOLD
@printf "cor(disambig_step, pcc_diff) = %7.4f  (n=%d, threshold=%.1f)\n" cor(disambigs, pcc_diffs) length(results) DISAMBIG_THRESHOLD

# ─────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────

out_csv = joinpath(@__DIR__, "abstract_structure_analysis.csv")
open(out_csv, "w") do io
    println(io, "exp_id,plan_len,mean_entropy,disambig_step,sips_pcc,abstract_pcc,pcc_diff")
    for r in results
        @printf io "%s,%d,%.6f,%d,%.6f,%.6f,%.6f\n" r.exp_id r.plan_len r.mean_entropy r.disambig r.sips_pcc r.abs_pcc r.pcc_diff
    end
end
println("\nSaved → $out_csv")

# ─────────────────────────────────────────────────────────────
# Scatter plot: mean_entropy vs PCC
# ─────────────────────────────────────────────────────────────

entropies_plot = [r.mean_entropy for r in results]
sips_pccs      = [r.sips_pcc     for r in results]
abs_pccs       = [r.abs_pcc      for r in results]
exp_labels     = [r.exp_id       for r in results]

fig, ax = plt.subplots(figsize=(6.5, 5.5))

ax.scatter(entropies_plot, sips_pccs,
    color  = "steelblue",
    s      = 60,
    zorder = 3,
    label  = @sprintf("SIPS  (r = %.3f)", cor(entropies_plot, sips_pccs)),
)
ax.scatter(entropies_plot, abs_pccs,
    color  = "tomato",
    s      = 60,
    zorder = 3,
    label  = @sprintf("Abstract  (r = %.3f)", cor(entropies_plot, abs_pccs)),
)

# Label each point with exp_id
for (i, label) in enumerate(exp_labels)
    ax.annotate(label,
        xy        = (entropies_plot[i], sips_pccs[i]),
        xytext    = (3, 4),
        textcoords= "offset points",
        fontsize  = 8,
        color     = "steelblue",
    )
    ax.annotate(label,
        xy        = (entropies_plot[i], abs_pccs[i]),
        xytext    = (3, -10),
        textcoords= "offset points",
        fontsize  = 8,
        color     = "tomato",
    )
end

# Trend lines
for (ys, color) in [(sips_pccs, "steelblue"), (abs_pccs, "tomato")]
    xs_arr = entropies_plot
    m = cor(xs_arr, ys) * std(ys) / std(xs_arr)
    b = mean(ys) - m * mean(xs_arr)
    x_line = [minimum(xs_arr) - 0.02, maximum(xs_arr) + 0.02]
    ax.plot(x_line, m .* x_line .+ b,
        color     = color,
        alpha     = 0.4,
        linewidth = 1.2,
        linestyle = "--",
    )
end

ax.axhline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.7)
ax.set_xlabel("Mean entropy of human goal probabilities", fontsize=12)
ax.set_ylabel("PCC (model vs human)", fontsize=12)
ax.set_title("Goal ambiguity vs model accuracy\n(problem sets 1 & 2)", fontsize=12)
ax.legend(loc="upper right", frameon=true)

plot_path = joinpath(@__DIR__, "entropy_vs_pcc.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
println("Saved → $plot_path")
