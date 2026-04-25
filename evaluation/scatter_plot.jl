"""
scatter_plot.jl  –  Scatter plot: model predictions (x) vs human inference (y).

Uses PyPlot (matplotlib) instead of GLMakie — no GPU/OpenGL required.

Run from the repository root:
    julia --project=. evaluation/scatter_plot.jl

Output: evaluation/scatter_plot.png
"""

using PyCall
pyimport("matplotlib").use("Agg")

using PyPlot

using DelimitedFiles, Statistics, Printf, PyPlot

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

# ─────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────

function load_stimuli_times(path::String)
    times_map = Dict{String, Vector{Int}}()
    text = read(path, String)
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

function load_model(path::String)
    raw  = readdlm(path, ',', String)
    data = parse.(Float64, raw[2:end, :])
    return collect(transpose(data))   # (n_goals × T)
end

function load_human(path::String, n_goals::Int = N_GOALS)
    human_data_1d    = vec(readdlm(path, ',', Float64))
    n_time_steps     = div(length(human_data_1d), n_goals)
    human_goal_probs = reshape(human_data_1d, n_goals, n_time_steps)
    return human_goal_probs
end

function pearson_r(x::AbstractVector, y::AbstractVector)
    (length(x) < 2 || std(x) ≈ 0 || std(y) ≈ 0) && return NaN
    x̄, ȳ = mean(x), mean(y)
    num  = sum((x .- x̄) .* (y .- ȳ))
    den  = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

function extract_pairs(model::Matrix, human::Matrix, times::Vector{Int})
    n_human = size(human, 2)
    valid   = filter(t -> 1 <= t <= size(model, 2), times)
    n       = min(length(valid), n_human)
    n < 1   && return Float64[], Float64[]

    model_subset = model[:, valid[1:n]]
    human_subset = human[:, 1:n]

    return vec(model_subset), vec(human_subset)
end

# ─────────────────────────────────────────────────────────────
# Collect scatter data across all 16 experiments
# ─────────────────────────────────────────────────────────────

stimuli_times = load_stimuli_times(STIMULI_PATH)

sips_x = Float64[];  sips_y = Float64[]
abs_x  = Float64[];  abs_y  = Float64[]

for problem in PROBLEMS, s in SETS
    exp_id     = "$(problem)_$(s)"
    human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    sips_path  = joinpath(SIPS_DIR,  "goal_probs_SIPS_$(exp_id).csv")
    abs_path   = joinpath(ABS_DIR,   "goal_probs_hierarchical_$(exp_id).csv")

    (!isfile(human_path) || !haskey(stimuli_times, exp_id)) && continue

    human_mat = load_human(human_path)
    times     = stimuli_times[exp_id]

    if isfile(sips_path)
        mx, hx = extract_pairs(load_model(sips_path), human_mat, times)
        append!(sips_x, mx);  append!(sips_y, hx)
    end

    if isfile(abs_path)
        mx, hx = extract_pairs(load_model(abs_path), human_mat, times)
        append!(abs_x, mx);  append!(abs_y, hx)
    end
end

# ─────────────────────────────────────────────────────────────
# Pearson r for legend labels
# ─────────────────────────────────────────────────────────────

sips_r = pearson_r(sips_x, sips_y)
abs_r  = pearson_r(abs_x,  abs_y)

@printf "SIPS PCC (all data):     %.4f\n" sips_r
@printf "Abstract PCC (all data): %.4f\n" abs_r

# ─────────────────────────────────────────────────────────────
# Plot with PyPlot (matplotlib)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 6.0))

ax.scatter(sips_x, sips_y;
    color  = "steelblue",
    alpha  = 0.35,
    s      = 20,
    label  = @sprintf("SIPS  (r = %.3f)", sips_r),
    zorder = 2,
)
ax.scatter(abs_x, abs_y;
    color  = "tomato",
    alpha  = 0.35,
    s      = 20,
    label  = @sprintf("Abstract  (r = %.3f)", abs_r),
    zorder = 2,
)

# Identity line y = x
ax.plot([0.0, 1.0], [0.0, 1.0];
    color     = "black",
    alpha     = 0.5,
    linewidth = 1.0,
    linestyle = "--",
    label     = "y = x",
    zorder    = 1,
)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")
ax.set_xlabel("Model predicted probability")
ax.set_ylabel("Human inferred probability")
ax.set_title("Model predictions vs Human goal inference\n(doors-keys-gems, all experiments)")
ax.legend(loc="upper left", frameon=true)

out_path = joinpath(@__DIR__, "scatter_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
println("Saved → $out_path")