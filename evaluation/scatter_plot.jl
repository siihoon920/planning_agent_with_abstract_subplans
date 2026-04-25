"""
scatter_plot.jl  –  Scatter plot: model predictions (x) vs human inference (y).

Uses PyPlot (matplotlib) — no GPU/OpenGL required.

Each dot is one (gem, survey-timestep, experiment) triple:
    x = model's predicted probability for that gem at that action step
    y = human's inferred probability for that gem at that survey point

Only the timesteps listed in stimuli/stimuli.json are used.
Total dots per model = sum over 16 experiments of len(times_i) * 3.

Run from the repository root:
    julia --project=. evaluation/scatter_plot.jl

Output: evaluation/scatter_plot.png
See evaluation/doc.md for full usage notes.
"""

using PyCall
pyimport("matplotlib").use("Agg")
using PyPlot
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

# ─────────────────────────────────────────────────────────────
# Helpers  (identical logic to compute_pcc.jl)
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

"""
Load human CSV trimmed to exactly n_times survey points.
Extra rows beyond n_times*3 are dropped (undocumented survey points).
Returns (n_goals × n_times).
"""
function load_human(path::String, n_times::Int, n_goals::Int = N_GOALS)
    human_data_1d = vec(readdlm(path, ',', Float64))
    n_available   = div(length(human_data_1d), n_goals)
    n_use         = min(n_times, n_available)
    trimmed       = human_data_1d[1 : n_use * n_goals]
    return reshape(trimmed, n_goals, n_use)   # (n_goals × n_use)
end

function pearson_r(x::AbstractVector, y::AbstractVector)
    (length(x) < 2 || std(x) ≈ 0 || std(y) ≈ 0) && return NaN
    x̄, ȳ = mean(x), mean(y)
    num  = sum((x .- x̄) .* (y .- ȳ))
    den  = sqrt(sum((x .- x̄).^2) * sum((y .- ȳ).^2))
    den ≈ 0.0 && return NaN
    return num / den
end

"""
Extract (model_vals, human_vals) pairs for the scatter plot.

For each documented survey point i (1..n_times):
    model_vals[3*(i-1)+1 : 3*i] = model[:, times[i]]   (3 gem probabilities)
    human_vals[3*(i-1)+1 : 3*i] = human[:, i]          (3 gem probabilities)

Returns two flat vectors, each of length n_valid * n_goals.
"""
function extract_scatter_pairs(model::Matrix, human::Matrix, times::Vector{Int})
    n_times = size(human, 2)

    # Keep only times within the model's output range
    valid_mask  = [1 <= t <= size(model, 2) for t in times[1:n_times]]
    valid_times = times[1:n_times][valid_mask]
    valid_cols  = (1:n_times)[valid_mask]

    isempty(valid_times) && return Float64[], Float64[]

    model_subset = model[:, valid_times]   # (n_goals × n_valid)
    human_subset = human[:, valid_cols]    # (n_goals × n_valid)

    # vec() flattens column-major: all gems at t1, then all gems at t2, ...
    return vec(model_subset), vec(human_subset)
end

# ─────────────────────────────────────────────────────────────
# Collect scatter data across all 16 experiments
# ─────────────────────────────────────────────────────────────

stimuli_times = load_stimuli_times(STIMULI_PATH)

sips_x = Float64[];  sips_y = Float64[]
abs_x  = Float64[];  abs_y  = Float64[]
total_points = 0

for problem in PROBLEMS, s in SETS
    exp_id     = "$(problem)_$(s)"
    human_path = joinpath(HUMAN_DIR, "$(exp_id).csv")
    sips_path  = joinpath(SIPS_DIR,  "goal_probs_SIPS_$(exp_id).csv")
    abs_path   = joinpath(ABS_DIR,   "goal_probs_hierarchical_$(exp_id).csv")

    (!isfile(human_path) || !haskey(stimuli_times, exp_id)) && continue

    times     = stimuli_times[exp_id]
    human_mat = load_human(human_path, length(times))
    global total_points += size(human_mat, 2) * N_GOALS

    if isfile(sips_path)
        mx, hx = extract_scatter_pairs(load_model(sips_path), human_mat, times)
        append!(sips_x, mx);  append!(sips_y, hx)
    end

    if isfile(abs_path)
        mx, hx = extract_scatter_pairs(load_model(abs_path), human_mat, times)
        append!(abs_x, mx);  append!(abs_y, hx)
    end
end

@printf "Total scatter points per model: %d  (= Σ len(times_i) × 3 over 16 experiments)\n" total_points

# ─────────────────────────────────────────────────────────────
# Pearson r for legend labels
# ─────────────────────────────────────────────────────────────

sips_r = pearson_r(sips_x, sips_y)
abs_r  = pearson_r(abs_x,  abs_y)

@printf "SIPS PCC (all data):     %.4f\n" sips_r
@printf "Abstract PCC (all data): %.4f\n" abs_r

# ─────────────────────────────────────────────────────────────
# Scatter plot
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 6.0))

ax.scatter(sips_x, sips_y,
    color  = "steelblue",
    alpha  = 0.35,
    s      = 20,
    label  = @sprintf("SIPS  (r = %.3f)", sips_r),
    zorder = 2,
)
ax.scatter(abs_x, abs_y,
    color  = "tomato",
    alpha  = 0.35,
    s      = 20,
    label  = @sprintf("Abstract  (r = %.3f)", abs_r),
    zorder = 2,
)

# Identity line y = x
ax.plot([0.0, 1.0], [0.0, 1.0],
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
ax.set_xlabel("Model predicted probability", fontsize=12)
ax.set_ylabel("Human inferred probability",  fontsize=12)
ax.set_title("Model predictions vs Human goal inference\n(doors-keys-gems, all experiments)", fontsize=12)
ax.legend(loc="upper left", frameon=true)

out_path = joinpath(@__DIR__, "scatter_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
println("Saved → $out_path")
