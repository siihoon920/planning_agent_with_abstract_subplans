## debug_abstract_plan_1_2.jl
## Runs AbstractPlanner (no inference) for each goal on the 1_2 trajectory.
## Logs plan contents and trajectory alignment to diagnose why gem1 probability
## drops immediately in SIPS inference despite being the true goal.

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen
using PDDLViz, GLMakie

include("utils.jl")
include("../../src_new/AbstractPlanner.jl")

PDDL.Arrays.register!()

# ── Load domain / problem / trajectory ────────────────────────────────────────

domain_path  = joinpath(@__DIR__, "domain.pddl")
plans_dir    = joinpath(@__DIR__, "../../domains/doors-keys-gems/plans")

# Accept experiment ID from command line (default: 1_1)
exp_id    = length(ARGS) > 0 ? ARGS[1] : "1_1"
all_files = readdir(plans_dir)
matched   = filter(f -> startswith(f, "$(exp_id)_"), all_files)
isempty(matched) && error("No plan file found for exp_id=$exp_id")
plan_file = matched[1]
println("Experiment: $exp_id  →  plan file: $plan_file")

m = match(r"problem_(\d+)_", plan_file)
prob_id = m === nothing ? error("Cannot parse problem ID") : m.captures[1]

domain  = load_domain(domain_path)
problem = load_problem(joinpath(@__DIR__, "problems/problem-$(prob_id).pddl"))
state   = initstate(domain, problem)
domain, state = PDDL.compiled(domain, state)

plan_strings = filter(l -> !isempty(strip(l)), readlines(joinpath(plans_dir, plan_file)))
plan         = [parse_pddl(a) for a in plan_strings]
obs_traj     = PDDL.simulate(domain, state, plan)

println("Observed trajectory: $(length(plan)) actions, $(length(obs_traj)) states")
println("Observed actions:")
for (i, a) in enumerate(plan)
    println("  t=$i: $a")
end
println()

# ── Planner config (mirror human_example_abstract.jl) ─────────────────────────

abs_planner = AbstractPlanners.AbstractPlanner(
    RelaxedMazeDist(); search_noise=0.1, save_search=true,
    max_nodes=15   # generous budget to ensure full exploration
)

# ── Debug callback helpers ────────────────────────────────────────────────────

function fmt_abstract_node(node_id, search_tree)
    node = get(search_tree, node_id, nothing)
    isnothing(node) && return "?"
    isnothing(node.parent) && return "ROOT"
    isempty(node.parent.plan) && return "ROOT"
    string(node.parent.plan[end])
end

function fmt_abstract_path(node_id, search_tree)
    node = get(search_tree, node_id, nothing)
    isnothing(node) && return "?"
    isnothing(node.parent) && return "ROOT"
    actions = String[]
    cur = node
    while !isnothing(cur) && !isnothing(cur.parent) && !isempty(cur.parent.plan)
        pushfirst!(actions, string(cur.parent.plan[end]))
        cur = get(search_tree, cur.parent.id, nothing)
    end
    isempty(actions) ? "ROOT" : join(actions, " → ")
end

function make_debug_callback()
    return function(planner, sol, node_id, priority)
        st    = sol.search_tree
        queue = sol.search_frontier
        act_str = fmt_abstract_node(node_id, st)

        if sol.status == :in_progress
            f, h, _ = priority
            pc = haskey(st, node_id) ? st[node_id].path_cost : Inf32
            path_str = fmt_abstract_path(node_id, st)
            println("    [expand #$(sol.expanded)] path=[$path_str]  g=$pc  f=$f  queue_size=$(length(queue))")
        elseif sol.status == :success
            chosen_path = fmt_abstract_path(node_id, st)
            println("    [COMMIT] → [$chosen_path]  priority=$priority")
            if !isnothing(planner.search_noise) && !isempty(queue)
                entries = collect(queue)
                fs      = [e[2][1] for e in entries]
                min_f   = minimum(fs)
                raw     = [exp((min_f - f) / planner.search_noise) for f in fs]
                total   = sum(raw)
                println("    Frontier probabilities:")
                for (i, (qid, pr)) in enumerate(entries)
                    path = fmt_abstract_path(qid, st)
                    pc2  = haskey(st, qid) ? st[qid].path_cost : Inf32
                    mark = qid == node_id ? " ← CHOSEN" : ""
                    @printf("      [%-60s]  g=%5.1f  f=%6.1f  prob=%.4f%s\n",
                            path, pc2, pr[1], raw[i]/total, mark)
                end
            end
        end
    end
end

goals      = @pddl("(has gem1)", "(has gem2)", "(has gem3)")
goal_names = ["gem1 (red/TRUE)", "gem2 (yellow)", "gem3 (blue)"]

# ── Helper: compare two state vectors element-wise ────────────────────────────

function states_match(s1::PDDL.State, s2::PDDL.State)
    xpos = parse_pddl("(xpos)"); ypos = parse_pddl("(ypos)")
    return s1[xpos] == s2[xpos] && s1[ypos] == s2[ypos]
end

function safe_has(s::PDDL.State, term)
    try; Int(s[term]); catch; 0; end
end

function fmt_state(s::PDDL.State)
    xpos = parse_pddl("(xpos)"); ypos = parse_pddl("(ypos)")
    has1 = parse_pddl("(has gem1)"); has2 = parse_pddl("(has gem2)"); has3 = parse_pddl("(has gem3)")
    hk1  = parse_pddl("(has key1)");  hk2  = parse_pddl("(has key2)");  hk3  = parse_pddl("(has key3)")
    g = (safe_has(s,has1), safe_has(s,has2), safe_has(s,has3))
    k = (safe_has(s,hk1),  safe_has(s,hk2),  safe_has(s,hk3))
    "($(s[xpos]),$(s[ypos])) gems=$g keys=$k"
end

# ── Run planner once per goal from INITIAL state ──────────────────────────────

println("="^70)
println("PLANNING FROM INITIAL STATE")
println("="^70)

for (goal, gname) in zip(goals, goal_names)
    println("\n--- Goal: $gname ---")
    spec = Specification(goal)
    dbg_planner = AbstractPlanners.AbstractPlanner(
        RelaxedMazeDist();
        search_noise = abs_planner.search_noise,
        save_search  = abs_planner.save_search,
        max_nodes    = abs_planner.max_nodes,
        callback     = make_debug_callback()
    )
    sol  = SymbolicPlanners.solve(dbg_planner, domain, state, spec)

    println("  status       : $(sol.status)")
    println("  plan length  : $(length(sol.plan))")
    println("  traj length  : $(length(sol.trajectory))")

    println("\n  Plan actions:")
    for (i, a) in enumerate(sol.plan)
        println("    step $i: $a")
    end

    println("\n  Trajectory states (plan):")
    for (i, s) in enumerate(sol.trajectory)
        println("    traj[$i]: $(fmt_state(s))")
    end

    println("\n  Observed states (first $(min(length(sol.trajectory), length(obs_traj))) steps):")
    n = min(length(sol.trajectory), length(obs_traj))
    for i in 1:n
        ps = sol.trajectory[i]
        os = obs_traj[i]
        match = states_match(ps, os) ? "✓" : "✗ MISMATCH"
        println("    obs[$i]: $(fmt_state(os))  |  plan[$i]: $(fmt_state(ps))  $match")
    end
    if length(sol.trajectory) < length(obs_traj)
        println("    ... plan ends at step $(length(sol.trajectory)), obs has $(length(obs_traj)) states")
    end
end

# ── Re-run from a few intermediate states to check replanning behaviour ────────

println()
println("="^70)
println("PLANNING FROM INTERMEDIATE STATES (replanning scenario)")
println("="^70)

check_steps = [2, 6, 13, 16]   # t=2 (up×2), t=6 (up×4+right×2), t=13 (just picked up key3), t=16 (just unlocked door2)

for t in check_steps
    t > length(obs_traj) && continue
    cur_state = obs_traj[t]
    println("\n--- Replanning from t=$t  state: $(fmt_state(cur_state)) ---")

    for (goal, gname) in zip(goals, goal_names)
        spec = Specification(goal)
        sol  = SymbolicPlanners.solve(abs_planner, domain, cur_state, spec)
        println("  [$gname]  status=$(sol.status)  plan_len=$(length(sol.plan))  first_action=$(isempty(sol.plan) ? "NONE" : string(sol.plan[1]))")

        # Check trajectory alignment with remaining obs_traj from t onward
        n_remaining = min(length(sol.trajectory), length(obs_traj) - t + 1)
        mismatches = 0
        for i in 1:n_remaining
            ps = sol.trajectory[i]
            os = obs_traj[t + i - 1]
            if !states_match(ps, os)
                mismatches += 1
                if mismatches <= 3
                    println("    MISMATCH at offset $i: plan=$(fmt_state(ps))  obs=$(fmt_state(os))")
                end
            end
        end
        if mismatches == 0
            println("    trajectory aligns with obs_traj for all $n_remaining steps ✓")
        else
            println("    $mismatches mismatches in first $n_remaining steps")
        end
    end
end

# ── Full abstract plan from each boundary (multi-step AbstractPlanner) ────────

println()
println("="^70)
println("FULL ABSTRACT PLAN AT EACH BOUNDARY (multi-step AbstractPlanner search)")
println("="^70)

# Abstract step boundaries: initial state + state after each pickup/unlock
abstract_boundaries = Int[1]
for (t, act) in enumerate(plan)
    if act.name in (:pickup, :unlock)
        push!(abstract_boundaries, t + 1)
    end
end
println("Abstract boundaries at obs_traj indices: $abstract_boundaries")

for t in abstract_boundaries
    t > length(obs_traj) && continue
    cur_state = obs_traj[t]
    println("\n── t=$(t-1)  state: $(fmt_state(cur_state)) ──")

    for (goal, gname) in zip(goals, goal_names)
        println("  --- $gname ---")
        spec = Specification(goal)
        dbg_b = AbstractPlanners.AbstractPlanner(
            RelaxedMazeDist();
            search_noise = abs_planner.search_noise,
            save_search  = abs_planner.save_search,
            max_nodes    = abs_planner.max_nodes,
            callback     = make_debug_callback()
        )
        sol = SymbolicPlanners.solve(dbg_b, domain, cur_state, spec)
        println("  status=$(sol.status)  plan=$(join(string.(sol.plan), " → "))")
    end
end

# ── Check best_action behaviour (what Plinf actually calls) ───────────────────

println()
println("="^70)
println("best_action CHECK  (what SIPS uses at each timestep)")
println("="^70)

for (goal, gname) in zip(goals, goal_names)
    spec = Specification(goal)
    sol  = SymbolicPlanners.solve(abs_planner, domain, state, spec)
    println("\n--- Goal: $gname ---")
    println("  plan length=$(length(sol.plan)), traj length=$(length(sol.trajectory))")
    for t in 1:min(15, length(obs_traj)-1)
        cur   = obs_traj[t]
        act   = SymbolicPlanners.best_action(sol, cur)
        truth = plan[t]
        if ismissing(act)
            println("  t=$t  state=$(fmt_state(cur))  best_action=MISSING (state not on plan trajectory)  (observed=$truth)")
        else
            match = (act == truth) ? "✓" : "✗  (observed=$truth)"
            println("  t=$t  state=$(fmt_state(cur))  best_action=$act  $match")
        end
    end
end

println("\nDone.")
