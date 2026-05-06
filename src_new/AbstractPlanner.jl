module AbstractPlanners

# Full using declarations — needed for bare names in included utils.jl
using PDDL
using SymbolicPlanners
using PDDLViz
using GLMakie
using Plinf
using DataStructures
using Gen
using GenParticleFilters
using Logging

PDDL.Arrays.register!()

const LinkedNodeRef = SymbolicPlanners.LinkedNodeRef
const reconstruct_internal = SymbolicPlanners.reconstruct
const LoggerCallback   = SymbolicPlanners.LoggerCallback
const simplify_goal    = SymbolicPlanners.simplify_goal
const prob_peek        = SymbolicPlanners.prob_peek
const prob_dequeue!    = SymbolicPlanners.prob_dequeue!

include("../example/doors-keys-gems/utils.jl")
include("PhysicalPlanner.jl")

# ──────────────────────────────────────────────────────────────────────────────
# AbstractPlanner struct
#
# Proper subtype of SymbolicPlanners.Planner with all necessary fields
# (matching ForwardPlanner's field set). This avoids delegation wrappers and
# ensures SIPS can copy the planner and set budget variables directly.
# ──────────────────────────────────────────────────────────────────────────────

mutable struct AbstractPlanner <: SymbolicPlanners.Planner
    heuristic::SymbolicPlanners.Heuristic
    search_noise::Union{Nothing, Float64}
    g_mult::Float32
    h_mult::Float32
    max_nodes::Int
    max_time::Float64
    fail_fast::Bool
    refine_method::Symbol
    reset_node_count::Bool
    save_search::Bool
    save_search_order::Bool
    save_parents::Bool
    save_children::Bool
    verbose::Bool
    callback::Union{Nothing, Function}
end

function AbstractPlanner(
    heuristic::SymbolicPlanners.Heuristic = GoalCountHeuristic();
    search_noise::Union{Nothing,Float64} = nothing,
    g_mult::Float32 = 1.0f0,
    h_mult::Float32 = 1.0f0,
    max_nodes::Int = typemax(Int),
    max_time::Float64 = Inf,
    fail_fast::Bool = false,
    refine_method::Symbol = :continue,
    reset_node_count::Bool = true,
    save_search::Bool = true,
    save_search_order::Bool = true,
    save_parents::Bool = false,
    save_children::Bool = false,
    verbose::Bool = false,
    callback = nothing
)
    return AbstractPlanner(
        heuristic, search_noise, g_mult, h_mult, max_nodes, max_time,
        fail_fast, refine_method, reset_node_count, save_search,
        save_search_order, save_parents, save_children, verbose, callback
    )
end

Base.copy(p::AbstractPlanner) = AbstractPlanner(
    p.heuristic, p.search_noise, p.g_mult, p.h_mult, p.max_nodes, p.max_time,
    p.fail_fast, p.refine_method, p.reset_node_count, p.save_search,
    p.save_search_order, p.save_parents, p.save_children, p.verbose, p.callback
)

# Budget variable for SIPS replanning: use node count (same as ForwardPlanner)
Plinf.default_budget_var(::AbstractPlanner) = :max_nodes

# Lightweight logger: show queue plus a short state summary per entry
log_pq(op, queue, search_tree) = begin
    println("Abstract level PQ after $op:")
    for (qid, pr) in collect(queue)
        node = get(search_tree, qid, nothing)
        if isnothing(node)
            println("  id=$(qid), priority=$(pr) (missing node)")
            continue
        end
        st = node.state
        if pr isa Tuple && length(pr) == 3
            f, h, n = pr
            println("  id=$(qid) priority=(f=$f, h=$h, n=$n) cost=$(node.path_cost)")
        else
            println("  id=$(qid) priority=$pr cost=$(node.path_cost)")
        end
        println("    facts: ", collect(PDDL.get_facts(st)))
        println("    fluents: ", join(
            [string(k, "=", v) for (k, v) in PDDL.get_fluents(st) if k != :walls], ", "
        ))
    end
end

"version of LinkedNodesRef for abstract states. Instead of a single physical action, a sequence of them lead to next abstract state"
mutable struct MultipleLinkedNodesRef{
    S<:PDDL.State
}
    id::UInt
    plan::Vector{PDDL.Term}
    trajectory::Vector{S}
    next::Union{MultipleLinkedNodesRef, Nothing}
end

MultipleLinkedNodesRef(id, plan, trajectory) = MultipleLinkedNodesRef(id, plan, trajectory, nothing)

"version of PathNode for abstract states. parent and child of an abstract node must be referred to with MultipleLinkedNodesRef"
mutable struct AbstractPathNode{S <: PDDL.State}
    id::UInt
    state::S
    path_cost::Float32
    parent::Union{MultipleLinkedNodesRef{S},Nothing}
    child::Union{MultipleLinkedNodesRef{S},Nothing}
end

AbstractPathNode(id::UInt, state::S, path_cost::Real=0.0) where {S<:PDDL.State} =
    AbstractPathNode{S}(id, state, Float32(path_cost), nothing, nothing)

"Internal abstract-level search solution. NOT a SymbolicPlanners.Solution subtype —
wrap into SymbolicPlanners.PathSearchSolution before returning from solve."
mutable struct AbstractSearchSolution{S, T}
    status::Symbol
    plan::Vector{PDDL.Term}
    trajectory::Vector{S}
    expanded::Int
    search_tree::Dict{UInt, AbstractPathNode{S}}
    search_frontier::T
    goal_queue::T
    search_order::Vector{UInt}
end

function solve(planner::AbstractPlanner,
               domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    heuristic, save_search = planner.heuristic, planner.save_search
    spec = simplify_goal(spec, domain, state)
    # Just like in Forward.jl, precompute heuristic information
    SymbolicPlanners.precompute!(heuristic, domain, state, spec)
    abs_sol = init_sol(planner, heuristic, domain, state, spec)
    # log_pq("initial", abs_sol.search_frontier, abs_sol.search_tree)
    if SymbolicPlanners.is_violated(spec, domain, state)
        abs_sol.status = :failure
    else
        abs_sol = search!(abs_sol, planner, heuristic, domain, spec)
    end
  
    empty_queue = DataStructures.PriorityQueue{UInt, Tuple{Float32,Float32,Int64}}()
    wrapped_status = abs_sol.status == :failure ? :failure : :in_progress
    sol = SymbolicPlanners.PathSearchSolution(
        wrapped_status,
        abs_sol.plan,
        abs_sol.trajectory,
        abs_sol.expanded,
        nothing,    # discard abstract search tree (not a PathNode tree)
        empty_queue,
        UInt[]
    )
    if save_search
        return sol
    elseif sol.status == :failure
        return SymbolicPlanners.NullSolution(sol.status)
    else
        return sol
    end
end

# Extend SymbolicPlanners.solve so SIPS/Plinf can dispatch through the standard interface
SymbolicPlanners.solve(p::AbstractPlanner, d::PDDL.Domain, s::PDDL.State,
                       spec::SymbolicPlanners.Specification) = solve(p, d, s, spec)

function init_sol(planner::AbstractPlanner, heuristic::SymbolicPlanners.Heuristic,
                  domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    node_id = hash(state)
    node = AbstractPathNode(node_id, state, 0.0f0, nothing, nothing)
    search_tree = Dict(node_id => node)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    h_val::Float32 = SymbolicPlanners.compute(heuristic, domain, state, spec)
    priority = (planner.h_mult * h_val, h_val, 0)
    queue = DataStructures.PriorityQueue(node_id => priority)
    goal_queue = DataStructures.PriorityQueue{UInt, Tuple{Float32,Float32,Int64}}()
    search_order = UInt[]
    sol = AbstractSearchSolution(
        :in_progress, PDDL.Term[], Vector{typeof(state)}(),
        0, search_tree, queue, goal_queue, search_order
    )
    return sol
end

function reinit_sol!(
    sol::PathSearchSolution{S, T},
    planner::AbstractPlanner, heuristic::SymbolicPlanners.Heuristic,
    domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification
) where {S, T <: DataStructures.PriorityQueue}
    search_tree, queue = sol.search_tree, sol.search_frontier
    sol.status = :in_progress
    empty!(sol.plan)
    empty!(sol.trajectory)
    empty!(sol.search_order)
    empty!(search_tree)
    node_id = hash(state)
    node = AbstractPathNode(node_id, state, 0.0f0, nothing, nothing)
    search_tree[node_id] = node
    empty!(queue)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    h_val::Float32 = SymbolicPlanners.compute(heuristic, domain, state, spec)
    priority = (planner.h_mult * h_val, h_val, 0)
    queue[node_id] = priority
    return sol
end

function search!(sol::AbstractSearchSolution,
                 planner::AbstractPlanner, heuristic::SymbolicPlanners.Heuristic,
                 domain::PDDL.Domain, spec::SymbolicPlanners.Specification)
    search_noise = planner.search_noise
    start_time = time()
    queue, search_tree = sol.search_frontier, sol.search_tree

    while length(queue) > 0
        # Budget / timeout check
        if sol.expanded >= planner.max_nodes
            sol.status = :max_nodes; break
        elseif time() - start_time >= planner.max_time
            sol.status = :max_time; break
        end

        # Expansion: uniform random for stochastic, best-first for deterministic.
        # Uniform exploration avoids heuristic bias during search; heuristic is
        # used only at commitment (prob_peek), after the frontier is built.
        if isnothing(search_noise)
            node_id, priority = Base.peek(queue)
        else
            frontier_ids = collect(keys(queue)) # turn dictionary into array
            node_id = frontier_ids[rand(1:length(frontier_ids))] # pick an element of the array at random
            priority = queue[node_id] # get priority of the element from the dictionary
        end
        node = search_tree[node_id]

        # Determine parent action for goal checking
        parent_action = if isnothing(node.parent)
            nothing
        elseif node.parent isa MultipleLinkedNodesRef
            isempty(node.parent.plan) ? nothing : node.parent.plan[end]
        else
            nothing
        end

        is_goal= SymbolicPlanners.is_goal(spec, domain, node.state, parent_action)

        #= commented out: immediate return on goal — let search continue to explore all paths
        if is_goal_node
            # Return immediately on goal detection — same logic for stochastic and deterministic
            sol.status = :success
            isnothing(search_noise) ?
                DataStructures.dequeue!(queue) : delete!(queue, node_id)
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, priority)
            end
            return sol
        end
        =#
        isnothing(search_noise) ?
            DataStructures.dequeue!(queue) : delete!(queue, node_id)
        expand!(planner, heuristic, node, search_tree, queue, domain, spec, sol.goal_queue)
        sol.expanded += 1
        if planner.save_search && planner.save_search_order
            push!(sol.search_order, node_id)
        end
        if !isnothing(planner.callback)
            planner.callback(planner, sol, node_id, priority)
        end
    end

    # Stochastic commitment via prob_peek — two-stage.
    # Stage 1: if any goal node was found during search, commit from goal nodes
    #   only. Without this, budget overshoot fills the frontier with partial-plan
    #   nodes whose f ≤ goal.f (admissible heuristic underestimates remaining
    #   cost), causing prob_peek to ignore the found goal.
    # Stage 2: no goal found within budget — refresh placeholder h=0 values for
    #   non-goal frontier nodes and prob_peek over the full frontier.
    if !isnothing(search_noise) && sol.status in (:in_progress, :max_nodes, :max_time)
        if !isempty(sol.goal_queue)
            node_id = first(prob_peek(sol.goal_queue, search_noise))
            sol.status = :success
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, sol.goal_queue[node_id])
            end
            return sol
        end
        for id in keys(queue)
            nd = search_tree[id]
            cur_f, _, _ = queue[id]
            if cur_f == 0.0f0
                h = Float32(SymbolicPlanners.compute(heuristic, domain, nd.state, spec))
                f = planner.g_mult * nd.path_cost + planner.h_mult * h
                queue[id] = (f, h, length(search_tree))
            end
        end
        if !isempty(queue)
            node_id = first(prob_peek(queue, search_noise))
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, queue[node_id])
            end
            return sol
        end
    end # closes if !isnothing(search_noise)
    if !isnothing(planner.callback)
        planner.callback(planner, sol, nothing, (Inf32, Inf32, 0))
    end
    sol.status = :failure
    return sol
end

function expand!(
    planner::AbstractPlanner, heuristic::SymbolicPlanners.Heuristic,
    node::AbstractPathNode{S},
    search_tree::Dict{UInt,AbstractPathNode{S}},
    queue::DataStructures.PriorityQueue,
    domain::PDDL.Domain, spec::SymbolicPlanners.Specification,
    goal_queue::DataStructures.PriorityQueue
) where {S <: PDDL.State}
    g_mult, h_mult = planner.g_mult, planner.h_mult
    state = node.state
    # from abstract state, call physical planner to get candidate next states
    subgoal = PhysicalPlanner.solve(domain, state)

    for i in 1:length(subgoal.path_costs)
        subgoal_state = subgoal.trajectories[i][end]
        subgoal_action = subgoal.plans[i][end]
        subgoal_id = hash(subgoal_state)
        if SymbolicPlanners.is_violated(spec, domain, subgoal_state) continue end
        act_cost = subgoal.path_costs[i]
        path_cost = node.path_cost + act_cost
        is_action_goal = false
        if SymbolicPlanners.has_action_goal(spec) &&
           SymbolicPlanners.is_goal(spec, domain, subgoal_state, subgoal_action)
            is_action_goal = true
            subgoal_id = hash((subgoal_state, subgoal_action))
        end
        # Check goal at discovery time so two-stage Stage 1 works even if never selected
        
        is_goal = SymbolicPlanners.is_goal(spec, domain, subgoal_state, subgoal_action)
        
        next_node = get!(search_tree, subgoal_id) do
            AbstractPathNode(subgoal_id, subgoal_state, Inf32)
        end
        cost_diff = next_node.path_cost - path_cost
        if cost_diff > 0
            next_node.path_cost = path_cost
            if planner.save_parents
                next_node.parent = MultipleLinkedNodesRef(node.id, subgoal.plans[i], subgoal.trajectories[i], next_node.parent)
            else
                next_node.parent = MultipleLinkedNodesRef(node.id, subgoal.plans[i], subgoal.trajectories[i])
            end

            if planner.save_children
                node.child = MultipleLinkedNodesRef(subgoal_id, nothing, nothing, node.child)
            end

            h_val::Float32 = is_goal ?
                0.0f0 : SymbolicPlanners.compute(heuristic, domain, subgoal_state, spec)
            f_val::Float32 = g_mult * path_cost + h_mult * h_val
            priority = (f_val, h_val, length(search_tree))

            if !(subgoal_id in keys(queue)) && !(subgoal_id in keys(goal_queue))
                if is_goal
                    DataStructures.enqueue!(goal_queue, subgoal_id, priority)
                else
                    DataStructures.enqueue!(queue, subgoal_id, priority)
                end
            elseif subgoal_id in keys(goal_queue)
                if isnothing(planner.search_noise)
                    _, h2, n2 = goal_queue[subgoal_id]
                    goal_queue[subgoal_id] = (f_val, h2, n2)
                end
            else
                if isnothing(planner.search_noise)
                    f2, h2, n2 = queue[subgoal_id]
                    queue[subgoal_id] = (f2 - cost_diff, h2, n2)
                end
            end

        elseif planner.save_parents
            next_node.parent.next =
                MultipleLinkedNodesRef(node.id, subgoal.plans[i], subgoal.trajectories[i], next_node.parent.next)
        end
    end
end

# similar to reconstruct in SymbolicPlanner, but uses MultipleLinkedNodesRef and connects abstract states
function reconstruct(node_id::UInt, search_tree::Dict)
    plan, trajectory = PDDL.Term[], PDDL.State[]
    curr_id = node_id
    while haskey(search_tree, curr_id)
        node = search_tree[curr_id]
        if isempty(trajectory)
            push!(trajectory, node.state)
        end
        parent_ref = node.parent
        if isnothing(parent_ref) || parent_ref.id == curr_id
            break
        end
        # parent_ref is MultipleLinkedNodesRef
        prepend!(plan, parent_ref.plan)
        # parent_ref.trajectory has states from parent to current.
        # The last state is current node.state, which is already in trajectory.
        prepend!(trajectory, parent_ref.trajectory[1:end-1])
        curr_id = parent_ref.id
    end
    return plan, trajectory
end

function refine!(
    sol::PathSearchSolution{S, T},
    planner::AbstractPlanner,
    domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification
) where {S, T <: DataStructures.PriorityQueue}
    heuristic, refine_method, reset_node_count =
        planner.heuristic, planner.refine_method, planner.reset_node_count
    spec = simplify_goal(spec, domain, state)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    if refine_method == :restart
        (sol.status == :failure && SymbolicPlanners.is_reached(state, sol)) && return sol
        if SymbolicPlanners.is_violated(spec, domain, state)
            sol.status = :failure
            return sol
        end
        reinit_sol!(sol, planner, heuristic, domain, state, spec)
    elseif refine_method == :reroot
        reroot!(sol, planner, heuristic, domain, state, spec)
        sol.status = :in_progress
    elseif refine_method == :continue
        (sol.status == :success || sol.status == :failure) && return sol
        sol.status = :in_progress
    end
    planner.reset_node_count && (sol.expanded = 0)
    return search!(sol, planner, heuristic, domain, spec)
end

end # module AbstractPlanners
