module AbstractPlanner

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

# --- INTERNAL ALIASES ---
# SymbolicPlanners unexported internals
const LinkedNodeRef = SymbolicPlanners.LinkedNodeRef
const reconstruct_internal = SymbolicPlanners.reconstruct
const LoggerCallback   = SymbolicPlanners.LoggerCallback
const simplify_goal    = SymbolicPlanners.simplify_goal
const prob_peek        = SymbolicPlanners.prob_peek
const prob_dequeue!    = SymbolicPlanners.prob_dequeue!

include("../examples/doors-keys-gems/utils.jl")
include("PhysicalPlanner.jl")

# Lightweight logger: show queue plus a short state summary per entry
log_pq(op, queue, search_tree) = begin
    println("Abstract level PQ after $op:")
    for (qid, pr) in collect(queue)
        node = get(search_tree, qid, nothing)
        if isnothing(node)
            println("  id=$(qid), pr=$(pr) (missing node)")
            continue
        end
        st = node.state
        println("  id=$(qid) pr=$(pr) cost=$(node.path_cost)")
        println("    facts: ", collect(PDDL.get_facts(st)))
        println("    fluents: ", join(
            [string(k, "=", v) for (k, v) in PDDL.get_fluents(st) if k != :walls], ", "
        ))
    end
end

mutable struct MultipleLinkedNodesRef{
    S<:PDDL.State
}
    id::UInt
    plan::Vector{PDDL.Term}
    trajectory::Vector{S}
    next::Union{MultipleLinkedNodesRef, Nothing}
end

MultipleLinkedNodesRef(id, plan, trajectory) = MultipleLinkedNodesRef(id, plan, trajectory, nothing) 

mutable struct AbstractPathNode{S <: PDDL.State}
    id::UInt
    state::S
    path_cost::Float32
    parent::Union{MultipleLinkedNodesRef{S},Nothing}
    child::Union{MultipleLinkedNodesRef{S},Nothing}
end

AbstractPathNode(id::UInt, state::S, path_cost::Real=0.0) where {S<:PDDL.State} = 
    AbstractPathNode{S}(id, state, Float32(path_cost), nothing, nothing)

mutable struct PathSearchSolution{S, T}
    status::Symbol
    plan::Vector{PDDL.Term}
    trajectory::Vector{S}
    expanded::Int
    search_tree::Dict{UInt, AbstractPathNode{S}}
    search_frontier::T
    search_order::Vector{UInt}
end

function solve(planner::SymbolicPlanners.ForwardPlanner,
               domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    heuristic, save_search = planner.heuristic, planner.save_search
    # Simplify goal specification
    spec = simplify_goal(spec, domain, state)
    # Precompute heuristic information
    SymbolicPlanners.precompute!(heuristic, domain, state, spec)
    # Initialize solution
    sol = init_sol(planner, heuristic, domain, state, spec)
    log_pq("initial", sol.search_frontier, sol.search_tree)
    # Check if initial state satisfies trajectory constraints
    if SymbolicPlanners.is_violated(spec, domain, state)
        sol.status = :failure
    else
        sol = search!(sol, planner, heuristic, domain, spec)
    end
    # Print subgoal results
    println("Status: ", sol.status)
    # Return solution
    if save_search
        return sol
    elseif sol.status == :failure
        return SymbolicPlanners.NullSolution(sol.status)
    else
        return sol
    end
end

function init_sol(planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
                  domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    node_id = hash(state)
    node = AbstractPathNode(node_id, state, 0.0f0, nothing, nothing)
    search_tree = Dict(node_id => node)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    h_val::Float32 = SymbolicPlanners.compute(heuristic, domain, state, spec)
    priority = (planner.h_mult * h_val, h_val, 0)
    queue = DataStructures.PriorityQueue(node_id => priority)
    search_order = UInt[]
    sol = PathSearchSolution(
        :in_progress, PDDL.Term[], Vector{typeof(state)}(),
        0, search_tree, queue, search_order
    )
    return sol
end

function reinit_sol!(
    sol::PathSearchSolution{S, T},
    planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
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

function search!(sol::PathSearchSolution,
                 planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
                 domain::PDDL.Domain, spec::SymbolicPlanners.Specification)
    search_noise = planner.search_noise
    start_time = time()
    queue, search_tree = sol.search_frontier, sol.search_tree
    while length(queue) > 0
        node_id, priority = isnothing(search_noise) ?
            Base.peek(queue) : prob_peek(queue, search_noise)
        node = search_tree[node_id]
        
        # Determine parent action for goal checking
        parent_action = if isnothing(node.parent)
            nothing
        elseif node.parent isa MultipleLinkedNodesRef
            isempty(node.parent.plan) ? nothing : node.parent.plan[end]
        else
            nothing
        end

        # Check search termination criteria
        if SymbolicPlanners.is_goal(spec, domain, node.state, parent_action)
            sol.status = :success
        elseif SymbolicPlanners.on_goal_path(spec, domain, node.state)
            sol.status = :success
        elseif sol.expanded >= planner.max_nodes
            sol.status = :max_nodes
        elseif time() - start_time >= planner.max_time
            sol.status = :max_time
        elseif planner.fail_fast && priority[1] == Inf
            sol.status = :failure
            break
        end
        if sol.status == :in_progress
            isnothing(search_noise) ?
                DataStructures.dequeue!(queue) : delete!(queue, node_id)
            log_pq("dequeue", queue, search_tree)
            expand!(planner, heuristic, node, search_tree, queue, domain, spec)
            sol.expanded += 1
            if planner.save_search && planner.save_search_order
                push!(sol.search_order, node_id)
            end
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, priority)
            end
        else
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, priority)
            end
            return sol
        end
    end
    if !isnothing(planner.callback)
        planner.callback(planner, sol, nothing, (Inf32, Inf32, 0))
    end
    sol.status = :failure
    return sol
end

function expand!(
    planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
    node::AbstractPathNode{S},
    search_tree::Dict{UInt,AbstractPathNode{S}},
    queue::DataStructures.PriorityQueue,
    domain::PDDL.Domain, spec::SymbolicPlanners.Specification
) where {S <: PDDL.State}
    g_mult, h_mult = planner.g_mult, planner.h_mult
    state = node.state
    # Call physical planner to get candidate next states
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
            
            if !(subgoal_id in keys(queue))
                h_val::Float32 = is_action_goal ?
                    0.0f0 : SymbolicPlanners.compute(heuristic, domain, subgoal_state, spec)
                f_val::Float32 = g_mult * path_cost + h_mult * h_val
                priority = (f_val, h_val, length(search_tree))
                DataStructures.enqueue!(queue, subgoal_id, priority)
                log_pq("enqueue", queue, search_tree)
            else
                f_val, h_val, n_nodes = queue[subgoal_id]
                queue[subgoal_id] = (f_val - cost_diff, h_val, n_nodes)
                log_pq("priority update", queue, search_tree)
            end
            
        elseif planner.save_parents
            next_node.parent.next =
                MultipleLinkedNodesRef(node.id, subgoal.plans[i], subgoal.trajectories[i], next_node.parent.next)
        end
    end
end

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
    planner::SymbolicPlanners.ForwardPlanner,
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

function reroot!(
    sol::PathSearchSolution{S, T},
    planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
    domain::PDDL.Domain, state::S, spec::SymbolicPlanners.Specification
) where {S <: PDDL.State, T}
    h_mult, g_mult, callback = planner.h_mult, planner.g_mult, planner.callback
    queue, search_tree = sol.search_frontier, sol.search_tree
    verbose = callback isa LoggerCallback
    cb = callback
    root_id = hash(state)
    if sol.status == :failure && SymbolicPlanners.is_reached(root_id, sol)
        return sol
    end
    if sol.status == :success && state in sol.trajectory
        return sol
    end
    if !SymbolicPlanners.is_expanded(root_id, sol)
        return reinit_sol!(sol, planner, heuristic, domain, state, spec)
    end
    root_node = search_tree[root_id]
    root_node.parent = nothing # Reset root parent
    verbose && Logging.@logmsg cb.loglevel "Marking nodes for deletion..."
    prev_root_id = hash(sol.trajectory[1])
    deleted = Set{UInt}()
    del_queue = [prev_root_id]
    while !isempty(del_queue)
        del_id = pop!(del_queue)
        del_id == root_id && continue
        del_node = search_tree[del_id]
        push!(deleted, del_id)
        child_ref = del_node.child
        del_node.child = nothing
        while !isnothing(child_ref)
            child_id = child_ref.id
            child_ref = child_ref.next
            child_id in deleted && continue
            child = get(search_tree, child_id, nothing)
            isnothing(child) && continue
            child.parent.id == del_id || continue
            push!(del_queue, child_id)
        end
    end
    verbose && Logging.@logmsg cb.loglevel "Deleting or reparenting marked nodes..."
    adopters = Set{UInt}()
    n_adopted = 0
    n_saved = length(search_tree) - length(deleted)
    filter!(!in(deleted), sol.search_order)
    for del_id in deleted
        del_node = search_tree[del_id]
        del_node.path_cost = Inf32
        del_state = del_node.state
        parent_ref = del_node.parent
        del_node.parent = nothing
        while !isnothing(parent_ref)
            parent_id = parent_ref.id
            parent_ref = parent_ref.next
            parent_id in deleted && continue
            parent_id in keys(search_tree) || continue
            parent_id in keys(queue) && continue
            parent = search_tree[parent_id]
            push!(adopters, parent_id)
        end
        in_queue = haskey(queue, del_id)
        in_queue || (sol.expanded -= 1)
        if isnothing(del_node.parent)
            delete!(search_tree, del_id)
            in_queue && delete!(queue, del_id)
        else
            h_val::Float32 = in_queue ?
                queue[del_id][2] : SymbolicPlanners.compute(heuristic, domain, del_state, spec)
            f_val::Float32 = g_mult * del_node.path_cost + h_mult * h_val
            priority = (f_val, h_val, n_saved + n_adopted)
            queue[del_id] = priority
            n_adopted += 1
        end
    end
    if verbose
        n_marked = length(deleted)
        n_deleted = length(deleted) - n_adopted
        n_adopters = length(adopters)
        stats_str = "marked = $n_marked, deleted = $n_deleted, " *
            "adopted = $n_adopted, adopters = $n_adopters, saved = $n_saved"
        Logging.@logmsg cb.loglevel "Rerooting complete: " * stats_str
    end
    return sol
end

function (cb::LoggerCallback)(
    planner::SymbolicPlanners.ForwardPlanner,
    sol::PathSearchSolution,
    node_id::Union{UInt, Nothing}, priority
)
    node = isnothing(node_id) ? nothing : sol.search_tree[node_id]
    f, h, _ = priority
    g = isnothing(node) ? Inf32 : node.path_cost
    m, n = length(sol.search_tree), sol.expanded
    schedule = get(cb.options, :log_period_schedule,
                   [(10, 2), (100, 10), (1000, 100), (typemax(Int), 1000)])
    idx = findfirst(x -> n < x[1], schedule)
    log_period = isnothing(idx) ? 1000 : schedule[idx][2]
    if n <= 1 && get(cb.options, :log_header, true)
        Logging.@logmsg cb.loglevel "Starting forward search..."
        max_nodes, max_time = planner.max_nodes, planner.max_time
        Logging.@logmsg cb.loglevel "max_nodes = $max_nodes, max_time = $max_time"
        search_noise = planner.search_noise
        if !isnothing(search_noise)
            Logging.@logmsg cb.loglevel "search_noise = $search_noise"
        end
    end
    if n % log_period == 0 || sol.status != :in_progress
        Logging.@logmsg cb.loglevel "f = $f, g = $g, h = $h, $m evaluated, $n expanded"
    end
    if sol.status != :in_progress && get(cb.options, :log_solution, true)
        k = length(sol.plan)
        Logging.@logmsg cb.loglevel "Search terminated with status: $(sol.status)"
        if sol.status != :failure
            sol_str = sol.status == :success ? "Solution" : "Partial solution"
            init_node = sol.search_tree[hash(sol.trajectory[1])]
            init_cost = init_node.path_cost
            c = g - init_cost
            stats_str = iszero(init_cost) ?
                "$k actions, $c cost, $m evaluated, $n expanded" :
                "$k actions, $c cost ($g total), $m evaluated, $n expanded"
            Logging.@logmsg cb.loglevel "$sol_str: $stats_str"
        end
    end
end

end # module AbstractPlanner
