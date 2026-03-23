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
const PathNode      = SymbolicPlanners.PathNode
const LinkedNodeRef = SymbolicPlanners.LinkedNodeRef
const reconstruct   = SymbolicPlanners.reconstruct
const LoggerCallback   = SymbolicPlanners.LoggerCallback
const simplify_goal    = SymbolicPlanners.simplify_goal
const prob_peek        = SymbolicPlanners.prob_peek
const prob_dequeue!    = SymbolicPlanners.prob_dequeue!

include("../examples/doors-keys-gems/utils.jl")
include("PhysicalPlanner.jl")

function solve(planner::SymbolicPlanners.ForwardPlanner,
               domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    heuristic, save_search = planner.heuristic, planner.save_search
    # Simplify goal specification
    spec = simplify_goal(spec, domain, state)
    # Precompute heuristic information
    SymbolicPlanners.precompute!(heuristic, domain, state, spec)
    # Initialize solution
    sol = init_sol(planner, heuristic, domain, state, spec)
    # Check if initial state satisfies trajectory constraints
    if SymbolicPlanners.is_violated(spec, domain, state)
        sol.status = :failure
    else
        sol = search!(sol, planner, heuristic, domain, spec)
    end
    # Print subgoal results
    println("Status: ", sol.status)
    for (i, g) in pairs(current_subgoals)
        if i <= length(sol.plans)
            println("Subgoal $i $g")
            println("  Plan length = ", length(sol.plans[i]))
            println("  Plan = ", sol.plans[i])
        else
            println("Subgoal $i $g not reached")
        end
    end
    # Return solution
    if save_search
        return sol
    elseif sol.status == :failure
        return SymbolicPlanners.NullSolution(sol.status)
    else
        return SymbolicPlanners.PathSearchSolution(sol.status, sol.plan, sol.trajectory)
    end
end

function init_sol(planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
                  domain::PDDL.Domain, state::PDDL.State, spec::SymbolicPlanners.Specification)
    node_id = hash(state)
    node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))
    search_tree = Dict(node_id => node)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    h_val::Float32 = SymbolicPlanners.compute(heuristic, domain, state, spec)
    priority = (planner.h_mult * h_val, h_val, 0)
    queue = DataStructures.PriorityQueue(node_id => priority)
    search_order = UInt[]
    sol = SymbolicPlanners.PathSearchSolution(
        :in_progress, PDDL.Term[], Vector{typeof(state)}(),
        0, search_tree, queue, search_order
    )
    return sol
end

function reinit_sol!(
    sol::SymbolicPlanners.PathSearchSolution{S, T},
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
    node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))
    search_tree[node_id] = node
    empty!(queue)
    SymbolicPlanners.ensure_precomputed!(heuristic, domain, state, spec)
    h_val::Float32 = SymbolicPlanners.compute(heuristic, domain, state, spec)
    priority = (planner.h_mult * h_val, h_val, 0)
    queue[node_id] = priority
    return sol
end

function search!(sol::SymbolicPlanners.PathSearchSolution,
                 planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
                 domain::PDDL.Domain, spec::SymbolicPlanners.Specification)
    search_noise = planner.search_noise
    start_time = time()
    queue, search_tree = sol.search_frontier, sol.search_tree
    while length(queue) > 0
        node_id, priority = isnothing(search_noise) ?
            Base.peek(queue) : prob_peek(queue, search_noise)
        node = search_tree[node_id]
        # Check search termination criteria
        if SymbolicPlanners.is_goal(spec, domain, node.state, node.parent.action)
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
    node::PathNode{S},
    search_tree::Dict{UInt,PathNode{S}},
    queue::DataStructures.PriorityQueue,
    domain::PDDL.Domain, spec::SymbolicPlanners.Specification
) where {S <: PDDL.State}
    g_mult, h_mult = planner.g_mult, planner.h_mult
    state = node.state
    # Call physical planner to get candidate next states
    physical_sol = PhysicalPlanner.solve(domain, state)

    for i in length(physical_sol.path_costs)
        next_state = physical_sol.trajectories[i][end]   # fixed: [end] not [-1]
        next_id = hash(next_state)
        if SymbolicPlanners.is_violated(spec, domain, next_state) continue end
        act_cost = physical_sol.path_costs[i]
        path_cost = node.path_cost + act_cost
        is_action_goal = false
        if SymbolicPlanners.has_action_goal(spec) &&
           SymbolicPlanners.is_goal(spec, domain, next_state, act)
            is_action_goal = true
            next_id = hash((next_state, act))
        end
        next_node = get!(search_tree, next_id) do
            PathNode{S}(next_id, next_state, Inf32)
        end
        cost_diff = next_node.path_cost - path_cost
        if cost_diff > 0
            next_node.path_cost = path_cost
            if planner.save_parents
                next_node.parent = LinkedNodeRef(node.id, act, next_node.parent)
            else
                next_node.parent = LinkedNodeRef(node.id, act)
            end
            if planner.save_children
                node.child = LinkedNodeRef(next_id, nothing, node.child)
            end
            if !(next_id in keys(queue))
                h_val::Float32 = is_action_goal ?
                    0.0f0 : SymbolicPlanners.compute(heuristic, domain, next_state, spec)
                f_val::Float32 = g_mult * path_cost + h_mult * h_val
                priority = (f_val, h_val, length(search_tree))
                DataStructures.enqueue!(queue, next_id, priority)
            else
                f_val, h_val, n_nodes = queue[next_id]
                queue[next_id] = (f_val - cost_diff, h_val, n_nodes)
            end
        elseif planner.save_parents
            next_node.parent.next =
                LinkedNodeRef(node.id, act, next_node.parent.next)
        end
    end
end

function refine!(
    sol::SymbolicPlanners.PathSearchSolution{S, T},
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
    sol::SymbolicPlanners.PathSearchSolution{S},
    planner::SymbolicPlanners.ForwardPlanner, heuristic::SymbolicPlanners.Heuristic,
    domain::PDDL.Domain, state::S, spec::SymbolicPlanners.Specification
) where {S <: PDDL.State}
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
    root_node.parent = LinkedNodeRef(root_id)
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
            parent_act = parent_ref.action
            parent_ref = parent_ref.next
            parent_id in deleted && continue
            parent_id in keys(search_tree) || continue
            parent_id in keys(queue) && continue
            parent = search_tree[parent_id]
            act_cost = SymbolicPlanners.get_cost(spec, domain, parent.state,
                                                  parent_act, del_state)
            path_cost = parent.path_cost + act_cost
            if path_cost < del_node.path_cost
                del_node.path_cost = path_cost
                del_node.parent =
                    LinkedNodeRef(parent_id, parent_act, del_node.parent)
                parent.child =
                    LinkedNodeRef(del_id, nothing, parent.child)
                push!(adopters, parent_id)
            else
                del_node.parent.next =
                    LinkedNodeRef(parent_id, parent_act, del_node.parent.next)
            end
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
    for parent_id in adopters
        parent = search_tree[parent_id]
        parent.child = unique(parent.child)
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
    sol::SymbolicPlanners.PathSearchSolution,
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