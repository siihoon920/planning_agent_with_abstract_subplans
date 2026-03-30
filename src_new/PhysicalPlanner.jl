module PhysicalPlanner

using PDDL
using SymbolicPlanners
using DataStructures


const PathNode = SymbolicPlanners.PathNode
const LinkedNodeRef = SymbolicPlanners.LinkedNodeRef
const reconstruct = SymbolicPlanners.reconstruct

mutable struct MultiplePathsSearchSolution{
    S <: PDDL.State, T
}
    "Status of the returned solution."
    status::Symbol
    "Sequence of actions that reach the subgoal."
    plans::Vector{Vector{PDDL.Term}}
    "Trajectory of states that will be traversed while following the subplan."
    trajectories::Vector{Vector{S}}
    "Number of physical nodes expanded during search."
    path_costs::Vector{Float32}
    "path cost of subplan"
    expanded::Int
    "Tree of physical pathnodes expanded or evaluated during physical search."
    search_tree::Union{Dict{UInt,PathNode{S}}}
    "Frontier of yet-to-be-expanded search physical nodes (stored as references)."
    search_frontier::T
    "Order of nodes expanded during physical search (stored as references)."
    search_order::Vector{UInt}
end

function abstract_actions(domain::PDDL.Domain, state::PDDL.State)
    ground = PDDL.ground(domain, state)
    actions = ground.actions
    for action in actions
        println(action.first)
    end
    filtered_actions = filter(action -> action.first in (:pickup, :unlock), actions)

    println("Detailed GroundAction Inspection:")

    #=
    for (key, group) in filtered_actions
        for (term, action) in group.actions
            println("--- Action: $term ---")
            for field in fieldnames(typeof(action))
                println("$field: ", getfield(action, field))
            end
        end
    end
    =#

    goals = SymbolicPlanners.ActionGoal[]
    for group in values(filtered_actions)
        for act in values(group.actions)
            push!(goals, SymbolicPlanners.ActionGoal(act.term))
        end
    end

    return goals
end

"""
    init_sol(domain, state)

Initialize frontier and search tree for a Dijkstra-style physical search.
"""
function solve(domain::PDDL.Domain, state::PDDL.State)
    sol = init_sol(domain, state)
    sol = search!(sol, domain, abstract_actions(domain, state))
    return sol
end

function init_sol(domain::PDDL.Domain, state::PDDL.State)
    node_id = hash(state)
    node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))
    search_tree = Dict(node_id => node)
    queue = DataStructures.PriorityQueue(node_id => 0)
    search_order = UInt[]
    sol = MultiplePathsSearchSolution(
        :in_progress,
        Vector{Vector{PDDL.Term}}(),
        Vector{Vector{typeof(state)}}(),
        Float32[],
        0,
        search_tree,
        queue,
        UInt[]
    )
    return sol
end

"""
    search!(sol, domain, specs)

Dijkstra/Uniform-cost search without heuristics. Iterates over multiple
subgoal specs. Returns the same `sol`, with `plans`/`trajectories` filled
on success.
"""

# Lightweight logger: show queue plus a short state summary per entry
log_pq(op, queue, search_tree) = begin
    println("PQ after $op:")
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

function search!(
    sol::MultiplePathsSearchSolution,
    domain::PDDL.Domain,
    specs::AbstractVector{<:SymbolicPlanners.Specification}
) where {S <: PDDL.State}
    start_time = time()
    reached_goals = UInt[]
    queue, search_tree = sol.search_frontier, sol.search_tree

    while length(queue) > 0
        sol.status = :in_progress
        node_id, priority = peek(queue)
        node = search_tree[node_id]

        if time() - start_time >= 200000
            sol.status = :max_time && break
        elseif priority == Inf
            sol.status = :exhausted && break
        else
            parent_action = isnothing(node.parent) ? nothing : node.parent.action
            for spec in specs
                if SymbolicPlanners.is_goal(spec, domain, node.state, node.parent.action)
                    sol.status = :deadend
                    push!(reached_goals, node_id)
                end
            end

            DataStructures.dequeue!(queue)
            log_pq("dequeue", queue, search_tree)

            if sol.status == :in_progress
                expand!(node, search_tree, queue, domain, specs)
                sol.expanded += 1
                push!(sol.search_order, node_id)
            end
        end
    end

    if sol.status == :in_progress
        sol.status = :finished
    end

    if !isempty(reached_goals)
        for id in reached_goals
            push!(sol.path_costs, search_tree[id].path_cost)
            plan, traj = reconstruct(id, search_tree)
            push!(sol.plans, plan)
            push!(sol.trajectories, traj)
        end
    end
    
    if !isempty(reached_goals)
        for gid in reached_goals
            node = search_tree[gid]
            st = node.state
            
            println("Goal node $gid (cost=$(node.path_cost))")
            println("  objects: ", PDDL.get_objtypes(st))
            println("  facts: ", collect(PDDL.get_facts(st)))
            println("  fluents: ", join(
                [string(k, "=", v) for (k, v) in PDDL.get_fluents(st) if k != :walls], ", "
            ))
        end
    end
    return sol
end

function expand!(
    node::PathNode{S},
    search_tree::Dict{UInt,PathNode{S}},
    queue::DataStructures.PriorityQueue,
    domain::PDDL.Domain,
    specs::AbstractVector{<:SymbolicPlanners.Specification}
) where {S <: PDDL.State}
    state = node.state
    for act in PDDL.available(domain, state)
        next_state = PDDL.transition(domain, state, act; check=false)
        next_id = hash(next_state)
        act_cost = 1
        path_cost = node.path_cost + act_cost

        for spec in specs
            if SymbolicPlanners.is_goal(spec, domain, next_state, act)
                next_id = hash((next_state, act))
            end
        end

        next_node = get!(search_tree, next_id) do
            PathNode{S}(next_id, next_state, Inf32)
        end
        cost_diff = next_node.path_cost - path_cost
        if cost_diff > 0
            next_node.path_cost = path_cost
            next_node.parent = LinkedNodeRef(node.id, act, next_node.parent)
            node.child = LinkedNodeRef(next_id, nothing, node.child)
            if !(next_id in keys(queue))
                DataStructures.enqueue!(queue, next_id, path_cost)
                log_pq("enqueue", queue, search_tree)
            else
                queue[next_id] = path_cost
                log_pq("priority update", queue, search_tree)
            end
        else
            next_node.parent.next = LinkedNodeRef(node.id, act, next_node.parent.next)
        end
    end
end

#=
domain = PDDL.load_domain("examples/doors-keys-gems/domain.pddl")
problem = PDDL.load_problem("examples/doors-keys-gems/problems/problem-1.pddl")

state = PDDL.initstate(domain, problem)
spec = SymbolicPlanners.Specification(problem)

domain, state = PDDL.compiled(domain, state)

sol = init_sol(domain, state)
sol = search!(sol, domain, abstract_actions(domain, state))

println("Status: ", sol.status)
for (i, g) in pairs(subgoals)
    if i <= length(sol.plans)
        println("Subgoal $i $g")
        println("  Plan length = ", length(sol.plans[i]))
        println("  Plan = ", sol.plans[i])
    else
        println("Subgoal $i $g not reached")
    end
end
=#

end # module PhysicalPlanner