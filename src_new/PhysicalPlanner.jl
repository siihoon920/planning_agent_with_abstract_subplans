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

"define abstract actions as subgoals"
function abstract_actions(domain::PDDL.Domain, state::PDDL.State)
    ground = PDDL.ground(domain, state)
    actions = ground.actions
    filtered_actions = filter(action -> action.first in (:pickup, :unlock), actions)

    goals = SymbolicPlanners.ActionGoal[]
    for group in values(filtered_actions)
        for act in values(group.actions)
            push!(goals, SymbolicPlanners.ActionGoal(act.term))
        end
    end

    return goals
end

function solve(domain::PDDL.Domain, state::PDDL.State)
    sol = init_sol(domain, state)
    sol = search!(sol, domain, abstract_actions(domain, state))
    return sol
end

"Initialize MultiplePathsSearchSolution for a Dijkstra-style physical search."
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

"Lightweight logger: show queue plus a short state summary per entry"
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

"""
Dijkstra-style search explores all physical states 
within search space bound by abstract subgoals,
path cost is number of physical steps taken
"""
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
            for spec in specs # for each subgoal
                if SymbolicPlanners.is_goal(spec, domain, node.state, node.parent.action) 
                    sol.status = :deadend # set status to deadend if a subgoal is reached
                    push!(reached_goals, node_id) # save the id of the state in array reached_goals
                end
            end

            DataStructures.dequeue!(queue)
            # log_pq("dequeue", queue, search_tree)

            if sol.status == :in_progress 
            # if search didn't reach a subgoal(=deadend) above, then expand -> limit search space within reachable subgoals this ways
                expand!(node, search_tree, queue, domain, specs)
                sol.expanded += 1
                push!(sol.search_order, node_id)
            end
        end
    end

    if sol.status == :in_progress 
    # if queue runs out while still in progress, 
    # this means all states within bounded search spaced have been explored
        sol.status = :finished
    end

    if !isempty(reached_goals)
        for id in reached_goals
        # add pathcost, plan and trajectory of each reached goal to MultiplePathsSearchSolution
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
        
        # bring up path node if given state (arrived with given action) was already explored 
        # otherwise create a new path node
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
                # log_pq("enqueue", queue, search_tree)
            else
                queue[next_id] = path_cost
                # log_pq("priority update", queue, search_tree)
            end
        else
            next_node.parent.next = LinkedNodeRef(node.id, act, next_node.parent.next)
        end
    end
end

end # module PhysicalPlanner