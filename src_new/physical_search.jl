
using PDDL
using SymbolicPlanners
using DataStructures
PDDL.Arrays.register!()

# --- IMPORT INTERNALS VIA ALIAS ---
# This is the robust way to "import" internal types so you don't have to prefix them
const PathNode = SymbolicPlanners.PathNode
const LinkedNodeRef = SymbolicPlanners.LinkedNodeRef
const reconstruct = SymbolicPlanners.reconstruct

mutable struct MultiplePathsSearchSolution{
    S <: State, T
}   "Status of the returned solution."
    status::Symbol
    "Sequence of actions that reach the goal. May be partial / incomplete."
    plans::Vector{Vector{Term}}
    "Trajectory of states that will be traversed while following the plan."
    trajectories::Vector{Vector{S}}
    "Number of nodes expanded during search."
    expanded::Int
    "Tree of [`PathNode`](@ref)s expanded or evaluated during search."
    search_tree::Union{Dict{UInt,PathNode{S}}}
    "Frontier of yet-to-be-expanded search nodes (stored as references)."
    search_frontier::T
    "Order of nodes expanded during search (stored as references)."
    search_order::Vector{UInt}
end

function abstract_actions(domain::PDDL.Domain,state::State)
    ground = PDDL.ground(domain,state)
    actions=ground.actions
    for action in actions
        println(action.first)
    end
    filtered_actions=filter(action -> action.first in (:pickup, :unlock), actions)

    println("Detailed GroundAction Inspection:")
    
    #=

    for (key, group) in filtered_actions
        # group is a GroundActionGroup, we iterate over its specific ground actions
        for (term, action) in group.actions
            println("--- Action: $term ---")
            # Reflection to print all fields of the GroundAction object
            for field in fieldnames(typeof(action))
                println("$field: ", getfield(action, field))
            end
        end
    end
    =# 
    goals = ActionGoal[]
    for group in values(filtered_actions)
        for act in values(group.actions)
             push!(goals, ActionGoal(act.term))
        end
    end
      # Convert GroundAction to ActionGoal by accessing the .term field
      # The most common way
    
    # dump(goals)
    # println("Filtered action goals: ", goals[0]) 
    return goals
    
end

"""
    init_sol(domain, state)

Initialize frontier and search tree for a Dijkstra-style physical search.
"""
function init_sol(domain::Domain, state::State)
    node_id = hash(state)
    node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))
    search_tree = Dict(node_id => node)
    queue = PriorityQueue(node_id => 0)
    search_order = UInt[]
    sol = MultiplePathsSearchSolution(
        :in_progress,
        Vector{Vector{Term}}(),
        Vector{Vector{typeof(state)}}(), # because in Julia specifying subtype is necessary
        0,
        search_tree,
        queue,
        UInt[]
    )
    return sol
end

"""
    search!(sol, domain, goal; cost_fn=(d,s1,a,s2)->1.0)

Dijkstra/Uniform-cost search without heuristics. Stops expanding a node as
soon as `goal` is satisfied in that state. Returns the same `sol`, with
`plan`/`trajectory` filled on success.
"""

# Lightweight logger: show queue plus a short state summary per entry
log_pq(op, queue, search_tree) = begin
    entries = collect(queue)
    println("PQ after " * op * ":")
    for (qid, pr) in entries
        node = get(search_tree, qid, nothing)
        if isnothing(node)
            println("  id=$(qid), pr=$(pr) (missing node)")
            continue
        end
        st = node.state
        println("  pr=$(pr)")
        println("    objects: ", get_objtypes(st))
        println("    facts: ", collect(get_facts(st)))
        println("    fluents: ", collect(get_fluents(st)))
    end
end

function search!(sol::MultiplePathsSearchSolution, domain::Domain, specs::AbstractVector{<:Specification})where {S <: State} # multiple subgoals vs one 
    start_time = time()
    reached_goals = UInt[]
    queue, search_tree = sol.search_frontier, sol.search_tree
    while length(queue) > 0
        sol.status=:in_progress # stopped, continue search from differnt frontier even when deadend
        # peek highest priority deterministically vs probabilistically
        node_id, priority = peek(queue)
        node = search_tree[node_id]
        
        if time() - start_time >= 200000
            sol.status = :max_time && break # Time budget reached 
        elseif priority == Inf # in case there are nodes left but infinite path costs
            sol.status = :exhausted && break # Search space exhausted means success vs failure
        else
            parent_action = isnothing(node.parent) ? nothing : node.parent.action
            for spec in specs
                if is_goal(spec, domain, node.state, node.parent.action) # do not declare success
                    sol.status = :deadend
                    push!(reached_goals, node_id)
                end
            end
        
            dequeue!(queue)
            log_pq("dequeue", queue, search_tree)
            
            if sol.status == :in_progress
            # Expand current node
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
            plan, traj = reconstruct(id, search_tree)
            push!(sol.plans, plan)
            push!(sol.trajectories, traj)
        end
    end
    
    return sol

end

function expand!(
    node::PathNode{S}, search_tree::Dict{UInt,PathNode{S}}, queue::PriorityQueue,
    domain::Domain, specs::AbstractVector{<:Specification}
) where {S <: State}
    state = node.state
    # Iterate over available actions, filtered by heuristic
    for act in available(domain, state)
        # Execute action and trigger all post-action events
        next_state = transition(domain, state, act; check=false)
        next_id = hash(next_state)
        # Check if next state satisfies trajectory constraints
        if is_violated(spec, domain, next_state) continue end
        # Compute path cost
        act_cost = 1 # replace get_cost()
        path_cost = node.path_cost + act_cost
        # Check if action goal is reached
    
        for spec in specs
            if is_goal(spec, domain, next_state, act)
                next_id = hash((next_state, act)) # if action leading to goal matter even if same final state
            end
        end
        # Construct or retrieve child node
        next_node = get!(search_tree, next_id) do
            PathNode{S}(next_id, next_state, Inf32)
        end
        cost_diff = next_node.path_cost - path_cost
        if cost_diff > 0  # Update path costs if new path is shorter
            next_node.path_cost = path_cost
            # Update parent and child pointers
            next_node.parent = LinkedNodeRef(node.id, act, next_node.parent)
            node.child = LinkedNodeRef(next_id, nothing, node.child)
            # Update estimated cost from next state to goal
            if !(next_id in keys(queue))
                enqueue!(queue, next_id, path_cost)
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

domain = load_domain("examples/doors-keys-gems/domain.pddl")
problem = load_problem("examples/doors-keys-gems/problems/problem-1.pddl")

state = initstate(domain, problem)
spec = Specification(problem)

domain, state = PDDL.compiled(domain, state)

# Example subgoals: first abstract step toward gem via key/door sequence
subgoals = [
    pddl"(has key2)",
    pddl"(not (locked door2))",
    pddl"(has key1)",
    pddl"(has gem3)"
]

sol = init_sol(domain, state)
sol = search!(sol, domain, abstract_actions(domain,state))

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
