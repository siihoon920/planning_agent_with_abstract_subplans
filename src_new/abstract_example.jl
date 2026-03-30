using PDDL              # bare names needed by utils.jl (@pddl_str, State, Domain, etc.)
using SymbolicPlanners  # bare names needed by utils.jl (Heuristic, Planner, AStarPlanner, etc.)
using PDDLViz           # bare names needed by utils.jl (GridworldRenderer, Canvas, etc.)
using GLMakie           # bare names needed by utils.jl (Figure, GridLayout, etc.)
using Plinf             # bare names needed by utils.jl (SIPSCallback, etc.)
using DataStructures    # bare names needed by utils.jl (OrderedDict)
using Gen
using GenParticleFilters
using Printf

include("../examples/doors-keys-gems/utils.jl")
include("PhysicalPlanner.jl")
include("AbstractPlanner.jl")

# Load the domain and problem-1
domain = PDDL.load_domain("examples/doors-keys-gems/domain.pddl")
problem = PDDL.load_problem("examples/doors-keys-gems/problems/problem-3.pddl")

# Initialize state, spec, and compile for better performance
state = PDDL.initstate(domain, problem)
spec = SymbolicPlanners.Specification(problem)
domain, state = PDDL.compiled(domain, state)

# Initialize a ProbAStarPlanner using RelaxedMazeDist heuristic from utils.jl
planner = SymbolicPlanners.ProbAStarPlanner(GoalManhattan(), save_search=true)

println("Starting abstract search on problem 3...")

# Run the abstract search via AbstractPlanner module
sol = AbstractPlanner.solve(planner, domain, state, spec)

# Print Final Output
println("\n=== Final Solution ===")
println("Status: ", sol.status)
if sol.status == :success
    println("Plan length: ", length(sol.plan))
    println("Plan: ", sol.plan)
    println("Trajectory: ", sol.trajectory)
end