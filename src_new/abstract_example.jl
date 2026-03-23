import DataStructures
import PDDL
import Printf
import SymbolicPlanners
import Gen
import GenParticleFilters
import PDDLViz
import GLMakie

include("../examples/doors-keys-gems/utils.jl")  # brings in GoalManhattan
include("PhysicalPlanner.jl")
include("AbstractPlanner.jl")

# Load the domain and problem-1
domain = PDDL.load_domain("examples/doors-keys-gems/domain.pddl")
problem = PDDL.load_problem("examples/doors-keys-gems/problems/problem-1.pddl")

# Initialize state, spec, and compile for better performance
state = PDDL.initstate(domain, problem)
spec = SymbolicPlanners.Specification(problem)
domain, state = PDDL.compiled(domain, state)

# Initialize a ProbAStarPlanner using RelaxedMazeDist heuristic from utils.jl
planner = SymbolicPlanners.ProbAStarPlanner(RelaxedMazeDist(), save_search=true)

println("Starting abstract search on problem 1...")

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