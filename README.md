# planning_agent_with_abstract_subplans

based on Plinf.jl https://github.com/ztangent/Plinf.jl/tree/master#

## Setup
Download Julia in the terminal or from Julia Website (If you haven't done so)

    brew install juliaup.

Clone the repository in directory of your choice   
(original tells us to use `add` and use chunks of their code to our liking, but I think exactly cloning the original helps more at this stage for us)

    git clone git@github.com:siihoon920/planning_agent_with_abstract_subplans.git
    
enter Julia REPL in terminal by typing `julia` & hitting enter. You will see your terminal say 'Julia' (If you haven't done so)

    add PDDL SymbolicPlanners
    add Gen GenParticleFilters
    add PDDLViz GLMakie
    

IMPORTANT: above packages fail with latest julia, you need to downgrade julia to version 1.6 with which the packages were written

    juliaup add 1.6
    juliaup default 1.6

Then inside julia, enter packages by typing `]`. You'll see your terminal say 'pkg'

    activate .
    instantiate
    
