# planning_agent_with_abstract_subplans

based on Plinf.jl https://github.com/ztangent/Plinf.jl/tree/master#

## Setup
Download Julia in the terminal or from Julia Website (If you haven't done so)

    brew install juliaup.

Clone this repository in directory of your choice   

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
    
## Changes so far
`example.jl` 
- functions of the package GLMakie fixed to suit latest version
- file outputs (2 gif's 1 png) now saves as files (previously disappears after run)
  
## Experiment Selection & Format Conversion (Doors-Keys-Gems)

### Why only 8 of 16 experiments?

The original human experiment data covers 16 trials across 12 distinct map problems (problems 1–12). However, the `master` branch uses a **new PDDL format** (bit-matrix walls, door objects) compatible only with PDDL.jl ≥ 0.2.x. New-format problem files exist in `examples/doors-keys-gems/problems/` only for problems 4–7. The remaining problems (8–12) do not have new-format counterparts and cannot be run with the current codebase.

**8 experiments kept** (problems 4–7, new-format problem files available):

| Experiment | Problem | True Goal |
|------------|---------|-----------|
| `1_1` | 6 | gem2 |
| `1_2` | 7 | gem1 |
| `1_3` | 4 | gem1 |
| `2_2` | 5 | gem1 |
| `3_1` | 6 | gem1 |
| `3_2` | 4 | gem2 |
| `4_1` | 7 | gem2 |
| `4_2` | 5 | gem3 |

**8 experiments dropped** (problems 8–12, no new-format problem files):
`1_4`, `2_1`, `2_3`, `2_4`, `3_3`, `3_4`, `4_3`, `4_4`

### What was converted?

The original human plan files in `domains/doors-keys-gems/plans/` were recorded against the **old PDDL format** (`project-blocks` branch). Two things differ from the `master` format:

**1. Coordinate system:**
- Old format: `y=1` is the bottom row, `y` increases upward
- New format: `y=1` is the top row, `y` increases downward
- Transform: `new_y = height + 1 - old_y` (height = 9 for all maps)
- `x` is unchanged

**2. Unlock action syntax:**
- Old format: `(unlock key direction)` — e.g., `(unlock key2 left)`
- New format: `(unlock key doorN)` — e.g., `(unlock key2 door2)`

Each unlock action was converted by tracing the agent's position to find which door it was adjacent to, then mapping that to the named door object in the new problem file. Movement and pickup actions are identical between formats.

Converted plan files are stored in `examples/doors-keys-gems/plans/`.

**3. Gem naming in problem files:**

For problems 4 and 6, gems were named in a different order in the new-format problem files compared to the old format, causing pickup actions to fail. The gem coordinates in the new problem files were corrected to match the old naming:

- `examples/doors-keys-gems/problems/problem-4.pddl`: gem2 and gem3 positions swapped
- `examples/doors-keys-gems/problems/problem-6.pddl`: gem1 and gem3 positions swapped

Only internal labels were changed — physical positions, trajectories, and human judgment data are unaffected.



Plinf Minimum Structure notes (prone to updates), to help quickly understand 
https://web.goodnotes.com/s/tGMu55c04BUG8SRlZjBhge