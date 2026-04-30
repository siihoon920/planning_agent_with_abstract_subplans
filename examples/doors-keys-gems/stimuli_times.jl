# Stimuli timepoints per experiment (provided by Sihoon).
#
# Each value is a 1-indexed state number: state T = after T-1 actions.
# These correspond to the endpoints of the video clips shown to human
# participants. Human goal-probability ratings are collected after each clip,
# so the number of entries equals the number of columns in the human CSV.

const STIMULI_TIMES = Dict{String, Vector{Int}}(
    "1_1" => [1, 7, 17, 23],
    "1_2" => [1, 9, 14, 17],
    "1_3" => [1, 9, 17, 24],
    "1_4" => [1, 7, 14, 23, 32],
    "2_1" => [1, 6, 11, 24],
    "2_2" => [1, 4,  6, 11],
    "2_3" => [1, 5,  8, 13],
    "2_4" => [1, 9, 12, 31, 44],
    "3_1" => [1, 7, 22, 37, 50],
    "3_2" => [1, 14, 24, 29, 40, 54],
    "3_3" => [1, 7, 13, 20, 26],
    "3_4" => [1, 6, 11, 26, 36, 49],
    "4_1" => [1, 8, 14, 20],
    "4_2" => [1, 4,  7, 10],
    "4_3" => [1, 5,  8, 10],
    "4_4" => [1, 7, 12, 18],
)