module CRockSample

using LinearAlgebra
using POMDPs
using CPOMDPs
using POMDPModelTools
using StaticArrays
using Parameters
using Random
using Compose
using Combinatorics
using ParticleFilters
using DiscreteValueIteration
using POMDPPolicies

export
    RockSampleCPOMDP,
    CRSPos,
    CRSState,
    CRSExit,
    CRSExitSolver,
    CRSMDPSolver,
    CRSQMDPSolver

const CRSPos = SVector{2, Int}

"""
    CRSState{K}
Represents the state in a RockSampleCPOMDP problem. 
`K` is an integer representing the number of rocks

# Fields
- `pos::RPos` position of the robot
- `rocks::SVector{K, Bool}` the status of the rocks (false=bad, true=good)
"""
struct CRSState{K}
    pos::CRSPos 
    rocks::SVector{K, Bool}
end

@with_kw struct RockSampleCPOMDP{K} <: CPOMDP{CRSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    rocks_positions::SVector{K,CRSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::CRSPos = (1,1)
    sensor_efficiency::Float64 = 20.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    step_penalty::Float64 = 0.
    sensor_use_penalty::Float64 = 0.
    exit_reward::Float64 = 10.
    terminal_state::CRSState{K} = CRSState(CRSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
    bad_rock_cost_limit::Float64 = 1.
end

# to handle the case where rocks_positions is not a StaticArray
function RockSampleCPOMDP(map_size,
                         rocks_positions,
                         args...
                        )

    k = length(rocks_positions)
    return RockSampleCPOMDP{k}(map_size,
                              SVector{k,CRSPos}(rocks_positions),
                              args...
                             )
end

# Generate a random instance of RockSample(n,m) with a n×n square map and m rocks
RockSampleCPOMDP(map_size::Int, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG) = RockSampleCPOMDP((map_size,map_size), rocknum, rng)

# Generate a random instance of RockSample with a n×m map and l rocks
function RockSampleCPOMDP(map_size::Tuple{Int, Int}, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    possible_ps = [(i, j) for i in 1:map_size[1], j in 1:map_size[2]]
    selected = unique(rand(rng, possible_ps, rocknum))
    while length(selected) != rocknum
        push!(selected, rand(rng, possible_ps))
        selected = unique!(selected)
    end
    return RockSampleCPOMDP(map_size=map_size, rocks_positions=selected)
end

# To handle the case where the `rocks_positions` is specified
RockSampleCPOMDP(map_size::Tuple{Int, Int}, rocks_positions::AbstractVector) = RockSampleCPOMDP(map_size=map_size, rocks_positions=rocks_positions)

POMDPs.isterminal(pomdp::RockSampleCPOMDP, s::CRSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::RockSampleCPOMDP) = pomdp.discount_factor

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")
include("visualization.jl")
include("heuristics.jl")

end # module
