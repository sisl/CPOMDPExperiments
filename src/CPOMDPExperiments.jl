module CPOMDPExperiments

using Infiltrator

using ProgressMeter
using Parameters
using POMDPGifs
using Cairo
using ParticleFilters

using POMDPs
using POMDPTools
using CPOMDPs
import CPOMDPs: costs, costs_limit, n_costs, min_reward, max_reward

# non-constrained baseline
using BasicPOMCP
using MCTS # belief-mcts for belief dpw
using POMCPOW

# constrained solvers
using CMCTS
using CPOMCP
using CPOMCPOW

# models 
using POMDPModels
include("cpomdps/clightdark.jl")

using RockSample
include("cpomdps/crocksample.jl")

using VDPTag2
include("cpomdps/cvdp.jl")

using SpillpointPOMDP
include("cpomdps/cspillpoint.jl")

#using RoombaPOMDPs

# constrained models 

# testing configurations
export
    test,
    run_all_tests
include("configs.jl")
include("experiments.jl")

end # module