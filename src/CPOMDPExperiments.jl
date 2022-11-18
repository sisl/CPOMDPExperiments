module CPOMDPExperiments

using Infiltrator

using ProgressMeter
using Parameters
using POMDPSimulators
using POMDPGifs
using Cairo
using ParticleFilters

# non-constrained baseline
using POMDPs
using BasicPOMCP
using MCTS # belief-mcts for belief dpw
#using POMCPOW

# constrained solvers
using CPOMDPs
using CMCTS
using CPOMCP
#using CPOMCPOW

# unconstrained models 
using POMDPModels
using RockSample
using VDPTag2
#using RoombaPOMDPs

# constrained models 
using CRockSample
include("cpomdps/clightdark.jl")
include("cpomdps/cvdp.jl")
include("cpomdps/cspillpoint.jl")

# testing configurations
export
    test,
    run_all_tests
include("configs.jl")
include("experiments.jl")

end # module