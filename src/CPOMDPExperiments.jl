module CPOMDPExperiments

using Infiltrator

using ProgressMeter
using Parameters
using POMDPGifs
# using Cairo
using ParticleFilters
using LinearAlgebra
using Plots
import Statistics
using Random
using Distributed
using FileIO

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
export CLightDark1D
include("cpomdps/clightdark.jl")
export LightDarkNew, CLightDarkNew, zeroV_trueC
include("cpomdps/clightdarknew.jl")

using RockSample
export RockSampleCPOMDP
include("cpomdps/crocksample.jl")

using VDPTag2
export CVDPTagPOMDP
include("cpomdps/cvdp.jl")

using SpillpointPOMDP
export SpillpointInjectionCPOMDP
include("cpomdps/cspillpoint.jl")

#using RoombaPOMDPs


# helpers
export
    plot_lightdark_beliefs,
    SoftConstraintPOMDPWrapper,
    LightExperimentResults,
    ExperimentResults,
    print_and_save,
    load_and_print,
    mean,
    std,
    zero_V,
    QMDP_V,
    LambdaExperiments,
    save_le, load_le,
    SearchProgress
include("utils.jl") 

# experiment scripts
export
    run_pomdp_simulation,
    run_cpomdp_simulation,
    run_lambda_experiments
include("experiments.jl")


end # module