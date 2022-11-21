module CPOMCP

#=
Current constraints:
- action space discrete
- action space same for all states, histories
- no built-in support for history-dependent rollouts (this could be added though)
- initial n and initial v are 0
=#

using CPOMDPs

using POMDPs
using BasicPOMCP
using Infiltrator
using Parameters
using ParticleFilters
using CPUTime
using Colors
using Random
using Printf
using POMDPLinter
using POMDPTools

import POMDPs: action, solve, updater, simulate, update, initialize_belief
import POMDPLinter: @POMDP_require, @show_requirements

using MCTS
import MCTS: convert_estimator, estimate_value, node_tag, tooltip_tag, default_action
import BasicPOMCP: estimate_value, rollout, extract_belief

using D3Trees

export
    # here
    AbstractCPOMCPSolver,
    AbstractCPOMCPPlanner,
    CPOMCPSolver,
    CPOMCPPlanner,
    CPOMCPDPWSolver,
    CPOMCPDPWPlanner,
    updater,
    update,
    initialize_belief,
    solve,
    POMCPBudgetUpdateWrapper,

    # solver
    action,
    AlphaSchedule,
    InverseAlphaSchedule,
    ConstantAlphaSchedule,
    default_action,

    # visualization
    D3Tree,
    node_tag,
    tooltip_tag

include("types.jl")
include("updater.jl")
include("solver.jl")
include("rollout.jl")
include("visualization.jl")
# include("requirements_info.jl")

end # module
