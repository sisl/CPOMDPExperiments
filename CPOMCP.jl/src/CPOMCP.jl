module CPOMCP

#=
Current constraints:
- action space discrete
- action space same for all states, histories
- no built-in support for history-dependent rollouts (this could be added though)
- initial n and initial v are 0
=#

using POMDPs
using BasicPOMCP

using CPOMDPs
using Infiltrator
using Parameters
using ParticleFilters
using CPUTime
using Colors
using Random
using Printf
using POMDPLinter
using POMDPTools

import POMDPs: action, solve, updater, simulate, update
import POMDPLinter: @POMDP_require, @show_requirements

using MCTS
import MCTS: convert_estimator, estimate_value, node_tag, tooltip_tag, default_action
import BasicPOMCP: estimate_value, rollout, extract_belief

using D3Trees

export
    # here
    AbstractCPOMCPSolver,
    CPOMCPSolver,
    CPOMCPPlanner,
    updater,
    update,
    solve,

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

abstract type AbstractCPOMCPSolver <: Solver end
abstract type AlphaSchedule end

"""
    CPOMCPSolver(#=keyword arguments=#)

Partially Observable Monte Carlo Planning Solver.

## Keyword Arguments

- `max_depth::Int`
    Rollouts and tree expension will stop when this depth is reached.
    default: `20`

- `c::Float64`
    UCB exploration constant - specifies how much the solver should explore.
    default: `1.0`

- `tree_queries::Int`
    Number of iterations during each action() call.
    default: `1000`

- `max_time::Float64`
    Maximum time for planning in each action() call.
    default: `Inf`

- `tree_in_info::Bool`
    If `true`, returns the tree in the info dict when action_info is called.
    default: `false`

- `estimate_value::Any`
    Function, object, or number used to estimate the value at the leaf nodes.
    default: `RolloutEstimator(RandomSolver(rng))`
    - If this is a function `f`, `f(pomdp, s, h::CBeliefNode, steps)` will be called to estimate the value.
    - If this is an object `o`, `estimate_value(o, pomdp, s, h::CBeliefNode, steps)` will be called.
    - If this is a number, the value will be set to that number
    Note: In many cases, the simplest way to estimate the value is to do a rollout on the fully observable MDP with a policy that is a function of the state. To do this, use `CFORollout(policy)`.

- `default_action::Any`
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    default: `ExceptionRethrow()`
    - If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
    - If this is a Policy `p`, `action(p, belief)` will be called.
    - If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.

- `rng::AbstractRNG`
    Random number generator.
    default: `Random.GLOBAL_RNG`
"""
@with_kw mutable struct CPOMCPSolver <: AbstractCPOMCPSolver
    max_depth::Int          = 20
    c::Float64              = 1.0
    nu::Float64             = 0.01 # slack to give when searching for multiple best actions
    tree_queries::Int       = 1000
    max_time::Float64       = Inf
    tree_in_info::Bool      = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    alpha_schedule::AlphaSchedule    = InverseAlphaSchedule()
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng)) # (rng; max_depth=50, eps=nothing)
end

struct CPOMCPTree{A,O}
    # for each observation-terminated history
    total_n::Vector{Int}                 # total number of visits for an observation node
    children::Vector{Vector{Int}}        # indices of each of the children
    o_labels::Vector{O}                  # actual observation corresponding to this observation node

    o_lookup::Dict{Tuple{Int, O}, Int}   # mapping from (action node index, observation) to an observation node index

    # for each action-terminated history
    n::Vector{Int}                       # number of visits for an action node
    v::Vector{Float64}                   # value estimate for an action node
    cv::Vector{Vector{Float64}}          # cost estimates for an action node
    a_labels::Vector{A}                  # actual action corresponding to this action node

    # constraints
    n_costs::Int                         # number of costs
    top_level_costs::Vector{Vector{Float64}}    # top-level average cost (only if step)
end

function CPOMCPTree(pomdp::CPOMDP, b, sz::Int=1000)
    acts = collect(actions(pomdp, b))
    cons = n_costs(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    sz = min(100_000, sz)
    return CPOMCPTree{A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                          sizehint!(Array{O}(undef, 1), sz),

                          sizehint!(Dict{Tuple{Int,O},Int}(), sz),

                          sizehint!(zeros(Int, length(acts)), sz),
                          sizehint!(zeros(Float64, length(acts)), sz),
                          sizehint!(repeat([zeros(Float64,cons)], length(acts)), sz), # cv
                          sizehint!(acts, sz),

                          cons,
                          repeat([zeros(Float64,cons)], length(acts)) # top_level_costs
                         )
end

function insert_obs_node!(t::CPOMCPTree, pomdp::CPOMDP, ha::Int, sp, o)
    acts = actions(pomdp, LeafNodeBelief(tuple((a=t.a_labels[ha], o=o)), sp))
    push!(t.total_n, 0)
    push!(t.children, sizehint!(Int[], length(acts)))
    push!(t.o_labels, o)
    hao = length(t.total_n)
    t.o_lookup[(ha, o)] = hao
    for a in acts
        n = insert_action_node!(t, hao, a)
        push!(t.children[hao], n)
    end
    return hao
end

function insert_action_node!(t::CPOMCPTree, h::Int, a)
    push!(t.n, 0)
    push!(t.v, 0.0)
    push!(t.a_labels, a)
    push!(t.cv, zeros(Float64, t.n_costs))
    return length(t.n)
end

struct CPOMCPObsNode{A,O} <: BeliefNode
    tree::CPOMCPTree{A,O}
    node::Int
end

mutable struct CPOMCPPlanner{P, SE, RNG} <: Policy
    solver::CPOMCPSolver
    problem::P
    solved_estimator::SE
    rng::RNG
    budget::Vector{Float64}     # remaining budget for constraint search
    _best_node_mem::Vector{Int}
    _tree::Union{Nothing, Any}
    _cost_mem::Union{Nothing,Vector{Float64}}   # estimate for one-step cost
    _lambda::Union{Nothing,Vector{Float64}}    # weights for dual ascent
    _tau::Vector{Float64}       # clips for dual ascent
end

function CPOMCPPlanner(solver::CPOMCPSolver, pomdp::CPOMDP)
    se = convert_estimator(solver.estimate_value, solver, pomdp) # FIXME??
    return CPOMCPPlanner(solver, pomdp, se, solver.rng, 
        costs_limit(pomdp), Int[], nothing, nothing, nothing, costs_limit(pomdp))
end

solve(solver::CPOMCPSolver, pomdp::CPOMDP) = CPOMCPPlanner(solver, pomdp)

Random.seed!(p::CPOMCPPlanner, seed) = Random.seed!(p.rng, seed)

struct BudgetUpdateWrapper <: Updater
    belief_updater::Updater
    planner::CPOMCPPlanner
end

function update(up::BudgetUpdateWrapper, b, a, o)
    if up.planner._tree != nothing
        up.planner.budget = (up.planner.budget - up.planner._cost_mem)/discount(up.planner.problem)
    end
    return update(up.belief_updater, b, a, o)
end

function updater(p::CPOMCPPlanner)
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    # p.budget = (p.budget - c)/discount(p.problem)
    return BudgetUpdateWrapper(UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng),
        p)
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end

include("solver.jl")
include("rollout.jl")
include("visualization.jl")
# include("requirements_info.jl")

end # module
