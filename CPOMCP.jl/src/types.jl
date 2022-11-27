abstract type AlphaSchedule end
struct ConstantAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
ConstantAlphaSchedule() = ConstantAlphaSchedule(1.e-3)
alpha(sched::ConstantAlphaSchedule, ::Int) = sched.scale

struct InverseAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
InverseAlphaSchedule() = InverseAlphaSchedule(1.)
alpha(sched::InverseAlphaSchedule, query::Int) = sched.scale/query

abstract type AbstractCPOMCPSolver <: Solver end
abstract type AbstractCPOMCPTree end
abstract type AbstractCPOMCPPlanner{P,SE,RNG} <: Policy end

struct CPOMCPObsNode <: BeliefNode
    tree::AbstractCPOMCPTree
    node::Int
end
children(n::CPOMCPObsNode) = n.tree.children[n.node]
n_children(n::CPOMCPObsNode) = length(children(n))
isroot(n::CPOMCPObsNode) = n.node == 1

### CPOMCP Solver, Tree, Planner ###

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
    search_progress_info::Bool  = false
    return_best_cost::Bool  = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    alpha_schedule::AlphaSchedule    = InverseAlphaSchedule()
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng)) # (rng; max_depth=50, eps=nothing)
end


struct CPOMCPTree{A,O} <: AbstractCPOMCPTree
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

mutable struct CPOMCPPlanner{P, SE, RNG} <: AbstractCPOMCPPlanner{P,SE,RNG}
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
    se = convert_estimator(solver.estimate_value, solver, pomdp)
    return CPOMCPPlanner(solver, pomdp, se, solver.rng, 
        costs_limit(pomdp), Int[], nothing, nothing, nothing, costs_limit(pomdp))
end

Random.seed!(p::AbstractCPOMCPPlanner, seed) = Random.seed!(p.rng, seed)
solve(solver::CPOMCPSolver, pomdp::CPOMDP) = CPOMCPPlanner(solver, pomdp)

### CPOMCP-DPW Solver, Tree, Planner ###
@with_kw mutable struct CPOMCPDPWSolver <: AbstractCPOMCPSolver
    max_depth::Int          = 20
    c::Float64              = 1.0
    nu::Float64             = 0.01 # slack to give when searching for multiple best actions
    tree_queries::Int       = 1000
    
    enable_action_pw::Bool      = true
    enable_observation_pw::Bool = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true
    
    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0

    max_time::Float64       = Inf
    tree_in_info::Bool      = false
    search_progress_info::Bool  = false
    return_best_cost::Bool  = false
    default_action::Any     = ExceptionRethrow()
    rng::AbstractRNG        = Random.GLOBAL_RNG
    next_action::Any            = RandomActionGenerator(rng)
    alpha_schedule::AlphaSchedule    = InverseAlphaSchedule()
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng)) # (rng; max_depth=50, eps=nothing)
end


struct CPOMCPDPWTree{S,A,O} <: AbstractCPOMCPTree
    # for each observation-terminated history
    total_n::Vector{Int}                 # total number of visits for an observation node
    children::Vector{Vector{Int}}        # indices of each of the children
    o_labels::Vector{O}                  # actual observation corresponding to this observation node
    o_lookup::Dict{Tuple{Int,O}, Int}   # mapping from (action node index, observation) to an observation node index
    states::Vector{Vector{S}}           # vector of states at each observation node

    # for each action-terminated history
    n::Vector{Int}                       # number of visits for an action node
    v::Vector{Float64}                   # value estimate for an action node
    cv::Vector{Vector{Float64}}          # cost estimates for an action node
    a_labels::Vector{A}                  # actual action corresponding to this action node
    a_lookup::Dict{Tuple{Int,A},Int}     # map from (hnode,a)->hanode

    # transitions
    transitions::Vector{Vector{Int}} # map (hanode to possible haonode)
    n_a_children::Vector{Int} # number of children from each ha node
    unique_transitions::Set{Tuple{Int,Int}} # set of unique (sanode, spnode) transitions

    # constraints
    n_costs::Int                         # number of costs
    top_level_costs::Dict{Int,Vector{Float64}}    # top-level average cost (only if step)
end

function CPOMCPDPWTree(pomdp::CPOMDP, b, sz::Int=1000)
    S = statetype(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    sz = min(100_000, sz)
    return CPOMCPDPWTree{S,A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[[]], sz),
                          sizehint!(Array{O}(undef, 1), sz),
                          sizehint!(Dict{Tuple{Int,O},Int}(), sz),
                          sizehint!([Array{S}(undef, 1)], sz),

                          sizehint!(Int[], sz),
                          sizehint!(Float64[], sz),
                          sizehint!(Vector{Float64}[], sz), # cv
                          sizehint!(A[], sz),
                          Dict{Tuple{Int,A},Int}(), # a_lookup
                          
                          sizehint!(Vector{Int}[], sz),
                          sizehint!(Int[], sz), #n_a_children
                          Set{Tuple{Int,Int}}(),
                          
                          n_costs(pomdp),
                          Dict{Int,Vector{Float64}}() # top_level_costs
                         )
end

function insert_obs_node!(t::CPOMCPDPWTree{S}, pomdp::CPOMDP, ha::Int, sp::S, o) where {S}
    push!(t.total_n, 0)
    push!(t.states, S[sp])
    push!(t.children, Int[])
    push!(t.o_labels, o)
    hao = length(t.total_n)
    t.o_lookup[(ha, o)] = hao
    return hao
end

function insert_action_node!(t::CPOMCPDPWTree, h::Int, a;top_level=false)
    push!(t.n, 0)
    push!(t.v, 0.0)
    push!(t.a_labels, a)
    push!(t.cv, zeros(Float64, t.n_costs))
    push!(t.transitions, Vector{Float64}[])
    ha = length(t.n)
    push!(t.children[h], ha)
    push!(t.n_a_children, 0)
    t.a_lookup[(h, a)] = ha
    return ha
end

Base.isempty(tree::CPOMCPDPWTree) = isempty(tree.n) && isempty(tree.v)


mutable struct CPOMCPDPWPlanner{P, SE, RNG} <: AbstractCPOMCPPlanner{P,SE,RNG}
    solver::CPOMCPDPWSolver
    problem::P
    solved_estimator::SE
    rng::RNG
    next_action::Any
    budget::Vector{Float64}     # remaining budget for constraint search
    _best_node_mem::Vector{Int}
    _tree::Union{Nothing, Any}
    _cost_mem::Union{Nothing,Vector{Float64}}   # estimate for one-step cost
    _lambda::Union{Nothing,Vector{Float64}}    # weights for dual ascent
    _tau::Vector{Float64}       # clips for dual ascent
end

function CPOMCPDPWPlanner(solver::CPOMCPDPWSolver, pomdp::CPOMDP)
    se = convert_estimator(solver.estimate_value, solver, pomdp)
    return CPOMCPDPWPlanner(solver, pomdp, se, solver.rng, solver.next_action,
        costs_limit(pomdp), Int[], nothing, nothing, nothing, costs_limit(pomdp))
end

solve(solver::CPOMCPDPWSolver, pomdp::CPOMDP) = CPOMCPDPWPlanner(solver, pomdp)







