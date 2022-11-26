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

const init_V = init_Q
function init_C end
init_C(f::Function, mdp::Union{CMDP,CPOMDP}, s, a) = f(mdp, s, a)
init_C(n::Vector{Number}, mdp::Union{CMDP,CPOMDP}, s, a) = convert.(Float64, n)
init_C(n::Number, mdp::Union{CMDP,CPOMDP}, s, a) = convert(Float64, n) * ones(Float64, n_costs(mdp)) # set to uniform n

### Solver ### 
"""
    CPOMCPOWSolver

Partially observable Monte Carlo planning with observation widening.

Fields:

- `eps::Float64`:
    Rollouts and tree expansion will stop when discount^depth is less than this.
    default: `0.01`
- `max_depth::Int`:
    Rollouts and tree expension will stop when this depth is reached.
    default: `10`
- `criterion::Any`:
    Criterion to decide which action to take at each node. e.g. `MaxUCB(c)`, `MaxQ`, or `MaxTries`
    default: `MaxUCB(1.0)`
- `final_criterion::Any`:
    Criterion for choosing the action to take after the tree is constructed.
    default: `MaxQ()`
- `tree_queries::Int`:
    Number of iterations during each action() call.
    default: `1000`
- `max_time::Float64`:
    Time limit for planning at each steps (seconds).
    default: `Inf`
- `rng::AbstractRNG`:
    Random number generator.
    default: Base.GLOBAL_RNG
- `node_sr_belief_updater::Updater`:
    Updater for state-reward distribution at the nodes.
    default: `POWNodeFilter()`
- `estimate_value::Any`: (rollout policy can be specified by setting this to RolloutEstimator(policy))
    Function, object, or number used to estimate the value at the leaf nodes.
    If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    If this is a number, the value will be set to that number
    default: `RolloutEstimator(RandomSolver(rng))`
- `enable_action_pw::Bool`:
    Controls whether progressive widening is done on actions; if `false`, the entire action space is used.
    default: `true`
- `check_repeat_obs::Bool`:
    Check if an observation was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same observation from being created. If the observation space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `check_repeat_act::Bool`:
    Check if an action was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same action from being created. If the action space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `tree_in_info::Bool`:
    If `true`, return the tree in the info dict when action_info is called, this can use a lot of memory if histories are being saved.
    default: `false`
- `k_action::Float64`, `alpha_action::Float64`, `k_observation::Float64`, `alpha_observation::Float64`:
        These constants control the double progressive widening. A new observation
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k: `10`, alpha: `0.5`
- `init_V::Any`:
    Function, object, or number used to set the initial V(h,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_V(o, pomdp, h, a)` will be called.
    If this is a number, V will be set to that number
    default: `0.0`
- `init_N::Any`:
    Function, object, or number used to set the initial N(s,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_N(o, pomdp, h, a)` will be called.
    If this is a number, N will be set to that number
    default: `0`
- `next_action::Any`
    Function or object used to choose the next action to be considered for progressive widening.
    The next action is determined based on the POMDP, the belief, `b`, and the current `BeliefNode`, `h`.
    If this is a function `f`, `f(pomdp, b, h)` will be called to set the value.
    If this is an object `o`, `next_action(o, pomdp, b, h)` will be called.
    default: `RandomActionGenerator(rng)`
- `default_action::Any`:
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
    If this is a Policy `p`, `action(p, belief)` will be called.
    If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and
    if this method is not implemented, `a` will be returned directly.
- `timer::Function`:
    Timekeeping method. Search iterations ended when `timer() - start_time ≥ max_time`.
"""
@with_kw mutable struct CPOMCPOWSolver{RNG<:AbstractRNG,T} <: Solver
    eps::Float64                = 0.01
    max_depth::Int              = typemax(Int)
    criterion                   = MaxCUCB(1.0,0.0) # c,nu
    final_criterion             = MaxCUCB(0.0,0.1) # c,nu
    tree_queries::Int           = 1000
    max_time::Float64           = Inf
    rng::RNG                    = Random.GLOBAL_RNG
    node_sr_belief_updater      = CPOWNodeFilter()

    estimate_value::Any         = RolloutEstimator(RandomSolver(rng))

    alpha_schedule::AlphaSchedule = InverseAlphaSchedule()

    enable_action_pw::Bool      = true
    check_repeat_obs::Bool      = true
    check_repeat_act::Bool      = true
    tree_in_info::Bool          = false
    search_progress_info::Bool  = false

    alpha_observation::Float64  = 0.5
    k_observation::Float64      = 10.0
    alpha_action::Float64       = 0.5
    k_action::Float64           = 10.0
    init_V::Any                 = 0.0
    init_N::Any                 = 0
    init_C::Any                 = 0.
    return_best_cost::Bool      = false # if true, simulate best costs up tree 
    next_action::Any            = RandomActionGenerator(rng)
    default_action::Any         = ExceptionRethrow()
    timer::T                    = () -> 1e-9*time_ns()
end

# unweighted ParticleCollections don't get anything pushed to them
function push_weighted!(::ParticleCollection, sp) end

### Tree ###
struct CPOMCPOWTree{B,A,O,RB}
    # action nodes
    n::Vector{Int}
    v::Vector{Float64}
    cv::Vector{Vector{Float64}}
    generated::Vector{Vector{Pair{O,Int}}} #triple? is this obs,reward or obs, sp
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A}
    n_a_children::Vector{Int}

    # observation nodes
    sr_beliefs::Vector{B} # first element is #undef
    total_n::Vector{Int}
    tried::Vector{Vector{Int}}
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params
    o_labels::Vector{O}

    # root
    root_belief::RB
    top_level_costs::Dict{Int,Vector{Float64}}
    n_costs::Int


    function CPOMCPOWTree{B,A,O,RB}(root_belief, sz::Int=1000, n_costs::Int=1) where{B,A,O,RB}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Int[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Vector{Vector{Float64}}[], sz), #qc
            sizehint!(Vector{Pair{O,Int}}[], sz),
            Dict{Tuple{Int,O}, Int}(),
            sizehint!(A[], sz),
            sizehint!(Int[], sz),

            sizehint!(Array{B}(undef, 1), sz),
            sizehint!(Int[0], sz),
            sizehint!(Vector{Int}[Int[]], sz),
            Dict{Tuple{Int,A}, Int}(),
            sizehint!(Array{O}(undef, 1), sz),

            root_belief,
            Dict{Int,Vector{Float64}}(),
            n_costs
        )
    end
end

@inline function push_anode!(tree::CPOMCPOWTree{B,A,O}, h::Int, a::A, n::Int=0, v::Float64=0.0, cv::Union{Vector{Float64},Nothing}=nothing,update_lookup=true) where {B,A,O}
    if cv == nothing
        cv=zeros(Float64,tree.n_costs)
    end
    anode = length(tree.n) + 1
    push!(tree.n, n)
    push!(tree.v, v)
    push!(tree.cv, cv)
    push!(tree.generated, Pair{O,Int}[])
    push!(tree.a_labels, a)
    push!(tree.n_a_children, 0)
    if update_lookup
        tree.o_child_lookup[(h, a)] = anode
    end
    push!(tree.tried[h], anode)
    tree.total_n[h] += n
    return anode
end

struct CPOWTreeObsNode{B,A,O,RB} <: BeliefNode
    tree::CPOMCPOWTree{B,A,O,RB}
    node::Int
end

isroot(h::CPOWTreeObsNode) = h.node==1
@inline function belief(h::CPOWTreeObsNode)
    if isroot(h)
        return h.tree.root_belief
    else
        return CStateBelief(h.tree.sr_beliefs[h.node])
    end
end
function sr_belief(h::CPOWTreeObsNode)
    if isroot(h)
        error("Tried to access the sr_belief for the root node in a POMCPOW tree")
    else
        return h.tree.sr_beliefs[h.node]
    end
end
n_children(h::CPOWTreeObsNode) = length(h.tree.tried[h.node])

### Planner ###
mutable struct CPOMCPOWPlanner{P,NBU,C,NA,SE,IN,IV,IC,SolverType} <: Policy
    solver::SolverType
    problem::P
    node_sr_belief_updater::NBU
    criterion::C
    next_action::NA
    solved_estimate::SE
    init_N::IN
    init_V::IV
    init_C::IC
    tree::Union{Nothing, CPOMCPOWTree} # this is just so you can look at the tree later
    budget::Vector{Float64}
    _cost_mem::Union{Nothing,Vector{Float64}}
    _lambda::Union{Nothing,Vector{Float64}}
    _tau::Vector{Float64}

end

function CPOMCPOWPlanner(solver, problem::CPOMDP)
    CPOMCPOWPlanner(solver,
                  problem,
                  solver.node_sr_belief_updater,
                  solver.criterion,
                  solver.next_action,
                  convert_estimator(solver.estimate_value, solver, problem),
                  solver.init_N,
                  solver.init_V,
                  solver.init_C,
                  nothing,
                  costs_limit(problem),
                  nothing,
                  nothing,
                  costs_limit(problem))
end

Random.seed!(p::CPOMCPOWPlanner, seed) = Random.seed!(p.solver.rng, seed)


