
mutable struct CRolloutEstimator
    solver::Union{Solver,Policy,Function} # rollout policy or solver
    max_depth::Union{Int, Nothing}
    eps::Union{Float64, Nothing}

    function CRolloutEstimator(solver::Union{Solver,Policy,Function};
                               max_depth::Union{Int, Nothing}=50,
                               eps::Union{Float64, Nothing}=nothing)
        new(solver, max_depth, eps)
    end
end

struct CPORollout
    solver::Union{POMDPs.Solver,POMDPs.Policy,Function}
    updater::POMDPs.Updater
end

struct SolvedCPORollout{P<:POMDPs.Policy,U<:POMDPs.Updater,RNG<:AbstractRNG}
    policy::P
    updater::U
    rng::RNG
end

struct CFORollout # fully observable rollout
    solver::Union{POMDPs.Solver,POMDPs.Policy}
end

struct SolvedCFORollout{P<:POMDPs.Policy,RNG<:AbstractRNG}
    policy::P
    rng::RNG
end

struct CFOValue
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

struct SolvedCFOValue{P<:POMDPs.Policy}
    policy::P
end

"""
    estimate_value(estimator, problem::POMDPs.POMDP, start_state, h::CBeliefNode, steps::Int)

Return an initial unbiased estimate of the value at belief node h.

By default this runs a rollout simulation
"""
function estimate_value end
estimate_value(f::Function, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) = f(pomdp, start_state, h, steps)
estimate_value(n::Number, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) = convert(Float64, n)

function estimate_value(estimator::Union{SolvedCPORollout,SolvedCFORollout}, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int)
    rollout(estimator, pomdp, start_state, h, steps)
end

function estimate_value(estimator::SolvedCFOValue, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int)
    POMDPs.value(estimator.policy, start_state)
end


function convert_estimator(ev::CRolloutEstimator, solver, pomdp)
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedCPORollout(policy, updater(policy), solver.rng)
end

function convert_estimator(ev::CPORollout, solver, pomdp)
    policy = MCTS.convert_to_policy(ev.solver, pomdp)
    SolvedCPORollout(policy, ev.updater, solver.rng)
end

function convert_estimator(est::CFORollout, solver, pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    SolvedCFORollout(policy, solver.rng)
end

function convert_estimator(est::CFOValue, solver, pomdp::CPOMDPs.CPOMDP)
    policy = MCTS.convert_to_policy(est.solver, UnderlyingMDP(pomdp))
    SolvedCFOValue(policy)
end


"""
Perform a rollout simulation to estimate the value.
"""
function rollout(est::SolvedCPORollout, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int)
    b = extract_belief(est.updater, h)
    sim = RolloutSimulator(est.rng,
                           steps)
    return POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end



function rollout(est::SolvedCFORollout, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int)
    sim = RolloutSimulator(est.rng,
                                        steps)
    return POMDPs.simulate(sim, pomdp, est.policy, start_state)
end


"""
    extract_belief(rollout_updater::POMDPs.Updater, node::CBeliefNode)

Return a belief compatible with the `rollout_updater` from the belief in `node`.

When a rollout simulation is started, this function is used to create the initial belief (compatible with `rollout_updater`) based on the appropriate `CBeliefNode` at the edge of the tree. By overriding this, a belief can be constructed based on the entire tree or entire observation-action history.
"""
#function extract_belief end

# some defaults are provided
# extract_belief(::NothingUpdater, node::CBeliefNode) = nothing
#
#function extract_belief(::PreviousObservationUpdater, node::CBeliefNode)
#    if node.node==1 && !isdefined(node.tree.o_labels, node.node)
#        missing
#    else
#        node.tree.o_labels[node.node]
#    end
#end
