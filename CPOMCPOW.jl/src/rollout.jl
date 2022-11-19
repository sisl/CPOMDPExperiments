"""
    estimate_value(estimator, problem::POMDPs.POMDP, start_state, h::CBeliefNode, steps::Int)

Return an initial unbiased estimate of the value at belief node h.

By default this runs a rollout simulation
"""
function estimate_value(estimator::BasicPOMCP.SolvedFOValue, pomdp::CPOMDPs.CPOMDP, start_state, h::BeliefNode, steps::Int)
    return POMDPs.value(estimator.policy, start_state), CPOMDPs.cost_value(estimator.policy, start_state)
end


"""
Perform a rollout simulation to estimate the value.
"""
function rollout(est::BasicPOMCP.SolvedPORollout, pomdp::CPOMDP, start_state, h::BeliefNode, steps::Int)
    b = extract_belief(est.updater, h)
    sim = ConstrainedRolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

@POMDP_require rollout(est::BasicPOMCP.SolvedPORollout, pomdp::CPOMDP, start_state, h::BeliefNode, steps::Int) begin
    @req extract_belief(::typeof(est.updater), ::typeof(h))
    b = extract_belief(est.updater, h)
    sim = ConstrainedRolloutSimulator(est.rng, steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

function rollout(est::BasicPOMCP.SolvedFORollout, pomdp::CPOMDP, start_state, h::BeliefNode, steps::Int)
    sim = ConstrainedRolloutSimulator(est.rng, steps)
    return simulate(sim, pomdp, est.policy, start_state)
end

@POMDP_require rollout(est::BasicPOMCP.SolvedFORollout, pomdp::CPOMDP, start_state, h::BeliefNode, steps::Int) begin
    sim = ConstrainedRolloutSimulator(est.rng, steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, start_state)
end