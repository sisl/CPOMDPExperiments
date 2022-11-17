@POMDP_require estimate_value(estimator::SolvedRolloutEstimator, mdp::CMDP, state, remaining_depth) begin
    sim = ConstrainedRolloutSimulator(rng=estimator.rng, max_steps=estimator.max_depth, eps=estimator.eps)
    @subreq POMDPs.simulate(sim, mdp, estimator.policy, s)
end

# rollouts occur here
function estimate_value(estimator::SolvedRolloutEstimator, mdp::CMDP, state, remaining_depth)
    if estimator.max_depth == -1
        max_steps = remaining_depth
    else
        max_steps = estimator.max_depth
    end
    sim = ConstrainedRolloutSimulator(rng=estimator.rng, max_steps=max_steps, eps=estimator.eps)
    return POMDPs.simulate(sim, mdp, estimator.policy, state)
end

"""
    init_Qc(initializer, mdp, s, a)
Return a value to initialize Qc(s,a) to based on domain knowledge.
"""
function init_Qc end
init_Qc(f::Function, mdp::Union{CMDP,CPOMDP}, s, a) = f(mdp, s, a)
init_Qc(n::Vector{Number}, mdp::Union{CMDP,CPOMDP}, s, a) = convert.(Float64, n)
init_Qc(n::Number, mdp::Union{CMDP,CPOMDP}, s, a) = convert(Float64, n) * ones(Float64, n_costs(mdp)) # set to uniform n