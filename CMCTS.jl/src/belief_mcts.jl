"""
    BeliefCMCTSSolver(CMCTS_solver, updater)

The belief CMCTS solver solves POMDPs by modeling them as an MDP on the belief space. The `updater` is used to update the belief as part of the belief MDP generative model.

Example:

    using ParticleFilters
    using POMDPModels
    using CMCTS

    pomdp = BabyPOMDP()
    updater = SIRParticleFilter(pomdp, 1000)

    solver = BeliefCMCTSSolver(CDPWSolver(), updater)
    planner = solve(solver, pomdp)

    simulate(HistoryRecorder(max_steps=10), pomdp, planner, updater)
"""
mutable struct BeliefCMCTSSolver
    solver::AbstractCMCTSSolver
    updater::Updater
end

function POMDPs.solve(sol::BeliefCMCTSSolver, p::POMDP)
    bmdp = GenerativeBeliefMDP(p, sol.updater)
    return solve(sol.solver, bmdp)
end
