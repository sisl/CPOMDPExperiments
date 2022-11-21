struct CPOMCPBudgetUpdateWrapper <: Updater
    updater::Updater
    planner::AbstractCPOMCPPlanner
end

function update(up::CPOMCPBudgetUpdateWrapper, b, a, o)
    if up.planner._tree != nothing
        up.planner.budget = (up.planner.budget - up.planner._cost_mem)/discount(up.planner.problem)
    end
    return update(up.updater, b, a, o)
end

function updater(p::AbstractCPOMCPPlanner)
    P = typeof(p.problem)
    return CPOMCPBudgetUpdateWrapper(UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng),
        p)
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end

initialize_belief(bu::CPOMCPBudgetUpdateWrapper, dist) = initialize_belief(bu.updater, dist)

