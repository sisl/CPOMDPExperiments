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

function updater(p::CPOMCPPlanner)
    P = typeof(p.problem)
    return CPOMCPBudgetUpdateWrapper(UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng), p)
end

function updater(p::CPOMCPDPWPlanner)
    P = typeof(p.problem)
    return CPOMCPBudgetUpdateWrapper(BootstrapFilter(p.problem, 10*p.solver.tree_queries, p.solver.rng), p)
end

initialize_belief(bu::CPOMCPBudgetUpdateWrapper, dist) = initialize_belief(bu.updater, dist)

