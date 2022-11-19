struct CPOMCPOWBudgetUpdateWrapper <: Updater
    updater::Updater
    planner::CPOMCPOWPlanner
end

function update(up::CPOMCPOWBudgetUpdateWrapper, b, a, o)
    if up.planner.tree != nothing
        up.planner.budget = (up.planner.budget - up.planner._cost_mem) / discount(up.planner.problem) # FIXME
    end
    return update(up.updater, b, a, o)
end

initialize_belief(bu::CPOMCPOWBudgetUpdateWrapper, dist) = initialize_belief(bu.updater, dist)

function updater(p::CPOMCPOWPlanner)
    P = typeof(p.problem)
    @assert P<:CPOMDP "planner problem not a CPOMDP"
    rng = MersenneTwister(rand(p.solver.rng, UInt32)) # how POMCPOW initializes updater but not sure why this over just passing the rng
    return CPOMCPOWBudgetUpdateWrapper(BootstrapFilter(p.problem, 10*p.solver.tree_queries, rng), p)
end


