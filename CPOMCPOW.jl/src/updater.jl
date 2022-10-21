function POMDPs.updater(p::CPOMCPOWPlanner)
    rng = MersenneTwister(rand(p.solver.rng, UInt32))
    return BootstrapFilter(p.problem, 10*p.solver.tree_queries, rng)
end
