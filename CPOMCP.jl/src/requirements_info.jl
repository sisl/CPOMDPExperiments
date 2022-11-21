function POMDPLinter.requirements_info(solver::AbstractCPOMCPSolver, problem::CPOMDP, b)
    policy = solve(solver, problem)
    requirements_info(policy, b)
end

POMDPs.requirements_info(policy::AbstractCPOMCPPlanner, b) = @show_requirements action(policy, b)

@POMDP_require action(p::CPOMCPPlanner, b) begin
    tree = CPOMCPTree(p.problem, b, p.solver.tree_queries)
    @subreq search(p, b, tree)
end

@POMDP_require action(p::CPOMCPDPWPlanner, b) begin
    tree = CPOMCPDPWTree(p.problem, b, p.solver.tree_queries)
    @subreq search(p, b, tree)
end

@POMDP_require search(p::AbstractCPOMCPPlanner, b, t::AbstractCPOMCPTree) begin
    P = typeof(p.problem)
    @req rand(::typeof(p.rng), ::typeof(b))
    s = rand(p.rng, b)
    @req isterminal(::P, ::statetype(P))
    @subreq simulate(p, s, CPOMCPObsNode(t, 1), p.solver.max_depth)
end

@POMDP_require simulate(p::AbstractCPOMCPPlanner, s, hnode::CPOMCPObsNode, steps::Int) begin
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    @req gen(::P, ::S, ::A, ::typeof(p.rng))
    @req isequal(::O, ::O)
    @req hash(::O)
    # from insert_obs_node!
    @req actions(::P)
    AS = typeof(actions(p.problem))
    @req length(::AS)
    @subreq estimate_value(p.solved_estimator, p.problem, s, hnode, steps)
    @req discount(::P)
    @req n_costs(::P)
    @req cost_limits(::P)
end