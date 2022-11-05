function POMDPLinter.requirements_info(solver::AbstractCPOMCPSolver, problem::CPOMDP)
    println("""
    Since CPOMCP is an online solver, most of the computation occurs in `action(planner, state)`. In order to view the requirements for this function, please, supply an initial beleif to `requirements_info`, e.g.

            @requirements_info $(typeof(solver))() $(typeof(problem))() initialstate(cpomdp)

        """)
end

function POMDPLinter.requirements_info(solver::AbstractCPOMCPSolver, problem::CPOMDP, b)
    policy = solve(solver, problem)
    requirements_info(policy, b)
end

POMDPs.requirements_info(policy::CPOMCPPlanner, b) = @show_requirements action(policy, b)

@POMDP_require action(p::CPOMCPPlanner, b) begin
    tree = CPOMCPTree(p.problem, b, p.solver.tree_queries)
    @subreq search(p, b, tree)
end

@POMDP_require search(p::CPOMCPPlanner, b, t::CPOMCPTree) begin
    P = typeof(p.problem)
    @req rand(::typeof(p.rng), ::typeof(b))
    s = rand(p.rng, b)
    @req isterminal(::P, ::statetype(P))
    @subreq simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
end

@POMDP_require simulate(p::CPOMCPPlanner, s, hnode::POMCPObsNode, steps::Int) begin
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
    # @subreq estimate_value(p.solved_estimator, p.problem, s, hnode, steps) # FIXME
    @req discount(::P)
end

@POMDP_require estimate_value(estimator::Union{SolvedCPORollout,SolvedCFORollout}, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) begin
    @subreq rollout(estimator, pomdp, start_state, h, steps)
end

@POMDP_require rollout(est::SolvedCPORollout, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) begin
    @req extract_belief(::typeof(est.updater), ::typeof(h))
    b = extract_belief(est.updater, h)
    sim = RolloutSimulator(est.rng,
                           steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

@POMDP_require rollout(est::SolvedCFORollout, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) begin
    sim = RolloutSimulator(est.rng,
                                        steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, start_state)
end

@POMDP_require estimate_value(f::Function, pomdp::CPOMDPs.CPOMDP, start_state, h::CBeliefNode, steps::Int) begin
    @req f(::typeof(pomdp), ::typeof(start_state), ::typeof(h), ::typeof(steps))
end