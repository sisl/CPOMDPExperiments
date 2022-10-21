function POMDPLinter.requirements_info(solver::AbstractCPOMCPSolver, problem::POMDP)
    println("""
    Since CPOMCP is an online solver, most of the computation occurs in `action(planner, state)`. In order to view the requirements for this function, please, supply an initial beleif to `requirements_info`, e.g.

            @requirements_info $(typeof(solver))() $(typeof(problem))() initialstate(pomdp)

        """)
end

function POMDPLinter.requirements_info(solver::AbstractCPOMCPSolver, problem::POMDP, b)
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
    @subreq simulate(p, s, CPOMCPObsNode(t, 1), p.solver.max_depth)
end

@POMDP_require simulate(p::CPOMCPPlanner, s, hnode::CPOMCPObsNode, steps::Int) begin
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
end

@POMDP_require estimate_value(f::Function, pomdp::POMDPs.POMDP, start_state, h::BeliefNode, steps::Int) begin
    @req f(::typeof(pomdp), ::typeof(start_state), ::typeof(h), ::typeof(steps))
end
