function POMDPs.updater(p::AbstractMCTSPlanner)
    P = typeof(p.mdp)
    @assert P <: GenerativeBeliefMDP "updater called on a AbstractCMCTSPlanner without an underlying BeliefMDP"
    return p.mdp.updater
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end

function test(model::String, solver::String)
    
    println("Testing POMDP $(model) with solver $(solver)")
    problem_test(models[model][1], solvers[solver][1], "test_pomdp_$(model)_$(solver)")

    println("Testing CPOMDP $(model) with solver $(solver)")
    problem_test(models[model][2], solvers[solver][2], "test_cpomdp_$(model)_$(solver)")
end

function generate_gif(p::POMDP, s, fname::String)
    try
        sim = GifSimulator(filename=fname, max_steps=30)
        simulate(sim, p, s)
    catch err
        println("Simulation $(fname) failed")
    end
end

function step_through(p::POMDP, planner::Policy, max_steps=100)
    for (s, a, o, r) in stepthrough(p, planner, "s,a,o,r", max_steps=max_steps)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r.")
    end
end

function step_through(p::CPOMDP, planner::Policy, max_steps=100)
    #@infiltrate
    for (s, a, o, r, c) in stepthrough(p, planner, "s,a,o,r,c", max_steps=max_steps)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r, ")
        println("Cost: $c.")
    end
end


function problem_test(p::Union{POMDP,CPOMDP}, solver_func::Function, name::String)
    solver = solver_func(p)
    planner = solve(solver, p)

    # stepthrough
    step_through(p,planner)

    
    # gif
    generate_gif(p,planner,name)

    return p, planner
end

function run_all_tests()
    for (m,s) in EXPERIMENTS
        test(m,s)
    end
end