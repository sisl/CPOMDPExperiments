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


function get_tree(planner)
    if hasproperty(planner, :tree)
        return planner.tree
    elseif hasproperty(planner, :_tree)
        return planner._tree
    else
        @error "Can't find tree for planner of type $(typeof(planner))"
    end
end

function run_cpomdp_simulation(p::ConstrainPOMDPWrapper, solver::Solver, max_steps=100)
    planner = solve(solver, p)
    R = 0
    C = zeros(n_costs(p))
    RC = 0
    γ = 1
    tree_hist = Any[deepcopy(get_tree(planner))]
    #@infiltrate
    for (s, a, o, r, c,sp) in stepthrough(p, planner, "s,a,o,r,c,sp", max_steps=max_steps)
        R += POMDPs.reward(p.pomdp,s,a,sp) * γ
        C .+= c.*γ
        RC += γ*r # the reward already includes a -λc term. 

        γ *= discount(p)
        push!(tree_hist, deepcopy(get_tree(planner)))
    end
    tree_hist, R, C, RC
end

function run_pomdp_simulation(p::ConstrainPOMDPWrapper, solver::Solver, max_steps=100)
    planner = solve(solver, p.pomdp)
    R = 0
    C = zeros(n_costs(p))
    RC = 0
    γ = 1
    hist = []
    #@infiltrate
    for (s, a, o, r,sp,b) in stepthrough(p.pomdp, planner, "s,a,o,r,sp,b", max_steps=max_steps)
        c = costs(p,s,a,sp)
        R += r * γ
        C .+= c.*γ
        RC += γ*(r-p.λ⋅c) # the reward already includes a -λc term. 

        γ *= discount(p)

        push!(hist, (;s, a, o, r, sp, b, tree=deepcopy(get_tree(planner))))
    end
    hist, R, C, RC
end