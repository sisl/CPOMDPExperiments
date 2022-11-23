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

function run_cpomdp_simulation(p::SoftConstraintPOMDPWrapper, solver::Solver, max_steps=100)
    planner = solve(solver, p.cpomdp)
    R = 0
    C = zeros(n_costs(p.cpomdp))
    RC = 0
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, o, r, c, sp, b, ai) in stepthrough(p.cpomdp, planner, "s,a,o,r,c,sp,b,action_info", max_steps=max_steps)
        
        # track fictitions augmented reward
        rc = r - p.λ⋅c
        
        R += r*γ
        C .+= c.*γ
        RC += rc*γ 

        γ *= discount(p)

        push!(hist, (;s, a, o, r, c, rc, sp, b, 
            tree = :tree in keys(ai) ? ai[:tree] : nothing,
            v_best = :v_best in keys(ai) ? ai[:v_best] : nothing,
            cv_best = :cv_best in keys(ai) ? ai[:cv_best] : nothing,
            v_taken = :v_taken in keys(ai) ? ai[:v_taken] : nothing,
            cv_taken = :cv_taken in keys(ai) ? ai[:cv_taken] : nothing,
            ))
    end
    hist, R, C, RC
end

function run_pomdp_simulation(p::SoftConstraintPOMDPWrapper, solver::Solver, max_steps=100)
    planner = solve(solver, p)
    
    R = 0
    C = zeros(n_costs(p.cpomdp))
    RC = 0
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, o, r, sp, b, ai) in stepthrough(p, planner, "s,a,o,r,sp,b,action_info", max_steps=max_steps)
        
        # the tracked reward is actually augmented reward, backtrack true reward and costs 
        rc = r
        r = reward(p.pomdp,s,a,sp)
        c = costs(p.cpomdp,s,a,sp)

        R += r*γ
        C .+= c.*γ
        RC += rc*γ 

        γ *= discount(p)

        push!(hist, (;s, a, o, r, c, rc, sp, b, 
            tree = :tree in keys(ai) ? ai[:tree] : nothing))
    end
    hist, R, C, RC
end