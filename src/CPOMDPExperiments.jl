module CPOMDPExperiments

using ProgressMeter
using POMDPSimulators
using POMDPGifs
using Cairo
using ParticleFilters

# non-constrained baseline
using POMDPs
using POMDPModels
using RockSample
#using VDPTag2
using RoombaPOMDPs
using BasicPOMCP
using MCTS # belief-mcts for belief dpw
using POMCPOW

using CPOMDPs
using CRockSample
using CMCTS
using CPOMCP
#using CPOMCPOW
using CRockSample

export
    test,
    run_all_tests

include("configs.jl")

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
    for (s, a, o, r) in stepthrough(p, planner, "s,a,o,r", max_steps=100)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r.")
    end
end

# FIXME: fix simulators to add c
function step_through(p::CPOMDP, planner::Policy, max_steps=100)
    for (s, a, o, r) in stepthrough(p, planner, "s,a,o,r", max_steps=100)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r, ")
        #println("Cost: $c.")
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
    for m in MODELS
        for s in SOLVERS
            test(m,s)
        end
    end
end

end # module