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

using CRockSample
using CPOMDPs
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
        simulate(sim,p,s)
    catch err
        println("Simulation $(fname) failed")
    end
end

function problem_test(p::POMDP, solver_func::Function, name::String)
    solver = solver_func(p)
    planner = solve(solver, p)

    # stepthrough
    for (s, a, o) in stepthrough(p, planner, "s,a,o", max_steps=10)
        println("State was $s,")
        println("action $a was taken,")
        println("and observation $o was received.\n")
    end
    
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