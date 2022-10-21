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
using VDPTag2
using BasicPOMCP
using MCTS # belief-mcts for belief dpw
using POMCPOW

using CRockSample
using CPOMDPs
using CMCTS
using CPOMCP
using CRockSample


# pairs of pomdp, cpomdp models
models = [
    (
        "rocksample",
        RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
            sensor_efficiency=20.0,
            discount_factor=0.95, 
            good_rock_reward = 20.0),
        RockSampleCPOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
            sensor_efficiency=20.0,
            discount_factor=0.95, 
            good_rock_reward = 20.0),
    ),
    (
        "vdptag",
        VDPTagPOMDP(),
        VDPTagPOMDP(),
    ),
    (
        "lightdark1d",
        LightDark1D(),
        LightDark1D(),
    ),
    (
        "roomba",
        RoombaPOMDP(),
        RoombaPOMDP(),
    ),
]

# pairs of pomdp, cpomdp solvers
solvers = [
    ("pomcp",
        POMCPSolver(tree_queries=10000, c=10), 
        POMCPOWSolver(tree_queries=10000, c=10)), #POMCP
    ("pft", 
        BeliefMCTSSolver(DPWSolver(), SIRParticleFilter(pomdp, 1000)), #FIXME
        BeliefCMCTSSolver(CDPWSolver(), SIRParticleFilter(cpomdp, 1000))), #FIXME #PFT-DPW
    ("pomcpow", 
        POMCPOWSolver(criterion=MaxUCB(20.0)), 
        CPOMCPOWSolver(criterion=MaxUCB(20.0))), # POMCPOW
]

### pomdp test

for (pname, pomdp, cpomdp) in models
    for (sname, pomdp_solver, cpomdp_solver) in solvers
        
        ### POMDP
        println("Testing POMDP $(pname) with solver $(sname)")
        pomdp_planner = solve(pomdp_solver, pomdp)

        filter = SIRParticleFilter(pomdp, 1000)
        for (s, a, o) in stepthrough(pomdp, pomdp_planner, filter, "s,a,o", max_steps=10)
            println("State was $s,")
            println("action $a was taken,")
            println("and observation $o was received.\n")
        end

        # save gif
        sim_pomdp = GifSimulator(filename="test_pomdp_$(pname)_$(sname).gif", max_steps=30)
        simulate(sim_pomdp, pomdp, pomdp_planner)

        ### CPOMDP
        println("Testing CPOMDP $(pname) with solver $(sname)")
        cpomdp_planner = solve(cpomdp_solver, cpomdp)

        filter = SIRParticleFilter(cpomdp, 1000)
        for (s, a, o) in stepthrough(cpomdp, cpomdp_planner, filter, "s,a,o", max_steps=10)
            println("State was $s,")
            println("action $a was taken,")
            println("and observation $o was received.\n")
        end

        # save gif
        sim_cpomdp = GifSimulator(filename="test_cpomdp_$(pname)_$(sname).gif", max_steps=30)
        simulate(sim_cpomdp, cpomdp, cpomdp_planner)
    end
end

end # module