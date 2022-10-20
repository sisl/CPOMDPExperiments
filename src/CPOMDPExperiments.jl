module CPOMDPExperiments

using CRockSample
using CPOMDPs
using CMCTS
using CPOMCP
using CRockSample

using POMDPSimulators
using POMDPGifs
using Cairo

pomdp = RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
                        sensor_efficiency=20.0,
                        discount_factor=0.95, 
                        good_rock_reward = 20.0)

solver = POMCPSolver()
planner = solve(solver, cpomdp)

# manual printing
for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end

# save gif
sim = GifSimulator(filename="test.gif", max_steps=30)
simulate(sim, pomdp, policy)

end # module