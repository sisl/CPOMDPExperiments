using Pkg
Pkg.activate(".")
using Revise
Pkg.resolve()
using CPOMDPExperiments
using Plots
using D3Trees




#test("rocksample","pomcp")

#test("lightdark1d","pomcp-ow", test_pomdp=false)
#test("lightdark1d","pft-ow")

solver = CPOMDPExperiments.POMCPOWSolver(tree_queries=100000,
                                         criterion=CPOMDPExperiments.POMCPOW.MaxUCB(260.0),
                                         k_observation = 0.1,
                                         alpha_action = 01/35,
                                         estimate_value=0.0)
hist, R, C, RC = run_pomdp_simulation(CLightDark1D(), solver)

R
C
rewards = [h[:r] for h in hist]
states = [h[:s].y for h in hist]
beliefs = [h[:b] for h in hist]
as = [h[:a] for h in hist]


xpts = []
ypts = []
max_particles = 100
for i=1:length(beliefs) 
    count = 0
    for s in beliefs[i].particles
        push!(xpts, i)
        push!(ypts, s.y)
        count += 1
        if count > max_particles
            break
        end
    end
end

scatter(xpts, ypts)
scatter!(1:length(states), states)
savefig("belief.png")


inchrome(D3Tree(hist[2][:tree]))
solver._tree



avgR_unconstrained, avgC_unconstrained = 0, 0
avgR_constrained, avgC_constrained = 0, 0
for i=1:20
    kwargs = (;tree_queries=100000, 
        k_observation = 0.1,
        alpha_action = 01/35)
    c = 260.0
    println("i=$i")
    _, R, C, RC = run_pomdp_simulation(CLightDark1D(), CPOMDPExperiments.POMCPOWSolver(;kwargs..., criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), estimate_value=0.0))
    avgR_unconstrained += R
    avgC_unconstrained += C[1]
    _, R, C, RC = run_cpomdp_simulation(CLightDark1D(bad_action_budget=5.0), CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, 0.0), estimate_value=(args...)->(0.0, [0.0])))
    avgR_constrained += R
    avgC_constrained += C[1]
end
avgR_unconstrained / 20
avgC_unconstrained / 20
avgR_constrained / 20
avgC_constrained / 20


#test("vdptag", "pomcp-dpw", test_pomdp=false)
# test("vdptag","pft-dpw")
#test("vdptag","pomcpow)

#test("spillpoint", "pomcp-dpw", test_pomdp=false)
#test("spillpoint", "pft-dpw")
#test("spillpoint", "pomcpow")

#test("roomba","pft-dpw", test_cpomdp=false)
#run_all_tests()
