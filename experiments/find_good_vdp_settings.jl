using Revise
using CPOMDPExperiments
using D3Trees
using Plots 

### Find Good Settings
kwargs = Dict(:tree_queries=>1e4,
        :k_action => 30.,
        :alpha_action => 1/30,
        :k_observation => 5.,
        :alpha_observation => 01/100,
        :max_depth => 30,
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(1e-2))
c = 110.0 
nu = 0.0
λ_test = [5.] # default meas_cost is 5., but set to 0. in CVDPTagPOMDP
filter_size = Int(1e4)

cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=100.);
    λ=λ_test)

solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zeroV_trueC,
    tree_in_info=true,
    search_progress_info=true)

updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
    planner)

hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

R3
C3[1]
RC3
plot_lightdark_beliefs(hist3,"figs/belief_vdp_unconstrained.png")

inchrome(D3Tree(hist3[1][:tree]))


sp3 = SearchProgress(hist3[1])

plot(sp3.v_best)
plot(sp3.cv_best)
plot(sp3.v_taken)
plot(sp3.cv_taken)
plot(sp3.lambda)


## constrained

cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=5.);
    λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zeroV_trueC,
    tree_in_info=true,
    search_progress_info=true)
updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
    planner)
hist4, R4, C4, RC4 = run_cpomdp_simulation(cpomdp, solver, updater)

R4
C4[1]
RC4
plot_lightdark_beliefs(hist4,"figs/belief_vdp_constrained.png")

inchrome(D3Tree(hist4[1][:tree]))


sp4 = SearchProgress(hist4[1])

plot(sp4.v_best)
plot(sp4.cv_best)
plot(sp4.v_taken)
plot(sp4.cv_taken)
plot(sp4.lambda)


