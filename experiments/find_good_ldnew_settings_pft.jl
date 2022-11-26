using Revise
using CPOMDPExperiments
using D3Trees
using Plots 

### Find Good Settings
kwargs = Dict(:tree_queries=>1e6, 
        :k_observation => 0.1,
        :alpha_observation => 0.5,
        :enable_action_pw=>false,
        :max_depth => 10,
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(1e-1))
c = 250.0 # 250
nu = 0.0
λ_test = [1.]


cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=20.);λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zeroV_trueC,
    tree_in_info=true,
    search_progress_info=true)
updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
    planner)
hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

R3
C3[1]
RC3
plot_lightdark_beliefs(hist3,"figs/belief_ldn_unconstrained.png")

inchrome(D3Tree(hist3[1][:tree]))


sp3 = SearchProgress(hist3[1])

#plot(sp.v_best)
#plot(sp.cv_best)
#plot(sp.v_taken)
#plot(sp.cv_taken)
#plot(sp.lambda)


## constrained

cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.5);λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zeroV_trueC,
    tree_in_info=true,
    search_progress_info=true)
updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
    planner)
hist4, R4, C4, RC4 = run_cpomdp_simulation(cpomdp, solver, updater)

R4
C4[1]
RC4
plot_lightdark_beliefs(hist4,"figs/belief_ldn_constrained.png")

inchrome(D3Tree(hist4[1][:tree]))


sp4 = SearchProgress(hist4[1])

#plot(sp.v_best)
#plot(sp.cv_best)
#plot(sp.v_taken)
#plot(sp.cv_taken)
#plot(sp.lambda)


