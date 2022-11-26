using Revise
Pkg.resolve()
using CPOMDPExperiments
using D3Trees
using Plots 

### Find Good Settings
kwargs = Dict(:tree_queries=>100000, 
        :k_observation => 0.1,
        :alpha_action => 01/100,
        :max_depth => 10,
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(1e-4))
c = 250.0 # 250
nu = 0.0
λ_test = [1.]
cpomdp = SoftConstraintPOMDPWrapper(CLightDark1D(cost_budget=50.);λ=λ_test)
solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zero_V,
    tree_in_info=true,
    search_progress_info=true)

updater = 

hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

R3
C3[1]
RC3
plot_lightdark_beliefs(hist3,"figs/belief_constrained.png")

inchrome(D3Tree(hist3[1][:tree]))

sp = SearchProgress(hist3[1])

plot(sp.v_best)
plot(sp.cv_best)
plot(sp.v_taken)
plot(sp.cv_taken)
plot(sp.lambda)

# notes:
# - better to use zero_V initialization (even if its super optimistic)
# - better to use zero-initialized lambdas, costant alpha schedule 
# - converges to lambda = 3ish 
# - cost 20. overconstraining, cost 30. 

