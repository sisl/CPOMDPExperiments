using Revise
using CPOMDPExperiments
using D3Trees
using Plots 
using Distributed
using Random
using ProgressMeter

### Find Good Settings
# default from paper, commented from repository
kwargs = Dict(:tree_queries=>1e5,
        :k_action => 30., # 25.
        :alpha_action => 1/30, #1/20
        :k_observation => 5., #6.
        :alpha_observation => 01/100, 
        :max_depth => 10,
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(1e-2))

c = 110.0  #100. 
nu = 0.0
λ_test = [5.] # default meas_cost is 5., but set to 0. in CVDPTagPOMDP
filter_size = Int(1e4)

runs = [false, false, false, false, true]


if runs[1]
    cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=100.);
    λ=λ_test)

    solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        tree_in_info=true,
        search_progress_info=true)

    updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
        planner)

    hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

    R3
    C3[1]
    RC3

    inchrome(D3Tree(hist3[1][:tree]))


    sp3 = SearchProgress(hist3[1])
end
#plot(sp3.v_best)
#plot(sp3.cv_best)
#plot(sp3.v_taken)
#plot(sp3.cv_taken)
#plot(sp3.lambda)

## constrained

if runs[2]
    cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=5.);
    λ=λ_test)
    
    solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        tree_in_info=true,
        search_progress_info=true)

    updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
        planner)

    hist4, R4, C4, RC4 = run_cpomdp_simulation(cpomdp, solver, updater)

    R4
    C4[1]
    RC4

    inchrome(D3Tree(hist4[1][:tree]))


    sp4 = SearchProgress(hist4[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist4[1][:tree].tried[1]
    hist4[1][:tree].v[layer1]
    hist4[1][:tree].cv[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs
end 

if runs[3]
    cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=5.);
    λ=λ_test)

    solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        tree_in_info=true,
        search_progress_info=true,
        return_best_cost=true)

    updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
        planner)

    hist5, R5, C5, RC5 = run_cpomdp_simulation(cpomdp, solver, updater)

    R5
    C5[1]
    RC5

    inchrome(D3Tree(hist5[1][:tree]))


    sp5 = SearchProgress(hist5[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist5[1][:tree].tried[1]
    hist5[1][:tree].v[layer1]
    hist5[1][:tree].cv[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs

end

#pomcp-dpw
if runs[4]

    kwargs[:alpha_schedule] = CPOMDPExperiments.CPOMCP.ConstantAlphaSchedule(1e-2)

    cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=5.);
        λ= λ_test)

    solver = CPOMDPExperiments.CPOMCPDPWSolver(;kwargs..., 
        c=c,
        nu=nu, 
        tree_in_info=true,
        search_progress_info=true,
        return_best_cost=true)

    updater(planner) = CPOMDPExperiments.CPOMCP.CPOMCPBudgetUpdateWrapper(
        CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng), 
        planner)

    hist6, R6, C6, RC6 = run_cpomdp_simulation(cpomdp, solver, updater)

    R6
    C6[1]
    RC6

    inchrome(D3Tree(hist6[1][:tree]))


    sp6 = SearchProgress(hist6[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist6[1][:tree].children[1]
    hist6[1][:tree].v[layer1]
    hist6[1][:tree].cv[layer1]
end

#scalarized POMDP, what rewards do you get 
if runs[5]
    nsims = 10
    filter_size = Int(1e4)
    kwargs = Dict(:tree_queries=>1e5,
        :k_action => 30., # 25.
        :alpha_action => 1/30, #1/20
        :k_observation => 5., #6.
        :alpha_observation => 01/100, 
        :max_depth => 10,
        :criterion=>CPOMDPExperiments.POMCPOW.MaxUCB(c))
    cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=100.);
    λ=λ_test)
    exp = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i in 1:nsims
        solver = CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
            rng=Random.MersenneTwister(i), 
        )

        updater = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, filter_size, solver.rng)

        exp[i] = run_pomdp_simulation(cpomdp, solver, updater;track_history=false)
        
    end
    R_m, C_m, RC_m = mean(exp)
    R_std, C_std, RC_std = std(exp)

    # with a look cost of 5., a scalarized pomcpow solver gets
    # (R, C, RC) = (42pm28, 2.8pm1.5, 28pm32). Want to therefore set look_budget to 2.5 and \lambda init to 5. pair this with a constant schedule of 1.

end