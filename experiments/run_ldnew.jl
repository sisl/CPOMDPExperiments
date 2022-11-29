using Revise
using CPOMDPExperiments
using Infiltrator
using ProgressMeter
using Distributed
using Random

nsims = 100
run = [true, true, true] #(pomcpow, pomcp, pft-dpw)

cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=[1.])

# global parameters
tree_queries = Int(1e5)
pft_tree_queries=Int(1e4)
k_observation = 5.
alpha_observation = 1/15
enable_action_pw = false
max_depth = 10
c = 90.0
nu = 0.0
asched = 0.5
update_filter_size = Int(1e4)
pf_filter_size = 10

if run[1] # POMCPOW

    kwargs = Dict(
        :tree_queries=>tree_queries, 
        :k_observation => k_observation, # 0.1,
        :alpha_observation => alpha_observation, #0.5,
        :enable_action_pw => false,
        :check_repeat_obs => false,
        :max_depth => max_depth,
        :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
        :estimate_value=>zeroV_trueC,
    )
    exp1 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, solver.rng), 
            planner)
        exp1[i] = run_cpomdp_simulation(cpomdp, solver, updater;track_history=false)
    end
    print_and_save(exp1,"results/lightdark_pomcpow_$(nsims)sims.jld2")
end

if run[2] # POMCP
    kwargs = Dict(
        :tree_queries=>pft_tree_queries, # POMCP trash anyway, keep small
        :k_observation => k_observation, # 0.1,
        :alpha_observation => alpha_observation, #0.5,
        :enable_action_pw => false,
        :check_repeat_obs => false,
        :max_depth => max_depth,
        :c=>c,
        :nu=>nu, 
        :alpha_schedule => CPOMDPExperiments.CPOMCP.ConstantAlphaSchedule(asched),
        :estimate_value=>zeroV_trueC,
    )
    exp2 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPDPWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCP.CPOMCPBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, solver.rng), 
            planner)
        exp2[i] = run_cpomdp_simulation(cpomdp, solver, updater;track_history=false)
    end
    print_and_save(exp2,"results/lightdark_pomcpdpw_$(nsims)sims.jld2")
end

if run[3] # PFT
    kwargs = Dict(
        :n_iterations=>pft_tree_queries, 
        :k_state => k_observation, # 0.1,
        :alpha_state => alpha_observation, #0.5,
        :enable_action_pw => false,
        :check_repeat_state => false,
        :depth => max_depth,
        :exploration_constant => c,
        :nu => nu, 
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(asched),
        :estimate_value=>CPOMDPExperiments.heuristicV,
    )
    exp3 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        rng = MersenneTwister(i)
        up = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, pf_filter_size, rng)
        solver = CPOMDPExperiments.CMCTS.BeliefCMCTSSolver(
            CPOMDPExperiments.CMCTS.CDPWSolver(;kwargs..., rng=rng),
            up)
        updater(planner) = CPOMDPExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, rng), 
            planner)
        exp3[i] = run_cpomdp_simulation(cpomdp, solver, updater;track_history=false)
    end
    print_and_save(exp3,"results/lightdark_pft_$(nsims)sims.jld2")
end