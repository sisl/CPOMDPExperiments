using Revise
using CPOMDPExperiments
using Infiltrator
using ProgressMeter
using Distributed
using Random

nsims = 25
run = [true, true, true] #(pomcpow, pomcp, pft-dpw)

cpomdp = SoftConstraintPOMDPWrapper(CVDPTagPOMDP(look_budget=2.5);λ=[5.])

# global parameters
tree_queries = Int(1e5)
pft_tree_queries = Int(1e4)
k_observation = 5.
alpha_observation = 1/100.
k_action = 30.
alpha_action = 1/30.
max_depth = 10
c = 110.0
nu = 0.0
asched = 1.
update_filter_size = Int(1e4)
pf_filter_size = 10
init_lam = [5.]
max_steps = 25



if run[3] # PFT
    kwargs = Dict(
        :n_iterations=>pft_tree_queries, 
        :k_state => k_observation, 
        :alpha_state => alpha_observation, 
        :k_action => k_action, 
        :alpha_action => alpha_action, 
        :check_repeat_state => false,
        :check_repeat_action => false,
        :depth => max_depth,
        :exploration_constant => c,
        :nu => nu, 
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value => zero_V,
    )
    exp3 = LightExperimentResults(nsims)
    println("running vdp pft-dpw")
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        rng = MersenneTwister(i)
        up = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, pf_filter_size, rng)
        solver = CPOMDPExperiments.CMCTS.BeliefCMCTSSolver(
            CPOMDPExperiments.CMCTS.CDPWSolver(;kwargs..., rng=rng),
            up)
        updater(planner) = CPOMDPExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, rng), 
            planner)
        exp3[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp3,"results/vdp_pft_$(nsims)sims_1.jld2")
end

if run[1] # POMCPOW

    kwargs = Dict(
        :tree_queries=>tree_queries, 
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :k_action => k_action,
        :alpha_action => alpha_action,
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value => zero_V,
    )
    exp1 = LightExperimentResults(nsims)
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, solver.rng), 
            planner)
        exp1[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp1,"results/vdp_pomcpow_$(nsims)sims_1.jld2")
end

if run[2] # POMCP
    kwargs = Dict(
        :tree_queries=>pft_tree_queries, # POMCP trash anyway, keep small 
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :k_action => k_action,
        :alpha_action => alpha_action,
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :c=>c,
        :nu=>nu, 
        :alpha_schedule => CPOMDPExperiments.CPOMCP.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value => zero_V,
    )
    exp2 = LightExperimentResults(nsims)
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPDPWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCP.CPOMCPBudgetUpdateWrapper(
            CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, update_filter_size, solver.rng), 
            planner)
        exp2[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp2,"results/vdp_pomcpdpw_$(nsims)sims_1.jld2")
end