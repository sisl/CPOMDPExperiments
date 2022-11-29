using Revise
using CPOMDPExperiments
using Infiltrator
using ProgressMeter
using Distributed
using Random

nsims = 5
run = [true, true, true] #(pomcpow, pomcp, pft-dpw)

cpomdp = SoftConstraintPOMDPWrapper(SpillpointInjectionCPOMDP(constraint_budget=0.);λ=[1000.])

# global parameters
max_steps=25
tree_queries = 100 # Int(1000) FIXME
pft_tree_queries = 10 # Int(100) FIXME
k_observation = 10.
alpha_observation = 0.3
max_depth = 10
c = 30.0
nu = 0.0
asched = 10000.
update_filter_size = Int(1e4)
pf_filter_size = 10
init_lam = [1000.]

default_updater = CPOMDPExperiments.SpillpointPOMDP.SIRParticleFilter(
        model=cpomdp.cpomdp,  
        N=200, 
        state2param=CPOMDPExperiments.SpillpointPOMDP.state2params, 
        param2state=CPOMDPExperiments.SpillpointPOMDP.params2state,
        N_samples_before_resample=100,
        clampfn=CPOMDPExperiments.SpillpointPOMDP.clamp_distribution,
        fraction_prior = .5,
        prior=CPOMDPExperiments.SpillpointPOMDP.param_distribution(
            CPOMDPExperiments.initialstate(cpomdp.cpomdp)),
        elite_frac=0.3,
        bandwidth_scale=.5,
        max_cpu_time=2 #20 #60 FIXME 
    )


if run[1] # POMCPOW

    kwargs = Dict(
        :tree_queries=>tree_queries, 
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
    )
    exp1 = LightExperimentResults(nsims)
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(default_updater, planner)
        exp1[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp1,"results/spillpoint_pomcpow_$(nsims)sims.jld2")
end

if run[2] # POMCP
    kwargs = Dict(
        :tree_queries=>pft_tree_queries, # POMCP trash anyway, keep small
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :c=>c,
        :nu=>nu, 
        :alpha_schedule => CPOMDPExperiments.CPOMCP.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
    )
    exp2 = LightExperimentResults(nsims)
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        solver = CPOMDPExperiments.CPOMCPDPWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMDPExperiments.CPOMCP.CPOMCPBudgetUpdateWrapper(default_updater, planner)
        exp2[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp2,"results/spillpoint_pomcpdpw_$(nsims)sims.jld2")
end

if run[3] # PFT
    kwargs = Dict(
        :n_iterations=>pft_tree_queries, 
        :k_state => k_observation, 
        :alpha_state => alpha_observation, 
        :check_repeat_state => false,
        :check_repeat_action => false,
        :depth => max_depth,
        :exploration_constant => c,
        :nu => nu, 
        :alpha_schedule => CPOMDPExperiments.CMCTS.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
    )
    exp3 = LightExperimentResults(nsims)
    @showprogress 1 for i = 1:nsims
        Random.seed!(i)
        rng = MersenneTwister(i)
        up = CPOMDPExperiments.ParticleFilters.BootstrapFilter(cpomdp, pf_filter_size, rng)
        solver = CPOMDPExperiments.CMCTS.BeliefCMCTSSolver(
            CPOMDPExperiments.CMCTS.CDPWSolver(;kwargs..., rng=rng),
            up)
        updater(planner) = CPOMDPExperiments.CMCTS.CMCTSBudgetUpdateWrapper(default_updater, planner)
        exp3[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp3,"results/spillpoint_pft_$(nsims)sims.jld2")
end