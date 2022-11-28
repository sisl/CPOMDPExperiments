using Revise
using CPOMDPExperiments
using D3Trees
using Plots 
using Random

### Find Good Settings
c = 20.0 
nu = 0.0
solver_kwargs = Dict(:tree_queries=>100, #5000,
        :init_λ => [1000.],
        :k_observation => 10.,
        :alpha_observation => 0.3,
        :max_depth => 30,
        :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(100.), # how much to update lambda by        
        :tree_in_info => true,
        :search_progress_info => true,
        :estimate_value => QMDP_V,
        :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu),
)

# default pomdp for reference
# SpillpointInjectionPOMDP(;exited_reward_amount=-1000, exited_reward_binary=-1000, obs_rewards=[-0.3, -0.7] , height_noise_std=0.01, sat_noise_std=0.01)


max_steps = 25 #100
λ_test = [0.] #  gets used when running unconstrained problem with scalarized reward

cpomdp = SoftConstraintPOMDPWrapper(SpillpointInjectionCPOMDP(constraint_budget=0.01);
    λ=λ_test)

solver = CPOMDPExperiments.CPOMCPOWSolver(;solver_kwargs..., 
    )

updater(planner) = CPOMDPExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMDPExperiments.SpillpointPOMDP.SIRParticleFilter(
        model=cpomdp.cpomdp,  
        N=200, 
        state2param=CPOMDPExperiments.SpillpointPOMDP.state2params, 
        param2state=CPOMDPExperiments.SpillpointPOMDP.params2state,
        N_samples_before_resample=100,
        clampfn=CPOMDPExperiments.SpillpointPOMDP.clamp_distribution,
        fraction_prior = 0.5,
        prior=CPOMDPExperiments.SpillpointPOMDP.param_distribution(
            CPOMDPExperiments.initialstate(cpomdp.cpomdp)),
        elite_frac=0.3,
        bandwidth_scale=.5,
        max_cpu_time=20 #60
    ), 
    planner)

hist, R, C, RC = run_cpomdp_simulation(cpomdp, solver, updater, max_steps)

R
C[1]
RC

inchrome(D3Tree(hist[1][:tree]))


sp = SearchProgress(hist[1])

# can plot values of best actions, taken actions, and lambdas across the search
plot(sp.v_best)
plot(sp.cv_best)
plot(sp.v_taken)
plot(sp.cv_taken)
plot(sp.lambda)


## constrained