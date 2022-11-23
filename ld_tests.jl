using Pkg
Pkg.activate(".")
using Revise
Pkg.resolve()
using CPOMDPExperiments
using Plots
using D3Trees
using Random

kwargs = (;tree_queries=10000, 
        k_observation = 0.1,
        alpha_action = 01/50)
c = 250.0
nu = 0.0
λ_test = [1.]


# basic pomcpow - lambda 0
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D())
solver = CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0,
    tree_in_info=true)
hist1, R1, C1, RC1 = run_pomdp_simulation(pomdp, solver)
R1
C1[1]
RC1
plot_lightdark_beliefs(hist1,"belief_l0.png")
inchrome(D3Tree(hist1[1][:tree]))
inchrome(D3Tree(hist1[5][:tree]))

# augmented pomcpow - lambda given
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D();λ=λ_test)
solver = CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0,
    tree_in_info=true)
hist2, R2, C2, RC2 = run_pomdp_simulation(pomdp, solver)
R2
C2[1]
RC2
plot_lightdark_beliefs(hist2,"belief_lgiven.png")

inchrome(D3Tree(hist2[1][:tree]))

# cpomcpow - fixed budget
cpomdp = SoftConstraintPOMDPWrapper(CLightDark1D();λ=λ_test)
solver = CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0,
    tree_in_info=true)
hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver)
R3
C3[1]
RC3
plot_lightdark_beliefs(hist3,"belief_constrained.png")

inchrome(D3Tree(hist3[1][:tree]))

# check for working lambda cutter
solver = CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0)
hist4, _, _, _, _ = run_cpomdp_simulation(CLightDark1D(), solver)
best_qs = hist4[1][:info][:best_qs]
best_cs = [c[0] for c in hist4[1][:info][:best_q_cs]]
chosen_qs = hist4[1][:info][:chosen_q]
chosen_cs = [c[0] for c in hist4[1][:info][:chosen_cs]]

######## multiple trials ######

nsims = 10
er_pomdp = ExperimentResults(nsims)
er_cpomdp = ExperimentResults(nsims)
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D(cost_budget=20.);λ=λ_test)
for i=1:nsims
    println("i=$i")
    er_pomdp[i] = run_pomdp_simulation(pomdp, 
        CPOMDPExperiments.POMCPOWSolver(;kwargs..., 
            criterion=CPOMDPExperiments.POMCPOW.MaxUCB(c), 
            estimate_value=0.0,
            rng=MersenneTwister(i)))
    
    er_cpomdp[i] = run_cpomdp_simulation(pomdp, 
        CPOMDPExperiments.CPOMCPOWSolver(;kwargs..., 
            criterion=CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
            estimate_value=(args...)->(0.0, [0.0]),
            rng=MersenneTwister(i)))
end
Rp_m, Cp_m, RCp_m = mean(er_pomdp)
Rp_std, Cp_std, RCp_std = std(er_pomdp)
Rc_m, Cc_m, RCc_m = mean(er_cpomdp)
Rc_std, Cc_std, RCc_std = std(er_cpomdp)

