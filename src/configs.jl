MODELS = ["rocksample", "lightdark1d", "roomba"]
SOLVERS = ["pomcp", "pft", "pomcpow"]

# model list. environment:(pomdp, cpomdp)
models = Dict(
    "rocksample" => (
    RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
        sensor_efficiency=20.0,
        discount_factor=0.95, 
        good_rock_reward = 20.0),
    RockSampleCPOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
        sensor_efficiency=20.0,
        discount_factor=0.95, 
        good_rock_reward = 20.0),
    ),
#    "vdptag" = (
#    VDPTagPOMDP(),
#    VDPTagPOMDP(),
#    ),
    "lightdark1d" => (
    LightDark1D(),
    LightDark1D(),
    ),
    "roomba" => (
    RoombaPOMDP(),
    RoombaPOMDP(),
    ),
)

# solvers: solver: (pomdp, cpomdp)
solvers = Dict(
    "pomcp" => ( #POMCP
        ::POMDP -> POMCPSolver(tree_queries=10000, c=10), 
        ::POMDP -> CPOMCPSolver(tree_queries=10000, c=10)
    ), 
    "pft" => ( #PFT-DPW
        p::POMDP -> BeliefMCTSSolver(DPWSolver(), SIRParticleFilter(p, 1000)),
        cp::POMDP -> BeliefCMCTSSolver(CDPWSolver(), SIRParticleFilter(cp, 1000))
    ), 
    "pomcpow" => ( # POMCPOW
        ::POMDP -> POMCPOWSolver(criterion=MaxUCB(20.0)), 
        ::POMDP -> CPOMCPOWSolver(criterion=MaxUCB(20.0))
    ), 
)
