MODELS = ["rocksample", "lightdark1d", "vdptag", "roomba"]
SOLVERS = ["pomcp", "pft", "pomcpow"]

# model list. environment:(pomdp, cpomdp)
# cpomcp paper has following environments (nxn grid, m rocks, h half_efficiency_dist): (5,5,4), (5,7,20), (7,8,20), (11,11,20)
# defaults: |actions| = m+5, |obs| = 3, reward_range=20, discount=0.95, smart_move_prob=0.95, uncertainty_count=0, 
# rewards: +10 for leaving east, -100 for any other direction (made impossible), +10 for sampling valuable rock, -10 for worthless rock, -100 for no rock
# costs: 1 for any sampling, 1 for any negative reward. Total cost constraint is 1 (cannot ever sample worthless rock)



EXPERIMENTS = [("rocksample","pomcp"),
    ("lightdark1d","pft-ow"),
    ("lightdark1d","pomcpow-ow"),
    ("vdptag","pft-dpw"),
    #("spillpoint","pft-dpw"),
    ]

models = Dict(
    "rocksample" => (
    #RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
    #    sensor_efficiency=20.0,
    #    discount_factor=0.95, 
    #    good_rock_reward = 20.0),
    #RockSampleCPOMDP(rocks_positions=[(2,3), (4,4), (4,2)], 
    #    sensor_efficiency=20.0,
    #    discount_factor=0.95, 
    #    good_rock_reward = 20.0),
    RockSamplePOMDP(5,5),
    RockSampleCPOMDP(pomdp=RockSamplePOMDP(5,5)),
    ),
    "lightdark1d" => (
    LightDark1D(),
    CLightDark1D(),
    ),
    "vdptag" => (
    VDPTagPOMDP(),
    CVDPTagPOMDP(),
    ),
    "spillpoint" => (
    SpillpointInjectionPOMDP(),
    SpillpointInjectionCPOMDP(),
    ),
    #"roomba" => (
    #RoombaPOMDP(),
    #RoombaPOMDP(),
    #),
)

# solvers: solver: (pomdp, cpomdp)
solvers = Dict(
    "pomcp" => ( #POMCP
        ::POMDP -> POMCPSolver(tree_queries=10000, c=2), 
        ::CPOMDP -> CPOMCPSolver(tree_queries=10000, c=2)
    ), 
    "pft-dpw" => ( #PFT-DPW
        p::POMDP -> BeliefMCTSSolver(DPWSolver(), ParticleFilters.SIRParticleFilter(p, 1000)),
        cp::POMDP -> BeliefCMCTSSolver(CDPWSolver(), ParticleFilters.SIRParticleFilter(cp, 1000))
    ), 
    "pft-ow" => ( #PFT-OW
        p::POMDP -> BeliefMCTSSolver(DPWSolver(enable_action_pw=false), ParticleFilters.SIRParticleFilter(p, 1000)),
        cp::POMDP -> BeliefCMCTSSolver(CDPWSolver(enable_action_pw=false), ParticleFilters.SIRParticleFilter(cp, 1000))
    ), 
    "pomcpow" => ( # POMCPOW
        ::POMDP -> POMCPOWSolver(criterion=MaxUCB(20.0)), 
        ::CPOMDP -> CPOMCPOWSolver(criterion=MaxCUCB(20.0,0.1),)
    ), 
    "pomcpow-ow" => ( # POMCPOW
        ::POMDP -> POMCPOWSolver(criterion=MaxUCB(20.0),enable_action_pw=false), 
        ::CPOMDP -> CPOMCPOWSolver(criterion=MaxCUCB(20.0,0.1),enable_action_pw=false)
    )
)