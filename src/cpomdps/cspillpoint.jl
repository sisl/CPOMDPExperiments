struct SpillpointInjectionCPOMDP{P<:SpillpointInjectionPOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P
    steps_exited_budget::Float64 # discounted number of steps with allowed exit (aka exit probability)
    total_exited_budget::Float64 # discounted total volume of exited gas allowed
end 

function SpillpointInjectionCPOMDP(;pomdp::P=SpillpointInjectionPOMDP(exited_reward_amount=0.,exited_reward_binary=0.), # default 0 bad rewards, goes into cost
    steps_exited_budget::Float64 = 1., # discounted number of steps with allowed exit (aka exit probability)
    total_exited_budget::Float64 = .5, # discounted total volume of exited gas allowed
    ) where {P<:SpillpointInjectionPOMDP}
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,steps_exited_budget,total_exited_budget)
end

# Option 1, two constraints
function costs(::SpillpointInjectionCPOMDP, s, a, sp) 
    Δexited = sp.v_exited - s.v_exited
    return [Δexited > eps(Float32), Δexited] 
end
costs_limit(p::SpillpointInjectionCPOMDP) = [p.steps_exited_budget, p.amount_exited_budget]
n_costs(::SpillpointInjectionCPOMDP) = 2

max_volume_diff(p::SpillpointInjectionCPOMDP) = 100 # maximum expected volume diff in a single step FIXME
max_reward(p::SpillpointInjectionCPOMDP) = p.pomdp.trapped_reward * max_volume_diff(p)
min_reward(p::SpillpointInjectionCPOMDP) = min(p.pomdp.obs_rewards)

### Other option, single cost for exiting binary, 
"""
function SpillpointInjectionCPOMDP(;pomdp::P=SpillpointInjectionPOMDP(exited_reward_amount=0.), # default 0 bad rewards, goes into cost
    steps_exited_budget::Float64 = 1., # discounted number of steps with allowed exit (aka exit probability)
    total_exited_budget::Float64 = 0., # discounted total volume of exited gas allowed
    ) where {P<:SpillpointInjectionPOMDP}
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,steps_exited_budget,total_exited_budget)
end
function costs(::SpillpointInjectionCPOMDP, s, a) 
    return [Δexited > eps(Float32)] 
end
costs_limit(p::SpillpointInjectionCPOMDP) = [p.steps_exited_budget]
n_costs(::SpillpointInjectionCPOMDP) = 1
"""

