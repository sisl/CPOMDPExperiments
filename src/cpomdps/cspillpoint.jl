@with_kw struct SpillpointCPOMDP <: ConstrainPOMDPWrapper
    pomdp::SpillpointPOMDP = SpillpointPOMDP(exited_reward_amount=0.,exited_reward_binary=0.)
    steps_exited_budget::Float64 = 1. # discounted number of steps with allowed exit (aka exit probability)
    total_exited_budget::Float64 = .5 # discounted total volume of exited gas allowed
end 

# FIXME - assuming 0 exit penalties in pomdp, all going into constraints
# first element of cost is whether there was an exit, second is how much
function costs(::SpillpointCPOMDP, s, a, sp) 
    Δexited = sp.v_exited - s.v_exited
    return [Δexited > eps(Float32), Δexited] 
end
costs_limit(p::SpillpointCPOMDP) = [p.steps_exited_budget, p.amount_exited_budget]
n_costs(::SpillpointCPOMDP) = 2

max_volume_diff(p::SpillpointCPOMDP) = 100 # maximum expected volume diff in a single step FIXME
max_reward(p::SpillpointCPOMDP) = p.pomdp.trapped_reward * max_volume_diff(p)
min_reward(p::SpillpointCPOMDP) = min(p.pomdp.obs_rewards)


