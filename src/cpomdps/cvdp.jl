@with_kw struct CVDPTagPOMDP <: ConstrainPOMDPWrapper
    pomdp::Union{VDPTagPOMDP,AODiscreteVDPTagPOMDP,ADiscreteVDPTagPOMDP} = VDPTagPOMDP(meas_cost=0.) # no measurement cost in CVDP, goes into hard constraint
    look_budget::Float64 = 5. # 5 discounted looks
end 

costs(p::CVDPTagPOMDP, s::TagState, a::TagAction) = Float64[a.look]
costs_limit(p::CVDPTagPOMDP) = [p.look_budget]
n_costs(::CVDPTagPOMDP) = 1
max_reward(p::CVDPTagPOMDP) = p.pomdp.tag_reward
min_reward(p::CVDPTagPOMDP) = -p.pomdp.step_cost

ModelTools.gbmdp_handle_terminal(pomdp::CVDPTagPOMDP, updater::Updater, b::ParticleCollection, s, a, rng) = ModelTools.gbmdp_handle_terminal(pomdp.pomdp, updater, b, s, a, rng)
