const POCVDPTag = Union{VDPTagPOMDP,AODiscreteVDPTagPOMDP,ADiscreteVDPTagPOMDP} 

struct CVDPTagPOMDP{P<:POCVDPTag,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    look_budget::Float64 
end 
function CVDPTagPOMDP(;pomdp::P=VDPTagPOMDP(meas_cost=0.), # no measurement cost in CVDP, goes into hard constraint
    look_budget::Float64= 5., # 5 discounted looks
    ) where {P<:POCVDPTag}
    return CVDPTagPOMDP{P,statetype(pomdp),actiontype(pomdp),obstype(pomdp)}(pomdp,look_budget)
end


costs(p::CVDPTagPOMDP, s::TagState, a::TagAction) = Float64[a.look]
costs_limit(p::CVDPTagPOMDP) = [p.look_budget]
n_costs(::CVDPTagPOMDP) = 1
max_reward(p::CVDPTagPOMDP) = p.pomdp.mdp.tag_reward
min_reward(p::CVDPTagPOMDP) = -p.pomdp.mdp.step_cost

ModelTools.gbmdp_handle_terminal(pomdp::CVDPTagPOMDP, updater::Updater, b::ParticleCollection, s, a, rng) = ModelTools.gbmdp_handle_terminal(pomdp.pomdp, updater, b, s, a, rng)
