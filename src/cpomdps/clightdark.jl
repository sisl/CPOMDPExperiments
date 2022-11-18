
# compare CLightDark1D with fixed bad_action_budget and 0 p.problem.incorrect_r with 
# LightDark1D with variable p.incorrect_r

struct CLightDark1D{P<:LightDark1D,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    bad_action_budget::Float64
end

function CLightDark1D(;pomdp::P=LightDark1D(0.9, 10.0, 0., 1.0, 0.0, POMDPModels.default_sigma), # default 0 incorrect_r, goes into cost
    bad_action_budget::Float64=0.5,
    ) where {P<:LightDark1D}
    return CLightDark1D{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,bad_action_budget)
end

costs(::CLightDark1D, s::LightDark1DState, a::Int) = Float64[a==0 && abs(s.y) >= 1]
costs_limit(p::CLightDark1D) = [p.bad_action_budget]
n_costs(::CLightDark1D) = 1
max_reward(p::CLightDark1D) = p.pomdp.correct_r
min_reward(p::CLightDark1D) = -p.pomdp.movement_cost