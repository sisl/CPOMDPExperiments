
# compare CLightDark1D with fixed bad_action_budget and 0 p.problem.incorrect_r with 
# LightDark1D with variable p.incorrect_r

struct CLightDark1D{P<:LightDark1D,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    bad_action_budget::Float64
    λ::Vector{Float64}
end

function CLightDark1D(;pomdp::P=LightDark1D(0.95, 100., -100., 1., -0., (x)->abs(x - 10)/sqrt(2) + 1e-2 ), # default 0 incorrect_r, goes into cost
    bad_action_budget::Float64=0.5,
    lambda::Vector{Float64}=Float64[0.]
    ) where {P<:LightDark1D}
    return CLightDark1D{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,bad_action_budget, lambda)
end

POMDPs.actions(::LightDark1D) = [-10, -1, 0, 1, 10]
function POMDPs.reward(p::LightDark1D, s::LightDark1DState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if abs(s.y) < 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end
POMDPs.reward(p::CLightDark1D, s::LightDark1DState, a::Int) = POMDPs.reward(p.pomdp, s, a) + p.λ⋅costs(p, s, a)
costs(::CLightDark1D, s::LightDark1DState, a::Int) = [abs(s.y)]
costs_limit(p::CLightDark1D) = [p.bad_action_budget]
n_costs(::CLightDark1D) = 1
max_reward(p::CLightDark1D) = p.pomdp.correct_r
min_reward(p::CLightDark1D) = -p.pomdp.movement_cost

