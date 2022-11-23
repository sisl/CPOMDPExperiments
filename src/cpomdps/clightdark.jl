# Fixing imported model

POMDPs.initialstate(pomdp::LightDark1D) = POMDPModels.LDNormalStateDist(0, 3)
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

# compare CLightDark1D with fixed cost_budget and 0 p.problem.incorrect_r with 
# LightDark1D with variable p.incorrect_r

struct CLightDark1D{P<:LightDark1D,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    cost_budget::Float64
end

#function CLightDark1D(;pomdp::P=LightDark1D(0.95, 100., -100., 1., -0., (x)->abs(x - 10)/sqrt(2) + 1e-2 ), # default 0 incorrect_r, goes into cost
function CLightDark1D(;pomdp::P=LightDark1D(0.95, 100., -100., 1., 1., (x)->abs(x - 10) + 1e-2 ), # default 0 incorrect_r, goes into cost
    cost_budget::Float64=30.,
    ) where {P<:LightDark1D}
    return CLightDark1D{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget)
end

# rough
function QMDP_V(p::LightDark1D, s::LightDark1DState, args...) 
    y = abs(s.y)
    steps = floor(Int, y/10)
    steps += floor(Int, y-10*steps)
    γ = discount(p)
    return -sum([(γ^i)*p.movement_cost  for i in 0:steps-1]) + (γ^steps)*p.correct_r 
end

function QMDP_V(p::CLightDark1D, s::LightDark1DState, args...)
    V = QMDP_V(p.pomdp,s,args...)
    y = abs(s.y)
    C = 0
    γ = 1
    while y > 1
        C+= γ*y
        if y > 10
            y -= 10
        else
            y -= 1
        end
        γ *= discount(p)
    end
    return (V, [C])
end

costs(::CLightDark1D, s::LightDark1DState, a::Int) = [abs(s.y)]
costs_limit(p::CLightDark1D) = [p.cost_budget]
n_costs(::CLightDark1D) = 1
max_reward(p::CLightDark1D) = p.pomdp.correct_r
min_reward(p::CLightDark1D) = -p.pomdp.movement_cost


### Light Dark New
# POMDPs.actions(::LightDark1D) = [-10, -5, -1, 0, 1, 5, 10]

struct CLightDarkNew{P<:LightDark1D,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    cost_budget::Float64
end

function CLightDarkNew(;pomdp::P=LightDark1D(0.95, 100., -100., 1., -0., (x)->abs(x - 10) + 1e-2 ), # default 0 incorrect_r, goes into cost
    cost_budget::Float64=0.5,
    lambda::Vector{Float64}=Float64[0.]
    ) where {P<:LightDark1D}
    return CLightDarkNew{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget)
end

costs(::CLightDarkNew, s::LightDark1DState, a::Int) = Float64[s.y > 12]
costs_limit(p::CLightDarkNew) = [p.cost_budget]
n_costs(::CLightDarkNew) = 1
max_reward(p::CLightDarkNew) = p.pomdp.correct_r
min_reward(p::CLightDarkNew) = -p.pomdp.movement_cost
