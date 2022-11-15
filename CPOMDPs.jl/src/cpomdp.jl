abstract type CMDP{S,A} <: MDP{S,A} end
abstract type CPOMDP{S,A,O} <: POMDP{S,A,O} end

"""
    costs(m::CPOMDP, s, a)
    costs(m::CMDP, s, a)

Return the immediate costs vector for the s-a pair.
    
    costs(m::CPOMDP, s, a, sp)
    costs(m::CMDP, s, a, sp)

Return the immediate costs vector for the s-a-s' triple

    costs(m::CPOMDP, s, a, sp, o)

Return the immediate costs vector for the s-a-s'-o quad

"""
function costs end

costs(m::Union{CPOMDP,CMDP}, s, a, sp) = costs(m, s, a)
POMDPLinter.@impl_dep costs(::P,::S,::A,::S) where {P<:Union{CPOMDP,CMDP},S,A} costs(::P,::S,::A)

costs(m::Union{POMDP,MDP}, s, a, sp, o) = costs(m, s, a, sp)
POMDPLinter.@impl_dep costs(::P,::S,::A,::S,::O) where {P<:Union{CPOMDP,CMDP},S,A,O} costs(::P,::S,::A,::S)

"""
    costs_limit(m::Union{CPOMDP,CMDP})

Return the upper limits vector of the cost functions

"""
function costs_limit end

"""
    cost_value(p::Policy, s)
    cost_value(p::Policy, s, a)
Returns the cost value from policy `p` given the state (or belief), or state-action (or belief-action) pair.
"""
function costs_value end
    
"""
    ncosts(m::Union{CPOMDP,CMDP})

Return the number of constraints
"""
function n_costs end

"""
    min_reward(m::Union{CPOMDP,CMDP})

Return the minimum single-step reward

"""
function min_reward end

"""
    max_reward(m::Union{CPOMDP,CMDP})

Return the maximum single-step reward

"""
function max_reward end