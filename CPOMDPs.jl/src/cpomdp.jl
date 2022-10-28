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
    cost_limits(m::Union{CPOMDP,CMDP})

Return the upper limits vector of the cost functions

"""
function cost_limits end
POMDPLinter.@impl_dep cost_limits(::P) where {P<:Union{CPOMDP,CMDP}}

"""
    cost_value(p::Policy, s)
    cost_value(p::Policy, s, a)
Returns the cost value from policy `p` given the state (or belief), or state-action (or belief-action) pair.
"""
function costs_value end
    