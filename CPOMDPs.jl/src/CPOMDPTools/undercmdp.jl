# Underlying CMDP is of type CMDP. Unfortunately can't also make it of type UnderlyingMDP, so recopying the methods

struct UnderlyingCMDP{P<:CPOMDP, S, A} <: CMDP{S,A}
    cpomdp::P
end

function UnderlyingCMDP(pomdp::CPOMDP{S, A, O}) where {S,A,O}
    P = typeof(pomdp)
    return UnderlyingCMDP{P,S,A}(pomdp)
end

UnderlyingCMDP(m::CMDP) = m

costs(mdp::UnderlyingCMDP{P, S, A}, s::S, a::A) where {P,S,A} = costs(mdp.cpomdp, s, a)
costs(mdp::UnderlyingCMDP{P, S, A}, s::S, a::A, sp::S) where {P,S,A} =  costs(mdp.cpomdp, s, a, sp)
n_costs(mdp::UnderlyingCMDP) = n_costs(mdp.cpomdp)
costs_budget(mdp::UnderlyingCMDP) = costs_budget(mdp.cpomdp)
POMDPs.transition(mdp::UnderlyingCMDP{P, S, A}, s::S, a::A) where {P,S,A}= transition(mdp.cpomdp, s, a)
POMDPs.initialstate(mdp::UnderlyingCMDP) = initialstate(mdp.cpomdp)
POMDPs.states(mdp::UnderlyingCMDP) = states(mdp.cpomdp)
POMDPs.actions(mdp::UnderlyingCMDP) = actions(mdp.cpomdp)
POMDPs.actions(mdp::UnderlyingCMDP{P, S, A}, s::S) where {P,S,A} = actions(mdp.cpomdp, s)
POMDPs.reward(mdp::UnderlyingCMDP{P, S, A}, s::S, a::A) where {P,S,A} = reward(mdp.cpomdp, s, a)
POMDPs.reward(mdp::UnderlyingCMDP{P, S, A}, s::S, a::A, sp::S) where {P,S,A} = reward(mdp.cpomdp, s, a, sp)
POMDPs.isterminal(mdp::UnderlyingCMDP{P, S, A}, s::S) where {P,S,A} = isterminal(mdp.cpomdp, s)
POMDPs.discount(mdp::UnderlyingCMDP) = discount(mdp.cpomdp)
POMDPs.stateindex(mdp::UnderlyingCMDP{P, S, A}, s::S) where {P,S,A} = stateindex(mdp.cpomdp, s)
POMDPs.stateindex(mdp::UnderlyingCMDP{P, Int, A}, s::Int) where {P,A} = stateindex(mdp.cpomdp, s) # fix ambiguity with src/convenience
POMDPs.stateindex(mdp::UnderlyingCMDP{P, Bool, A}, s::Bool) where {P,A} = stateindex(mdp.cpomdp, s)
POMDPs.actionindex(mdp::UnderlyingCMDP{P, S, A}, a::A) where {P,S,A} = actionindex(mdp.cpomdp, a)
POMDPs.actionindex(mdp::UnderlyingCMDP{P,S, Int}, a::Int) where {P,S} = actionindex(mdp.cpomdp, a)
POMDPs.actionindex(mdp::UnderlyingCMDP{P,S, Bool}, a::Bool) where {P,S} = actionindex(mdp.cpomdp, a)
POMDPs.gen(mdp::UnderlyingCMDP, s, a, rng) = gen(mdp.cpomdp, s, a, rng)
