struct CPOWNodeBelief{S,A,O,P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64,Vector{Float64}}}

    CPOWNodeBelief{S,A,O,P}(m,a,o,d) where {S,A,O,P<:CPOMDP} = new(m,a,o,d)
    function CPOWNodeBelief{S, A, O, P}(m::P, s::S, a::A, sp::S, o::O, r, c) where {S, A, O, P<:CPOMDP}
        cv = CategoricalVector{Tuple{S,Float64,Vector{Float64}}}((sp, convert(Float64, r), convert.(Float64,c)),
                                                 obs_weight(m, s, a, sp, o))
        new(m, a, o, cv)
    end
end

function CPOWNodeBelief(model::CPOMDP{S,A,O}, s::S, a::A, sp::S, o::O, r, c) where {S,A,O}
    CPOWNodeBelief{S,A,O,typeof(model)}(model, s, a, sp, o, r, c)
end

rand(rng::AbstractRNG, b::CPOWNodeBelief) = rand(rng, b.dist)
state_mean(b::CPOWNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::CPOWNodeBelief) = b.o
POMDPs.history(b::CPOWNodeBelief) = tuple((a=b.a, o=b.o))


struct CPOWNodeFilter end

belief_type(::Type{CPOWNodeFilter}, ::Type{P}) where {P<:CPOMDP} = CPOWNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::CPOWNodeFilter, p::CPOMDP, s, a, sp, o, r, c) = CPOWNodeBelief(p, s, a, sp, o, r, c)

function push_weighted!(b::CPOWNodeBelief, ::CPOWNodeFilter, s, sp, r, c)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r), convert.(Float64, c)), w)
end

struct CStateBelief{SRB<:CPOWNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::CStateBelief) = first(rand(rng, b.sr_belief))
mean(b::CStateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::CStateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::CStateBelief) = history(b.sr_belief)
