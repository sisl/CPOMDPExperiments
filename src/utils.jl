### Experiment Results

mutable struct ExperimentResults
    hists::Vector{Vector{NamedTuple}}
    Rs::Vector{Float64}
    Cs::Vector{Vector{Float64}}
    RCs::Vector{Float64}
end

ExperimentResults(num::Int) = ExperimentResults(
    Array{Vector{NamedTuple}}(undef,num),
    Array{Float64}(undef,num),
    Array{Vector{Float64}}(undef,num),
    Array{Float64}(undef,num),
    )

Base.getindex(X::ExperimentResults, i::Int)	= (X.hists[i],X.Rs[i],X.Cs[i],X.RCs[i])

function Base.setindex!(X::ExperimentResults, v::Tuple{Vector{NamedTuple},Float64,Vector{Float64},Float64}, i::Int)
    X.hists[i] = v[1]
    X.Rs[i] = v[2]
    X.Cs[i] = v[3]
    X.RCs[i] = v[4]
end

mean(er::ExperimentResults) = Statistics.mean(er.Rs), Statistics.mean(er.Cs), Statistics.mean(er.RCs)

function std(er::ExperimentResults;corrected::Bool=false)
    stdR = Statistics.std(er.Rs;corrected=corrected)
    stdC = Statistics.std(er.Cs;corrected=corrected)
    stdRC = Statistics.std(er.RCs;corrected=corrected)
    return stdR, stdC, stdRC
end

### Reward wrapper

struct SoftConstraintPOMDPWrapper{P,S,A,O} <: POMDP{S,A,O} where {P<:POMDP}
    cpomdp::ConstrainPOMDPWrapper{P,S,A,O}
    λ::Vector{Float64}
end
function SoftConstraintPOMDPWrapper(c::ConstrainPOMDPWrapper; 
    λ::Union{Nothing,Vector{Float64}}=nothing)

    if λ == nothing
        λ = zeros(Float64, n_costs(c))
    else
        @assert length(λ) == n_costs(c) "Length of λ should match length of CPOMDP"
    end
    P = typeof(c.pomdp)
    S = statetype(c.pomdp)
    A = actiontype(c.pomdp)
    O = obstype(c.pomdp)
    return SoftConstraintPOMDPWrapper{P,S,A,O}(c,λ)
end

POMDPs.reward(p::SoftConstraintPOMDPWrapper, args...) = POMDPs.reward(p.cpomdp, args...) - p.λ⋅CPOMDP.costs(p.cpomdp, args...)
POMDPs.discount(m::SoftConstraintPOMDPWrapper) = POMDPs.discount(m.cpomdp)
POMDPs.transition(m::SoftConstraintPOMDPWrapper, state, action) = POMDPs.transition(m.cpomdp, state, action)
POMDPs.observation(m::SoftConstraintPOMDPWrapper, args...) = POMDPs.observation(m.cpomdp, args...)
POMDPs.isterminal(m::SoftConstraintPOMDPWrapper, s) = POMDPs.isterminal(m.cpomdp, s)
POMDPs.initialstate(m::SoftConstraintPOMDPWrapper) = POMDPs.initialstate(m.cpomdp)
POMDPs.initialobs(m::SoftConstraintPOMDPWrapper, s) = POMDPs.initialobs(m.cpomdp, s)
POMDPs.stateindex(problem::SoftConstraintPOMDPWrapper, s) = POMDPs.stateindex(problem.cpomdp, s)
POMDPs.actionindex(problem::SoftConstraintPOMDPWrapper, a) = POMDPs.actionindex(problem.cpomdp, a)
POMDPs.obsindex(problem::SoftConstraintPOMDPWrapper, o) = POMDPs.obsindex(problem.cpomdp, o)
POMDPs.convert_s(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_s(a, b, problem.cpomdp)
POMDPs.convert_a(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_a(a, b, problem.cpomdp)
POMDPs.convert_o(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_o(a, b, problem.cpomdp) 
POMDPs.states(problem::SoftConstraintPOMDPWrapper) = POMDPs.states(problem.cpomdp)
POMDPs.actions(m::SoftConstraintPOMDPWrapper) = POMDPs.actions(m.cpomdp)
POMDPs.actions(m::SoftConstraintPOMDPWrapper, s) = POMDPs.actions(m.cpomdp, s)
POMDPs.observations(problem::SoftConstraintPOMDPWrapper) = POMDPs.observations(problem.cpomdp)
POMDPs.observations(problem::SoftConstraintPOMDPWrapper, s) = POMDPs.observations(problem.cpomdp, s) 
POMDPs.statetype(p::SoftConstraintPOMDPWrapper) = POMDPs.statetype(p.cpomdp)
POMDPs.actiontype(p::SoftConstraintPOMDPWrapper) = POMDPs.actiontype(p.cpomdp)
POMDPs.obstype(p::SoftConstraintPOMDPWrapper) = POMDPs.obstype(p.cpomdp)
POMDPs.gen(m::SoftConstraintPOMDPWrapper, s, a, rng::AbstractRNG) = POMDPs.gen(m.cpomdp, s, a, rng)

### Other utils 

function plot_lightdark_beliefs(hist::Vector{NamedTuple},saveloc::Union{String,Nothing}=nothing )
    states = [h[:s].y for h in hist]
    beliefs = [h[:b] for h in hist]

    xpts = []
    ypts = []
    max_particles = 100
    for i=1:length(beliefs) 
        count = 0
        for s in beliefs[i].particles
            push!(xpts, i)
            push!(ypts, s.y)
            count += 1
            if count > max_particles
                break
            end
        end
    end

    scatter(xpts, ypts)
    scatter!(1:length(states), states)
    if !(saveloc == nothing)
        savefig(saveloc)
    end
end

