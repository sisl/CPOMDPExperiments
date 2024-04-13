### Experiment Results

mutable struct LightExperimentResults
    Rs::Vector{Float64}
    Cs::Vector{Vector{Float64}}
    RCs::Vector{Float64}
end

LightExperimentResults(num::Int) = LightExperimentResults(
    Array{Float64}(undef,num),
    Array{Vector{Float64}}(undef,num),
    Array{Float64}(undef,num),
    )

Base.getindex(X::LightExperimentResults, i::Int)	= (X.Rs[i],X.Cs[i],X.RCs[i])

function Base.setindex!(X::LightExperimentResults, v::Tuple{Vector{NamedTuple},Float64,Vector{Float64},Float64}, i::Int)
    X.Rs[i] = v[2]
    X.Cs[i] = v[3]
    X.RCs[i] = v[4]
end

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

mean(er::Union{LightExperimentResults,ExperimentResults}) = Statistics.mean(er.Rs), Statistics.mean(er.Cs), Statistics.mean(er.RCs)

function std(er::Union{LightExperimentResults,ExperimentResults};corrected::Bool=false)
    stdR = Statistics.std(er.Rs;corrected=corrected)
    stdC = Statistics.std(er.Cs;corrected=corrected)
    stdRC = Statistics.std(er.RCs;corrected=corrected)
    return stdR, stdC, stdRC
end

function print_and_save(er::Union{LightExperimentResults,ExperimentResults}, fileloc::String)
    l = length(er.Rs)
    mR, mC, mRC = mean(er)
    stdR, stdC, stdRC = std(er)
    println("R: $(mR) pm $(stdR ./ sqrt(l))")
    println("C: $(mC) pm $(stdC ./ sqrt(l))")
    println("RC: $(mRC) pm $(stdRC ./ sqrt(l))")
    d = Dict(
        "R"=>er.Rs, "C"=> er.Cs, "RCs"=>er.RCs
    )
    FileIO.save(fileloc,d)
end

function load_and_print(fileloc::String)
    d = load(fileloc)
    er = LightExperimentResults(d["R"], d["C"], d["RCs"])
    l = length(er.Rs)
    mR, mC, mRC = mean(er)
    stdR, stdC, stdRC = std(er)
    println("R: $(mR) pm $(stdR ./ sqrt(l))")
    println("C: $(mC) pm $(stdC ./ sqrt(l))")
    println("RC: $(mRC) pm $(stdRC ./ sqrt(l))")
    return er
end

### Reward wrapper

"""
A POMDP wrapper that acts according to the underlying POMDP but augments rewards from the underlying CPOMDP
"""
struct SoftConstraintPOMDPWrapper{P,C,S,A,O} <: POMDP{S,A,O} where {P<:POMDP{S,A,O}, C<:CPOMDP{S,A,O}}
    pomdp::P
    cpomdp::C
    λ::Vector{Float64}
end

function SoftConstraintPOMDPWrapper(c::ConstrainPOMDPWrapper; 
    λ::Union{Nothing,Vector{Float64}}=nothing)

    if λ === nothing
        λ = zeros(Float64, n_costs(c))
    else
        @assert length(λ) == n_costs(c) "Length of λ should match length of CPOMDP"
    end
    P = typeof(c.pomdp)
    C = typeof(c)
    S = statetype(c.pomdp)
    A = actiontype(c.pomdp)
    O = obstype(c.pomdp)
    return SoftConstraintPOMDPWrapper{P,C,S,A,O}(c.pomdp,c,λ)
end

POMDPs.reward(p::SoftConstraintPOMDPWrapper, args...) = POMDPs.reward(p.pomdp, args...) - p.λ⋅CPOMDPs.costs(p.cpomdp, args...)
POMDPs.discount(m::SoftConstraintPOMDPWrapper) = POMDPs.discount(m.pomdp)
POMDPs.transition(m::SoftConstraintPOMDPWrapper, state, action) = POMDPs.transition(m.pomdp, state, action)
POMDPs.observation(m::SoftConstraintPOMDPWrapper, args...) = POMDPs.observation(m.pomdp, args...)
POMDPs.isterminal(m::SoftConstraintPOMDPWrapper, s) = POMDPs.isterminal(m.pomdp, s)
POMDPs.initialstate(m::SoftConstraintPOMDPWrapper) = POMDPs.initialstate(m.pomdp)
POMDPs.initialobs(m::SoftConstraintPOMDPWrapper, s) = POMDPs.initialobs(m.pomdp, s)
POMDPs.stateindex(problem::SoftConstraintPOMDPWrapper, s) = POMDPs.stateindex(problem.pomdp, s)
POMDPs.actionindex(problem::SoftConstraintPOMDPWrapper, a) = POMDPs.actionindex(problem.pomdp, a)
POMDPs.obsindex(problem::SoftConstraintPOMDPWrapper, o) = POMDPs.obsindex(problem.pomdp, o)
POMDPs.convert_s(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_s(a, b, problem.pomdp)
POMDPs.convert_a(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_a(a, b, problem.pomdp)
POMDPs.convert_o(a, b, problem::SoftConstraintPOMDPWrapper) = POMDPs.convert_o(a, b, problem.pomdp) 
POMDPs.states(problem::SoftConstraintPOMDPWrapper) = POMDPs.states(problem.pomdp)
POMDPs.actions(m::SoftConstraintPOMDPWrapper) = POMDPs.actions(m.pomdp)
POMDPs.actions(m::SoftConstraintPOMDPWrapper, s) = POMDPs.actions(m.pomdp, s)
POMDPs.observations(problem::SoftConstraintPOMDPWrapper) = POMDPs.observations(problem.pomdp)
POMDPs.observations(problem::SoftConstraintPOMDPWrapper, s) = POMDPs.observations(problem.pomdp, s) 
POMDPs.statetype(p::SoftConstraintPOMDPWrapper) = POMDPs.statetype(p.pomdp)
POMDPs.actiontype(p::SoftConstraintPOMDPWrapper) = POMDPs.actiontype(p.pomdp)
POMDPs.obstype(p::SoftConstraintPOMDPWrapper) = POMDPs.obstype(p.pomdp)
POMDPs.gen(m::SoftConstraintPOMDPWrapper, s, a, rng::AbstractRNG) = POMDPs.gen(m.pomdp, s, a, rng)

### Lambda Experiment Structures

struct Dist
    mean::Float64
    std::Float64
end

mutable struct LambdaExperiments
    λs::Vector{Float64}
    nsims::Int # per lambda
    Rs::Vector{Dist}
    Cs::Vector{Dist}
    RCs::Vector{Dist}
    
    R_CPOMDP::Union{Dist,Nothing}
    C_CPOMDP::Union{Dist,Nothing}
    R_CPOMDP_minC::Union{Dist,Nothing}
    C_CPOMDP_minC::Union{Dist,Nothing}
end

LambdaExperiments(lambdas::Vector{Float64};nsims::Int=10) = LambdaExperiments(
    lambdas,
    nsims,
    Array{Dist}(undef, length(lambdas)),
    Array{Dist}(undef, length(lambdas)),
    Array{Dist}(undef, length(lambdas)),
    nothing, nothing, nothing, nothing
)
    
function save_le(le::LambdaExperiments,saveloc::String)
    d = Dict(
        "lambdas"=>le.λs,
        "POMDP_C_mean"=>[i.mean for i in le.Cs],
        "POMDP_C_std"=>[i.std for i in le.Cs],
        "POMDP_R_mean"=>[i.mean for i in le.Rs],
        "POMDP_R_std"=>[i.std for i in le.Rs] ,
        "CPOMDP_C_mean"=> !(le.C_CPOMDP===nothing) ? le.C_CPOMDP.mean : nothing,
        "CPOMDP_C_std"=> !(le.C_CPOMDP===nothing) ? le.C_CPOMDP.std : nothing,
        "CPOMDP_R_mean"=> !(le.C_CPOMDP===nothing) ? le.R_CPOMDP.mean : nothing,
        "CPOMDP_R_std"=> !(le.C_CPOMDP===nothing) ? le.R_CPOMDP.std : nothing,
        "CPOMDP_minC_C_mean"=> !(le.C_CPOMDP_minC===nothing) ? le.C_CPOMDP_minC.mean : nothing,
        "CPOMDP_minC_C_std"=> !(le.C_CPOMDP_minC===nothing) ? le.C_CPOMDP_minC.std : nothing,
        "CPOMDP_minC_R_mean"=> !(le.C_CPOMDP_minC===nothing) ? le.R_CPOMDP_minC.mean : nothing,
        "CPOMDP_minC_R_std"=> !(le.C_CPOMDP_minC===nothing) ? le.R_CPOMDP_minC.std : nothing,
        )

    FileIO.save(saveloc,d)
end

function load_le(saveloc::String)
    d = load(saveloc)
    
    λs = d["lambdas"]
    nsims = 0
    Rs= Dist[Dist(m,s) for (m,s) in zip(d["POMDP_R_mean"],d["POMDP_R_std"])]
    Cs= Dist[Dist(m,s) for (m,s) in zip(d["POMDP_C_mean"],d["POMDP_C_std"])]
    RCs = Dist[]
    R_CPOMDP = (d["CPOMDP_C_mean"]===nothing) ? nothing : Dist(d["CPOMDP_R_mean"],d["CPOMDP_R_std"])
    C_CPOMDP = (d["CPOMDP_C_mean"]===nothing) ? nothing : Dist(d["CPOMDP_C_mean"],d["CPOMDP_C_std"])
    R_CPOMDP_minC = (d["CPOMDP_minC_C_mean"]===nothing) ? nothing : Dist(d["CPOMDP_minC_R_mean"],d["CPOMDP_minC_R_std"])
    C_CPOMDP_minC = (d["CPOMDP_minC_C_mean"]===nothing) ? nothing : Dist(d["CPOMDP_minC_C_mean"],d["CPOMDP_minC_C_std"])

    return LambdaExperiments(λs, nsims, Rs, Cs, RCs, 
        R_CPOMDP,C_CPOMDP, R_CPOMDP_minC, C_CPOMDP_minC)
end

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

zero_V(p::POMDP, args...) = 0.
zero_V(p::CPOMDP, args...) = (0.0, zeros(Float64, n_costs(p)))
zero_V(p::MDP, args...) = 0.
zero_V(p::CMDP, args...) = (0.0, zeros(Float64, n_costs(p)))
QMDP_V(args...) = zero_V(args...) #default
function QMDP_V(p::SoftConstraintPOMDPWrapper, args...) 
    V, C = QMDP_V(p.cpomdp, args...)
    return V - λ⋅C
end


### 
struct SearchProgress
    v_best::Vector{Float64}
    cv_best::Vector{Float64}
    v_taken::Vector{Float64}
    cv_taken::Vector{Float64}
    lambda::Vector{Float64}
end
    
SearchProgress(search_info::NamedTuple) = SearchProgress(
    search_info[:v_best],
    [c[1] for c in search_info[:cv_best]],
    search_info[:v_taken],
    [c[1] for c in search_info[:cv_taken]],
    [c[1] for c in search_info[:lambda]],
)