module CVDPTag

# package code goes here
using POMDPs
using CPOMDPs
using StaticArrays
using Parameters
using Plots
using Distributions
using POMDPTools
using ParticleFilters
using Random
using LinearAlgebra


const Vec2 = SVector{2, Float64}
const Vec8 = SVector{8, Float64}

# importall POMDPs
import Base: rand, eltype, convert
import MCTS: next_action, n_children
import ParticleFilters: obs_weight
import POMDPs: actions, isterminal

export
    CVDPTagMDP,
    CVDPTagPOMDP,
    Vec2,

    DiscreteCVDPTagMDP,
    DiscreteCVDPTagPOMDP,
    AODiscreteCVDPTagPOMDP,
    ADiscreteCVDPTagPOMDP,
    #TranslatedPolicy,
    #translate_policy,
    #cproblem,

    convert_s,
    convert_a,
    convert_o,
    obs_weight

    #ToNextML,
    #ToNextMLSolver,
    #NextMLFirst,
    #DiscretizedPolicy,
    #ManageUncertainty,
    #CardinalBarriers,
    #mdp
    #isterminal

struct TagState
    agent::Vec2
    target::Vec2
end

struct TagAction
    look::Bool
    angle::Float64
end

@with_kw struct CVDPTagMDP{B} <: CMDP{TagState, Float64}
    mu::Float64          = 2.0
    agent_speed::Float64 = 1.0
    dt::Float64          = 0.1
    step_size::Float64   = 0.5
    tag_radius::Float64  = 0.1
    tag_reward::Float64  = 100.0
    step_cost::Float64   = 1.0
    pos_std::Float64     = 0.05
    barriers::B          = nothing
    tag_terminate::Bool  = true
    discount::Float64    = 0.95
end

@with_kw struct CVDPTagPOMDP{B} <: CPOMDP{TagState, TagAction, Vec8}
    mdp::CVDPTagMDP{B}           = CVDPTagMDP()
    meas_cost::Float64          = 5.0
    active_meas_std::Float64    = 0.1
    meas_std::Float64           = 5.0
end

const CVDPTagProblem = Union{CVDPTagMDP,CVDPTagPOMDP}
mdp(p::CVDPTagMDP) = p
mdp(p::CVDPTagPOMDP) = p.mdp

function next_ml_target(p::CVDPTagMDP, pos::Vec2)
    steps = round(Int, p.step_size/p.dt)
    for i in 1:steps
        pos = rk4step(p, pos)
    end
    return pos
end
next_ml_target(p::CVDPTagMDP, pos::AbstractVector) = next_ml_target(p, convert(Vec2, pos))

function POMDPs.transition(pp::CVDPTagProblem, s::TagState, a::Float64)
    ImplicitDistribution(pp, s, a) do pp, s, a, rng
        p = mdp(pp)
        targ = next_ml_target(p, s.target) + p.pos_std*SVector(randn(rng), randn(rng))
        agent = barrier_stop(p.barriers, s.agent, p.agent_speed*p.step_size*SVector(cos(a), sin(a)))
        return TagState(agent, targ)
    end
end

function POMDPs.reward(pp::CVDPTagProblem, s::TagState, a::Float64, sp::TagState)
    p = mdp(pp)
    if norm(sp.agent-sp.target) < p.tag_radius
        return p.tag_reward
    else
        return -p.step_cost
    end
end

POMDPs.discount(pp::CVDPTagProblem) = mdp(pp).discount
isterminal(pp::CVDPTagProblem, s::TagState) = mdp(pp).tag_terminate && norm(s.agent-s.target) < mdp(pp).tag_radius

struct AngleSpace end
rand(rng::AbstractRNG, ::AngleSpace) = 2*pi*rand(rng)
POMDPs.actions(::CVDPTagMDP) = AngleSpace()

POMDPs.transition(p::CVDPTagPOMDP, s::TagState, a::TagAction) = transition(p, s, a.angle)

struct POVDPTagActionSpace end
rand(rng::AbstractRNG, ::POVDPTagActionSpace) = TagAction(rand(rng, Bool), 2*pi*rand(rng))
POMDPs.actions(::CVDPTagPOMDP) = POVDPTagActionSpace()

function POMDPs.reward(p::CVDPTagPOMDP, s::TagState, a::TagAction, sp::TagState)
    return reward(mdp(p), s, a.angle, sp) - a.look*p.meas_cost
end

#=
Beam | covers (deg)
-------------------
1    | (0,45]
2    | (45,90]
etc.
=#

struct BeamDist
    abeam::Int
    an::Normal{Float64}
    n::Normal{Float64}
end

function rand(rng::AbstractRNG, d::BeamDist)
    o = MVector{8, Float64}(undef)
    for i in 1:length(o)
        if i == d.abeam
            o[i] = rand(rng, d.an)
        else
            o[i] = rand(rng, d.n)
        end
    end
    return SVector(o)
end

function POMDPs.pdf(d::BeamDist, o::Vec8)
    p = 1.0
    for i in 1:length(o)
        if i == d.abeam
            p *= POMDPs.pdf(d.an, o[i])
        else
            p *= POMDPs.pdf(d.n, o[i])
        end
    end
    return p
end

function active_beam(rel_pos::Vec2)
    angle = atan(rel_pos[2], rel_pos[1])
    while angle <= 0.0
        angle += 2*pi
    end
    bm = ceil(Int, 8*angle/(2*pi))
    return clamp(bm, 1, 8)
end

function POMDPs.observation(p::CVDPTagPOMDP, a::TagAction, sp::TagState)
    rel_pos = sp.target - sp.agent
    dist = norm(rel_pos)
    abeam = active_beam(rel_pos)
    if a.look
        an = Normal(dist, p.active_meas_std)
    else
        an = Normal(dist, p.meas_std)
    end
    n = Normal(1.0, p.meas_std)
    BeamDist(abeam, an, n)
end

POMDPs.observation(p::CVDPTagPOMDP, a::Float64, sp::TagState) = observation(p, TagAction(false, a), sp)

include("rk4.jl")
include("barriers.jl")
include("initial.jl")
include("discretized.jl")
include("visualization.jl")
include("heuristics.jl")

function ModelTools.gbmdp_handle_terminal(pomdp::CVDPTagPOMDP, updater::Updater, b::ParticleCollection, s, a, rng)
    return ParticleCollection([s])
end

end # module
