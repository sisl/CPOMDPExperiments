import POMDPs.initialstate
const IVec8 = SVector{8, Int}

@with_kw struct AODiscreteCVDPTagPOMDP{B} <: POMDP{TagState, Int, IVec8}
    cpomdp::CVDPTagPOMDP{B}  = CVDPTagPOMDP()
    n_angles::Int           = 10
    binsize::Float64        = 0.5
end

@with_kw struct ADiscreteCVDPTagPOMDP{B} <: POMDP{TagState, Int, Vec8}
    cpomdp::CVDPTagPOMDP{B}  = CVDPTagPOMDP()
    n_angles::Int           = 10
end


const DiscreteCVDPTagProblem = Union{AODiscreteCVDPTagPOMDP, ADiscreteCVDPTagPOMDP}

cproblem(p::AODiscreteCVDPTagPOMDP) = p.cpomdp
cproblem(p::ADiscreteCVDPTagPOMDP) = p.cpomdp

convert_s(::Type{T}, x::T, p) where T = x
convert_a(::Type{T}, x::T, p) where T = x
convert_o(::Type{T}, x::T, p) where T = x

# state
function convert_s(::Type{Int}, s::TagState, p::DiscreteCVDPTagProblem)
    n = p.n_bins
    factor = n/(2*p.grid_lim)
    ai = clamp(ceil(Int, (s.agent[1]+p.grid_lim)*factor), 1, n)
    aj = clamp(ceil(Int, (s.agent[2]+p.grid_lim)*factor), 1, n)
    ti = clamp(ceil(Int, (s.target[1]+p.grid_lim)*factor), 1, n)
    tj = clamp(ceil(Int, (s.target[2]+p.grid_lim)*factor), 1, n)
    return sub2ind((n,n,n,n), ai, aj, ti, tj)
end
function convert_s(::Type{TagState}, s::Int, p::DiscreteCVDPTagProblem)
    n = p.n_bins
    factor = 2*p.grid_lim/n
    ai, aj, ti, tj = ind2sub((n,n,n,n), s)
    return TagState((Vec2(ai, aj)-0.5)*factor-p.grid_lim, (Vec2(ti, tj)-0.5)*factor-p.grid_lim)
end

# action
function convert_a(::Type{Int}, a::Float64, p::DiscreteCVDPTagProblem)
    i = ceil(Int, a*p.n_angles/(2*pi))
    while i > p.n_angles
        i -= p.n_angles
    end
    while i < 1
        i += p.n_angles
    end
    return i
end
convert_a(::Type{Float64}, a::Int, p::DiscreteCVDPTagProblem) = (a-0.5)*2*pi/p.n_angles

function convert_a(T::Type{Int}, a::TagAction, p::DiscreteCVDPTagProblem)
    i = convert_a(T, a.angle, p)
    if a.look
        return i + p.n_angles
    else
        return i
    end
end
function convert_a(::Type{TagAction}, a::Int, p::DiscreteCVDPTagProblem)
    return TagAction(a > p.n_angles, convert_a(Float64, a % p.n_angles, p))
end

# observation
function convert_o(::Type{IVec8}, o::Vec8, p::AODiscreteCVDPTagPOMDP)
    return floor.(Int, (o./p.binsize)::Vec8)::IVec8
end
# convert_o(::Type{Vec8}, o::Int, p::DiscreteCVDPTagProblem) = (o-0.5)*2*pi/p.n_obs_angles

n_states(p::AODiscreteCVDPTagPOMDP) = Inf
n_actions(p::DiscreteCVDPTagProblem) = 2*p.n_angles
POMDPs.discount(p::DiscreteCVDPTagProblem) = discount(cproblem(p))
isterminal(p::DiscreteCVDPTagProblem, s) = isterminal(cproblem(p), convert_s(TagState, s, p))

POMDPs.actions(p::DiscreteCVDPTagProblem) = 1:n_actions(p)

function POMDPs.gen(p::DiscreteCVDPTagProblem, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(actiontype(cproblem(p)), a, p)
    return gen(cproblem(p), s, ca, rng)
end

function POMDPs.gen(p::ADiscreteCVDPTagPOMDP, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(actiontype(cproblem(p)), a, p)
    sor = @gen(:sp,:o,:r)(cproblem(p), s, ca, rng)
    return (sp = sor[1], o = sor[2], r = sor[3])
end

function POMDPs.observation(p::ADiscreteCVDPTagPOMDP, s::TagState, a::Int, sp::TagState)
    ca = convert_a(actiontype(cproblem(p)), a, p)
    return POMDPs.observation(cproblem(p), s, ca, sp)
end

function POMDPs.gen(p::AODiscreteCVDPTagPOMDP, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(actiontype(cproblem(p)), a, p)
    csor = @gen(:sp,:o,:r)(cproblem(p), s, ca, rng)
    return (sp=csor[1], o=convert_o(IVec8, csor[2], p), r=csor[3])
end

function POMDPs.observation(p::AODiscreteCVDPTagPOMDP, s::TagState, a::Int, sp::TagState)
    ImplicitDistribution(p, s, a, sp) do p, s, a, sp, rng
        ca = convert_a(actiontype(cproblem(p)), a, p)
        co = rand(rng, observation(cproblem(p), s, ca, sp))
        return convert_o(IVec8, co, p)
    end
end

POMDPs.initialstate(p::DiscreteCVDPTagProblem) = VDPInitDist()

#=
gauss_cdf(mean, std, x) = 0.5*(1.0+erf((x-mean)/(std*sqrt(2))))
function obs_weight(p::AODiscreteCVDPTagPOMDP, a::Int, sp::TagState, o::Int)
    cp = cproblem(p)
    @assert cp.bearing_std <= 2*pi/6.0 "obs_weight assumes σ <= $(2*pi/6.0)"
    ca = convert_a(actiontype(cp), a, p)
    co = convert_o(obstype(cp), o, p) # float between 0 and 2pi
    upper = co + 0.5*2*pi/p.n_angles
    lower = co - 0.5*2*pi/p.n_angles
    if ca.look
        diff = sp.target - sp.agent
        bearing = atan(diff[2], diff[1])
        # three cases: o is in bin, below, or above
        if bearing <= upper && bearing > lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            prob = cdf_up - cdf_low
        elseif bearing <= lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            below_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper-2*pi)
            below_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower-2*pi)
            prob = cdf_up - cdf_low + below_cdf_up - below_cdf_low
        else # bearing > upper
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            above_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper+2*pi)
            above_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower+2*pi)
            prob = cdf_up - cdf_low + above_cdf_up - above_cdf_low
        end
        return prob
    else
        return 1.0
    end
end

function obs_weight(p::ADiscreteCVDPTagPOMDP, a::Int, sp::TagState, o::Float64)
    ca = convert_a(TagAction, a, p)
    return obs_weight(cproblem(p), ca, sp, o)
end
=#
