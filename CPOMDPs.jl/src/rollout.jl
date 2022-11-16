# ConstrainedRolloutSimulator

"""
    ConstrainedRolloutSimulator(rng, max_steps)
    ConstrainedRolloutSimulator(; <keyword arguments>)
A fast simulator that just returns the reward and costs
The simulation will be terminated when either
1) a terminal state is reached (as determined by `isterminal()` or
2) the discount factor is as small as `eps` or
3) max_steps have been executed
# Keyword arguments:
- rng: A random number generator to use.
- eps: A small number; if γᵗ where γ is the discount factor and t is the time step becomes smaller than this, the simulation will be terminated.
- max_steps: The maximum number of steps to simulate.
# Usage (optional arguments in brackets):
    ro = ConstrainedRolloutSimulator()
    history = simulate(ro, pomdp, policy, [updater [, init_belief [, init_state]]])
See also: [`HistoryRecorder`](@ref), [`run_parallel`](@ref)
"""
struct ConstrainedRolloutSimulator{RNG<:AbstractRNG} <: Simulator
    rng::RNG

    # optional: if these are null, they will be ignored
    max_steps::Union{Nothing,Int}
    eps::Union{Nothing,Float64}
end

ConstrainedRolloutSimulator(rng::AbstractRNG, d::Int=typemax(Int)) = ConstrainedRolloutSimulator(rng, d, nothing)
function RolloutSimulator(;rng=Random.GLOBAL_RNG,
                           eps=nothing,
                           max_steps=nothing)
    return RolloutSimulator{typeof(rng)}(rng, max_steps, eps)
end


POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy) begin
    @req updater(::typeof(policy))
    bu = updater(policy)
    @subreq simulate(sim, pomdp, policy, bu)
end

POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy, bu::Updater) begin
    @req initialstate(::typeof(pomdp))
    dist = initialstate(pomdp)
    @subreq simulate(sim, pomdp, policy, bu, dist)
end

function simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy, bu::Updater=updater(policy))
    dist = initialstate(pomdp)
    return simulate(sim, pomdp, policy, bu, dist)
end


POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy, updater::Updater, initial_belief) begin
    @req rand(::typeof(sim.rng), ::typeof(initial_belief))
    @subreq simulate(sim, pomdp, policy, updater, initial_belief, s)
end

function simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP{S}, policy::Policy, updater::Updater, initial_belief) where {S}
    s = rand(sim.rng, initial_belief)::S
    return simulate(sim, pomdp, policy, updater, initial_belief, s)
end

POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy, updater::Updater, initial_belief, s) begin
    P = typeof(pomdp)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    @req initialize_belief(::typeof(updater), ::typeof(initial_belief))
    @req isterminal(::P, ::S)
    @req discount(::P)
    @req n_costs(::P)
    @req gen(::P, ::S, ::A, ::typeof(sim.rng))
    b = initialize_belief(updater, initial_belief)
    @req action(::typeof(policy), ::typeof(b))
    @req update(::typeof(updater), ::typeof(b), ::A, ::O)
end

function simulate(sim::ConstrainedRolloutSimulator, pomdp::CPOMDP, policy::Policy, updater::Updater, initial_belief, s)
    
    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    disc = 1.0
    r_total = 0.0
    c_total = zeros(Float64, n_costs(pomdp))
    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > eps && !isterminal(pomdp, s) && step <= max_steps

        a = action(policy, b)

        sp, o, r, c = @gen(:sp,:o,:r,:c)(pomdp, s, a, sim.rng)

        r_total += disc*r
        c_total += disc*c 
        s = sp

        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total, c_total
end

POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, mdp::CMDP, policy::Policy) begin
    istate = initialstate(mdp, sim.rng)
    @subreq simulate(sim, mdp, policy, istate)
end

POMDPLinter.@POMDP_require simulate(sim::ConstrainedRolloutSimulator, mdp::CMDP, policy::Policy, initialstate) begin
    P = typeof(mdp)
    S = typeof(initialstate)
    A = actiontype(mdp)
    @req isterminal(::P, ::S)
    @req action(::typeof(policy), ::S)
    @req gen(::P, ::S, ::A, ::typeof(sim.rng))
    @req discount(::P)
    @req n_costs(::P)
end

function simulate(sim::ConstrainedRolloutSimulator, mdp::CMDP, policy::Policy)
    istate = rand(sim.rng, initialstate(mdp))
    simulate(sim, mdp, policy, istate)
end

function simulate(sim::ConstrainedRolloutSimulator, mdp::CMDP{S}, policy::Policy, initialstate::S) where {S}
    
    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initialstate

    disc = 1.0
    r_total = 0.0
    c_total = zeros(Float64, n_costs(mdp))
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r, c = @gen(:sp,:r,:c)(mdp, s, a, sim.rng)

        r_total += disc*r
        c_total += disc*c
        s = sp

        disc *= discount(mdp)
        step += 1
    end

    return r_total, c_total
end

function simulate(sim::ConstrainedRolloutSimulator, m::CPOMDP{S}, policy::Policy, initialstate::S) where {S}
    simulate(sim, UnderlyingCMDP(m), policy, initialstate)
end

simulate(sim::ConstrainedRolloutSimulator, m::CMDP, p::Policy, is) = simulate(sim, m, p, convert(statetype(m), is))

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


# stepthrough functions
POMDPTools.Simulators.default_spec(T::Type{M}) where M <: CMDP = tuple(:s, :a, :sp, :r, :c, :info, :t, :action_info)
POMDPTools.Simulators.default_spec(T::Type{M}) where M <: CPOMDP = tuple(:s, :a, :sp, :o, :r, :c, :info, :t, :action_info, :b, :bp, :update_info)
POMDPTools.Simulators.convert_spec(spec, T::Type{M}) where {M<:CPOMDP} = POMDPTools.Simulators.convert_spec(spec, Set(tuple(:s, :a, :sp, :o, :r, :c, :info, :bp, :b, :action_info, :update_info, :t)))
POMDPTools.Simulators.convert_spec(spec, T::Type{M}) where {M<:CMDP} = POMDPTools.Simulators.convert_spec(spec, Set(tuple(:s, :a, :sp, :r, :c, :info, :action_info, :t)))

function Base.iterate(it::POMDPTools.Simulators.MDPSimIterator{SPEC,M}, is::Tuple{Int, S}=(1, it.init_state)) where {SPEC, M<:CMDP, S}
    if isterminal(it.mdp, is[2]) || is[1] > it.max_steps 
        return nothing 
    end 
    t = is[1]
    s = is[2]
    a, ai = action_info(it.policy, s)
    out = @gen(:sp,:r,:c,:info)(it.mdp, s, a, it.rng)
    nt = merge(NamedTuple{(:sp,:r,:c,:info)}(out), (t=t, s=s, a=a, action_info=ai))
    return (POMDPTools.Simulators.out_tuple(it, nt), (t+1, nt.sp))
end

function Base.iterate(it::POMDPTools.Simulators.POMDPSimIterator{SPEC,M}, is::Tuple{Int,S,B} = (1, it.init_state, it.init_belief)) where {SPEC, M<:CPOMDP, S,B}
    if isterminal(it.pomdp, is[2]) || is[1] > it.max_steps 
        return nothing 
    end 
    t = is[1]
    s = is[2]
    b = is[3]
    a, ai = action_info(it.policy, b)
    out = @gen(:sp,:o,:r,:c,:info)(it.pomdp, s, a, it.rng)
    outnt = NamedTuple{(:sp,:o,:r,:c,:info)}(out)
    bp, ui = update_info(it.updater, b, a, outnt.o)
    nt = merge(outnt, (t=t, b=b, s=s, a=a, action_info=ai, bp=bp, update_info=ui))
    return (POMDPTools.Simulators.out_tuple(it, nt), (t+1, nt.sp, nt.bp))
end