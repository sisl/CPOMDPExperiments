# A fixed action policy which always takes the action `move east`.
struct CRSExitSolver <: Solver end
struct CRSExit <: Policy
    exit_return::Vector{Float64}
end
POMDPs.solve(::CRSExitSolver, m::RockSampleCPOMDP) = CRSExit([discount(m)^(m.map_size[1]-x) * m.exit_reward for x in 1:m.map_size[1]])
POMDPs.solve(solver::CRSExitSolver, m::UnderlyingMDP{P}) where P <: RockSampleCPOMDP = solve(solver, m.pomdp)
POMDPs.value(p::CRSExit, s::CRSState) = s.pos[1] == -1 ? 0.0 : p.exit_return[s.pos[1]]

function POMDPs.value(p::CRSExit, b::AbstractParticleBelief)
    utility = 0.0
    for (i, s) in enumerate(particles(b))
        if s.pos[1] != -1 # if s is not terminal
            utility += weight(b, i) * p.exit_return[s.pos[1]]
        end
    end
    return utility / weight_sum(b)
end
POMDPs.action(p::CRSExit, b) = 2 # Move east

# Dedicated MDP solver for RockSample
struct CRSMDPSolver <: Solver
    include_Q::Bool
end
CRSMDPSolver(;include_Q=false) = CRSMDPSolver(include_Q)
POMDPs.solve(solver::CRSMDPSolver, m::RockSampleCPOMDP) = solve(solver, UnderlyingMDP(m))
function POMDPs.solve(solver::CRSMDPSolver, m::UnderlyingMDP{P}) where P <: RockSampleCPOMDP
    util = rs_mdp_utility(m.pomdp)
    if solver.include_Q
        return solve(ValueIterationSolver(init_util=util, include_Q=true), m)
    else
        return ValueIterationPolicy(m, utility=util, include_Q=false)
    end
end

# Dedicated QMDP solver for RockSample
struct CRSQMDPSolver <: Solver end
function POMDPs.solve(::CRSQMDPSolver, m::RockSampleCPOMDP)
    vi_policy = solve(CRSMDPSolver(include_Q=true), m)
    return AlphaVectorPolicy(m, vi_policy.qmat, vi_policy.action_map)
end

# Solve for the optimal utility of RockSample, assuming full observability.
function rs_mdp_utility(m::RockSampleCPOMDP{K}) where K
    util = zeros(length(states(m)))
    discounts = discount(m) .^ (0:(m.map_size[1]+m.map_size[2]-2))

    # Rewards for exiting.
    exit_returns = [discounts[m.map_size[1] - x + 1] * m.exit_reward for x in 1:m.map_size[1]]

    # Calculate the optimal utility for states having no good rocks, which is the exit return.
    rocks = falses(K)
    for x in 1:m.map_size[1]
        for y in 1:m.map_size[2]
            util[stateindex(m, CRSState(CRSPos(x,y), SVector{K,Bool}(rocks)))] = exit_returns[x]
        end
    end

    # The optimal utility of states having k good rocks can be derived from the utility of states having k-1 good rocks:
    # Utility_k = max(ExitReturn, argmax_{r∈GoodRocks}(γ^{Manhattan distance to r}Utility_{k-1}))
    for good_rock_num in 1:K
        for good_rocks in combinations(1:K, good_rock_num)
            rocks = falses(K)
            for good_rock in good_rocks
                rocks[good_rock] = true
            end
            for x in 1:m.map_size[1]
                for y in 1:m.map_size[2]
                    best_return = exit_returns[x]
                    for good_rock in good_rocks
                        dist_to_good_rock = abs(x - m.rocks_positions[good_rock][1]) + abs(y - m.rocks_positions[good_rock][2])
                        rocks[good_rock] = false
                        sample_return = discounts[dist_to_good_rock+1] * (m.good_rock_reward + discounts[2] * util[stateindex(m, CRSState(m.rocks_positions[good_rock], SVector{K,Bool}(rocks)))])
                        rocks[good_rock] = true
                        if sample_return > best_return
                            best_return = sample_return
                        end
                    end
                    util[stateindex(m, CRSState(CRSPos(x,y), SVector{K,Bool}(rocks)))] = best_return
                end
            end
        end
    end

    return util
end
