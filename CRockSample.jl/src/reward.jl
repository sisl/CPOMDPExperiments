function POMDPs.reward(pomdp::RockSampleCPOMDP, s::CRSState, a::Int)
    r = pomdp.step_penalty
    if next_position(s, a)[1] > pomdp.map_size[1]
        r += pomdp.exit_reward
        return r
    end

    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        r += s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty 
    elseif a > N_BASIC_ACTIONS # using senssor
        r += pomdp.sensor_use_penalty
    end
    return r
end

function CPOMDPs.costs(pomdp::RockSampleCPOMDP, s::CRSState, a::Int)
    c = 0.
    if next_position(s, a)[1] > pomdp.map_size[1]
        return (c)
    end
    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        c += s.rocks[rock_ind] ? 0. : 1.
    end
    return (c)

end

function CPOMDPs.cost_limits(pomdp::RockSampleCPOMDP)
    return (pomdp.bad_rock_cost_limit)
end