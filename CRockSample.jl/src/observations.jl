const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::RockSampleCPOMDP) = 1:3
POMDPs.obsindex(pomdp::RockSampleCPOMDP, o::Int) = o

function POMDPs.observation(pomdp::RockSampleCPOMDP, a::Int, s::CRSState)
    if a <= N_BASIC_ACTIONS
        # no obs
        return SparseCat((1,2,3), (0.0,0.0,1.0)) # for type stability
    else
        rock_ind = a - N_BASIC_ACTIONS 
        rock_pos = pomdp.rocks_positions[rock_ind]
        dist = norm(rock_pos - s.pos)
        efficiency = 0.5*(1.0 + exp(-dist*log(2)/pomdp.sensor_efficiency))
        rock_state = s.rocks[rock_ind]
        if rock_state
            return SparseCat((1,2,3), (efficiency, 1.0 - efficiency, 0.0))
        else
            return SparseCat((1,2,3), (1.0 - efficiency, efficiency, 0.0))
        end
    end
end