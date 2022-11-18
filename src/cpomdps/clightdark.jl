
# compare CLightDark1D with fixed bad_action_budget and 0 p.problem.incorrect_r with 
# LightDark1D with variable p.incorrect_r

@with_kw struct CLightDark1D <: ConstrainPOMDPWrapper
    pomdp::LightDark1D = LightDark1D(0.9, 10.0, 0., 1.0, 0.0, default_sigma) # default 0 incorrect_r, goes into cost
    bad_action_budget::Float64 = 0.5
end
    
costs(::CLightDark1D, s::LightDark1DState, a::Int) = Float64[a==0 && abs(s.y) >= 1]
costs_limit(p::CLightDark1D) = [p.bad_action_budget]
n_costs(::CLightDark1D) = 1
max_reward(p::CLightDark1D) = p.problem.correct_r
min_reward(p::CLightDark1D) = -p.movement_cost