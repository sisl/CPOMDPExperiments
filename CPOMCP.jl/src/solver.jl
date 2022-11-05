function POMDPTools.action_info(p::CPOMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = CPOMCPTree(p.problem, b, p.solver.tree_queries)
        policy = search(p, b, tree, info)
        info[:policy] = policy
        a = tree.a_labels[rand(p.rng, policy)]
        p._cost_mem = dot(tree.top_level_costs[policy.vals], policy.probs)
        p._tree = tree
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        info[:exception] = ex
    end
    return a, info
end

action(p::CPOMCPPlanner, b) = first(action_info(p, b))

abstract type AlphaSchedule end

struct ConstantAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
ConstantAlphaSchedule() = ConstantAlphaSchedule(1.e-3)
alpha(sched::ConstantAlphaSchedule, ::Int) = sched.scale

struct InverseAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
InverseAlphaSchedule() = InverseAlphaSchedule(1.)
alpha(sched::InverseAlphaSchedule, query::Int) = sched.scale/query

function search(p::CPOMCPPlanner, b, t::CPOMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    max_clip = (max_reward(p.problem) - min_reward(p.problem))/discount(p.problem) ./ p.tau
    p._lambda = rand(p.rng, p._tree.n_costs) .* max_clip # random initialization

    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, CPOMCPObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end

        # dual ascent w/ clipping
        ha = rand(p.rng, action_policy_UCB(CPOMCPObsNode(t,1), p._lambda, 0.0, 0.0))
        p._lambda += alpha(p.solver.alpha, i) .* (t.cv[ha] - p.budget)
        p._lambda = min.(max.(p._lambda, 0.), max_clip)

    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    return action_policy_UCB(CPOMCPObsNode(t,1), p._lambda, 0.0, p.solver.nu)
end

# return sparse categorical policy over best action node indices
function action_policy_UCB(hnode::CPOMCPObsNode, lambda::Vector{Float64}, c::Float64, nu::Float64)
    t = hnode.tree
    h = hnode.node

    # Q_lambda = Q_value - lambda'Q_c + c sqrt(log(N)/N(h,a))
    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    criterion_values = sizehint!(Float64[],length(t.children[h]))
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node] - dot(lambda,t.ch[node])
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] - dot(lambda,t.ch[node])
            if c > 0
                criterion_value += c*sqrt(ltn/n)
            end
        end
        push!(criterion_values,criterion_value)
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    
    # get next best nodes
    val_diff = criterion_values .- best_criterion_val
    next_best_nodes = t.children[h][0 .< val_diff .< nu]
    append!(best_nodes, next_best_nodes)
    
    # weigh actions
    if length(best_nodes) == 1
        weights = [1.0]
    else
        weights = solve_lp(t, best_nodes)
    end
    return SparseCat(best_nodes, weights)
end

function solve_lp(t::CPOMCPTree, best_nodes::Vector{Int})
    error("Multiple CPOMCP best actions not implemented")
end

function simulate(p::CPOMCPPlanner, s, hnode::CPOMCPObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end

    acts = action_policy_UCB(hnode, p._lambda, p.solver.c, p.solver.nu)
    
    ha = rand(p.rng, acts)
    a = t.a_labels[ha]

    sp, o, r, c = @gen(:sp, :o, :r, :c)(p.problem, s, a, p.rng)
    
    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, sp, o)
        v, cv = estimate_value(p.solved_estimator,
                           p.problem,
                           sp,
                           CPOMCPObsNode(t, hao),
                           steps-1)
    else
        v, cv = simulate(p, sp, CPOMCPObsNode(t, hao), steps-1)
    end
    R = r + discount(p.problem)*v
    C = c + discount(p.problem)*cv

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]
    t.cv[ha] += (C-t.cv[ha])/t.n[ha]

    # top level cost estimator
    if steps == p.solver.max_depth
        t.top_level_costs[ha] += (c-t.top_level_costs[ha])/t.n[ha]
    end
    return R, C
end
