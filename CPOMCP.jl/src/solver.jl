make_tree(s::CPOMCPSolver, p::CPOMDP, b) = CPOMCPTree(p, b, s.tree_queries)
make_tree(s::CPOMCPDPWSolver, p::CPOMDP, b) = CPOMCPDPWTree(p, b, s.tree_queries)

function POMDPTools.action_info(p::AbstractCPOMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = make_tree(p.solver, p.problem, b)
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

action(p::AbstractCPOMCPPlanner, b) = first(action_info(p, b))

function search(p::AbstractCPOMCPPlanner, b, t::AbstractCPOMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    max_clip = (max_reward(p.problem) - min_reward(p.problem))/(1-discount(p.problem)) ./ p._tau
    #p._lambda = rand(p.rng, t.n_costs) .* max_clip # random initialization
    p._lambda = zeros(Float64, t.n_costs)
    
    if p.solver.search_progress_info
        info[:lambda] = sizehint!(Vector{Float64}[p._lambda], p.solver.tree_queries)
        info[:v_best] = sizehint!(Float64[], p.solver.tree_queries)
        info[:cv_best] = sizehint!(Vector{Float64}[], p.solver.tree_queries)
        info[:v_taken] = sizehint!(Float64[], p.solver.tree_queries)
        info[:cv_taken] = sizehint!(Vector{Float64}[], p.solver.tree_queries)
    end

    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, CPOMCPObsNode(t,1), p.solver.max_depth)
            all_terminal = false
        end

        # dual ascent w/ clipping
        ha = rand(p.rng, action_policy_UCB(CPOMCPObsNode(t,1), p._lambda, 0.0, 0.0))
        p._lambda += alpha(p.solver.alpha_schedule, i) .* (t.cv[ha] - p.budget)
        p._lambda = min.(max.(p._lambda, 0.), max_clip)
        
        # tracking
        if p.solver.search_progress_info
            push!(info[:lambda], p._lambda)
            push!(info[:v_taken], t.v[ha])
            push!(info[:cv_taken], t.cv[ha])

            # get absolute best node (no lambda weights)
            max_q = -Inf
            ha_best = nothing
            for nd in t.children[1]
                if t.v[nd] > max_q
                    max_q = t.v[nd]
                    ha_best = nd
                end
            end
            push!(info[:v_best],t.v[ha_best] )
            push!(info[:cv_best],t.cv[ha_best] )
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    return action_policy_UCB(CPOMCPObsNode(t,1), p._lambda, 0.0, p.solver.nu)
end

dot(a::Vector,b::Vector) = sum(a .* b)

# return sparse categorical policy over best action node indices
function action_policy_UCB(hnode::CPOMCPObsNode, lambda::Vector{Float64}, c::Float64, nu::Float64)
    t = hnode.tree
    h = hnode.node

    # Q_lambda = Q_value - lambda'Q_c + c sqrt(log(N)/N(h,a))
    ltn = log(t.total_n[h])
    best_nodes = Int[]
    criterion_values = sizehint!(Float64[],length(t.children[h]))
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node] - dot(lambda,t.cv[node])
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] - dot(lambda,t.cv[node])
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
    if nu > 0.
        val_diff = best_criterion_val .- criterion_values 
        next_best_nodes = t.children[h][0 .< val_diff .< nu]
        append!(best_nodes, next_best_nodes)
    end
    
    # weigh actions
    if length(best_nodes) == 1
        weights = [1.0]
    else
        weights = solve_lp(t, best_nodes)
    end
    return SparseCat(best_nodes, weights)
end

function solve_lp(t::AbstractCPOMCPTree, best_nodes::Vector{Int})
    # error("Multiple CPOMCP best actions not implemented")
    # random for now
    return ones(Float64, length(best_nodes)) / length(best_nodes)
end

function simulate(p::CPOMCPPlanner, s, hnode::CPOMCPObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0, zeros(Float64, hnode.tree.n_costs)
    end

    t = hnode.tree
    h = hnode.node
    acts = action_policy_UCB(hnode, p._lambda, p.solver.c, p.solver.nu)
    p._best_node_mem = acts.vals
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

function simulate(p::CPOMCPDPWPlanner, s, hnode::CPOMCPObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0, zeros(Float64, hnode.tree.n_costs)
    end
    sol = p.solver
    t = hnode.tree
    h = hnode.node
    top_level = steps==p.solver.max_depth
    # action pw
    if sol.enable_action_pw
        if length(t.children[h]) <= sol.k_action*t.total_n[h]^sol.alpha_action
            a = next_action(p.next_action, p.problem, s, hnode)
            if !sol.check_repeat_act || !haskey(t.a_loookup,(h,a))
                insert_action_node!(t,h,a;top_level=top_level)
            end
        end
    elseif isempty(t.children[h]) 
        for a in actions(p.problem, s)
            insert_action_node!(t,h,a;top_level=top_level)
        end
    end

    t.total_n[h] += 1
    acts = action_policy_UCB(hnode, p._lambda, sol.c, sol.nu)
    p._best_node_mem = acts.vals
    ha = rand(p.rng, acts)
    a = t.a_labels[ha]

    # observation progressive widening
    new_node = false
    if (sol.enable_observation_pw && t.n_a_children[ha] <= sol.k_observation*t.n[ha]^sol.alpha_observation) || t.n_a_children[ha] == 0
        sp, o, r, c = @gen(:sp, :o, :r, :c)(p.problem, s, a, p.rng)
        if sol.check_repeat_obs && haskey(t.o_lookup, (ha,o))
            hao = t.o_lookup[(ha,o)]
            push!(t.states[hao],sp)
        else
            hao = insert_obs_node!(t, p.problem, ha, sp, o)
            new_node = true
        end
        
        push!(t.transitions[ha],hao)

        if !sol.check_repeat_obs
            t.n_a_children[ha] += 1
        elseif !((ha,hao) in t.unique_transitions)
            push!(t.unique_transitions, (ha,hao))
            t.n_a_children[ha] += 1
        end
    else
        hao= rand(p.rng,t.transitions[ha])
        sp = rand(p.rng,t.states[hao])
        r = reward(p.problem,s,a,sp)
        c = costs(p.problem,s,a,sp)
    end
    
    if new_node
        v, cv = estimate_value(p.solved_estimator, p.problem, sp,
            CPOMCPObsNode(t, hao),steps-1)
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
        tlcs = length(t.top_level_costs)
        if ha <= tlcs
            t.top_level_costs[ha] += (c-t.top_level_costs[ha])/t.n[ha]
        else
            error("Improper indexing of top level costs")
        end
    end
    return R, C
end