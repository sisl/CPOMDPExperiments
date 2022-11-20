dot(a::Vector,b::Vector) = sum(a .* b)

function action_info(pomcp::CPOMCPOWPlanner{P,NBU}, b; tree_in_info=false) where {P,NBU}
    A = actiontype(P)
    info = Dict{Symbol, Any}()
    tree = make_tree(pomcp, b)
    pomcp.tree = tree
    local a::A
    try
        policy = search(pomcp, tree, info)
        info[:policy] = policy
        a = tree.a_labels[rand(pomcp.solver.rng,policy)]
        tlcs = map(i->tree.top_level_costs[i],policy.vals)
        pomcp._cost_mem = dot(tlcs,policy.probs)
        if pomcp.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

action(pomcp::CPOMCPOWPlanner, b) = first(action_info(pomcp, b))

function POMDPTools.actionvalues(p::CPOMCPOWPlanner, b)
    tree = make_tree(p, b)
    search(p, tree)
    values = Vector{Union{Float64,Missing}}(missing, length(actions(p.problem)))
    for anode in tree.tried[1]
        a = tree.a_labels[anode]
        values[actionindex(p.problem, a)] = tree.v[anode]
    end
    return values
end

function make_tree(p::CPOMCPOWPlanner{P, NBU}, b) where {P, NBU}
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    B = belief_type(NBU,P)
    return CPOMCPOWTree{B, A, O, typeof(b)}(b, 2*min(100_000, p.solver.tree_queries), n_costs(p.problem))
    # return CPOMCPOWTree{B, A, O, typeof(b)}(b, 2*p.solver.tree_queries)
end

function search(pomcp::CPOMCPOWPlanner, tree::CPOMCPOWTree, info::Dict{Symbol,Any}=Dict{Symbol,Any}())
    timer = pomcp.solver.timer
    all_terminal = true
    # gc_enable(false)
    i = 0
    max_clip = (max_reward(pomcp.problem) - min_reward(pomcp.problem))/(1-discount(pomcp.problem)) ./ pomcp._tau
    pomcp._lambda = rand(pomcp.solver.rng, tree.n_costs) .* max_clip # random initialization
    t0 = timer()

    while i < pomcp.solver.tree_queries
        i += 1
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            max_depth = min(pomcp.solver.max_depth, ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem))))
            simulate(pomcp, CPOWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end

        # dual ascent with clipping
        ha = rand(pomcp.solver.rng, select_best(MaxCUCB(0.,0.),CPOWTreeObsNode(tree,1),pomcp._lambda))
        pomcp._lambda += alpha(pomcp.solver.alpha_schedule,i) .*  (tree.cv[ha] - pomcp.budget)
        pomcp._lambda = min.(max.(pomcp._lambda, 0.), max_clip)

        if timer() - t0 >= pomcp.solver.max_time
            break
        end
    end
    info[:search_time] = timer() - t0
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    return select_best(pomcp.solver.final_criterion, CPOWTreeObsNode(tree,1), pomcp._lambda)
end

function simulate(pomcp::CPOMCPOWPlanner, h_node::CPOWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0, zeros(Float64, tree.n_costs)
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, CPOWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, CStateBelief(tree.sr_beliefs[h]), CPOWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            init_C(pomcp.init_C, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, CStateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            init_C(pomcp.init_C, pomcp.problem, CPOWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = rand(sol.rng, select_best(pomcp.criterion, h_node, pomcp._lambda))
    a = tree.a_labels[best_node]

    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)

        sp, o, r, c = @gen(:sp, :o, :r, :c)(pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r, c))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
    else

        sp, r, c = @gen(:sp, :r, :c)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("CPOMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        v, cv = estimate_value(pomcp.solved_estimate, pomcp.problem, sp, CPOWTreeObsNode(tree, hao), d-1)
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r, c)
        sp, r, c = rand(sol.rng, tree.sr_beliefs[hao])

        v, cv = simulate(pomcp, CPOWTreeObsNode(tree, hao), sp, d-1)
    end
    R = r + POMDPs.discount(pomcp.problem)*v
    C = c + POMDPs.discount(pomcp.problem)*cv

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end
    if tree.cv[best_node] != -Inf
        tree.cv[best_node] += (C-tree.cv[best_node])/tree.n[best_node]
    end

    # top level costs update
    if h==1
        if !(best_node in keys(tree.top_level_costs))
            tree.top_level_costs[best_node] = c 
        else
            tree.top_level_costs[best_node] += (c-tree.top_level_costs[best_node])/tree.n[best_node]
        end
    end
    return R, C
end

function solve(solver::CPOMCPOWSolver, problem::POMDP)
    return CPOMCPOWPlanner(solver, problem)
end

struct MaxCUCB
    c::Float64
    nu::Float64
end

function select_best(crit::MaxCUCB, h_node::CPOWTreeObsNode, lambda::Vector{Float64})
    tree = h_node.tree
    h = h_node.node
    ltn = log(tree.total_n[h])
    best_nodes = Int[]
    criterion_values = sizehint!(Float64[],length(tree.tried[h]))
    best_criterion_val = -Inf
    for node in tree.tried[h]
        n = tree.n[node]
        if isinf(tree.v[node])
            criterion_value = tree.v[node] 
        elseif n == 0
            criterion_value = Inf
        else
            criterion_value = tree.v[node] + crit.c*sqrt(ltn/n) - dot(lambda,tree.cv[node])
        end
        push!(criterion_values, criterion_value)
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end

    # get next best nodes
    if crit.nu > eps(Float32)
        val_diff = best_criterion_val .- criterion_values
        next_best_nodes = tree.tried[h][0 .< val_diff .< crit.nu]
        append!(best_nodes, next_best_nodes)
    end

    # weigh actions
    if length(best_nodes) == 1
        weights = [1.0]
    else
        weights = solve_lp(tree, best_nodes)
    end
    return SparseCat(best_nodes, weights)  
end

function solve_lp(t::CPOMCPOWTree, best_nodes::Vector{Int})
    # error("Multiple CPOMCP best actions not implemented")
    # random for now
    return ones(Float64, length(best_nodes)) / length(best_nodes)
end