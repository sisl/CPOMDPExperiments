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
        a = tree.a_labels[rand(p.rng,policy)]
        tlcs = map(i->p.top_level_costs[i],policy.vals)
        p._cost_mem = dot(tlcs,policy.probs)
        if pomcp.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        a = convert(A, default_action(pomcp.solver.default_action, pomcp.problem, b, ex))
    end
    return a, info
end

action(pomcp::CPOMCPOWPlanner, b) = first(action_info(pomcp, b))

function POMDPPolicies.actionvalues(p::CPOMCPOWPlanner, b)
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
    t0 = timer()
    while i < pomcp.solver.tree_queries
        i += 1
        s = rand(pomcp.solver.rng, tree.root_belief)
        if !POMDPs.isterminal(pomcp.problem, s)
            max_depth = min(pomcp.solver.max_depth, ceil(Int, log(pomcp.solver.eps)/log(discount(pomcp.problem))))
            simulate(pomcp, CPOWTreeObsNode(tree, 1), s, max_depth)
            all_terminal = false
        end
        if timer() - t0 >= pomcp.solver.max_time
            break
        end
    end
    info[:search_time] = timer() - t0
    info[:tree_queries] = i

    if all_terminal
        throw(AllSamplesTerminal(tree.root_belief))
    end

    best_node = select_best(pomcp.solver.final_criterion, CPOWTreeObsNode(tree,1), pomcp.solver.rng)

    return tree.a_labels[best_node]
end

function simulate(pomcp::CPOMCPOWPlanner, h_node::CPOWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, CPOWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), CPOWTreeObsNode(tree, h))
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
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
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

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
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
        R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])

        R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    return R
end

function solve(solver::CPOMCPOWSolver, problem::POMDP)
    return CPOMCPOWPlanner(solver, problem)
end

struct MaxUCB
    c::Float64
end

function select_best(crit::MaxUCB, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_criterion_val = -Inf
    local best_node::Int
    istied = false
    local tied::Vector{Int}
    ltn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node]
        if isinf(tree.v[node])
            criterion_value = tree.v[node]
        elseif n == 0
            criterion_value = Inf
        else
            criterion_value = tree.v[node] + crit.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
            istied = false
        elseif criterion_value == best_criterion_val
            if istied
                push!(tied, node)
            else
                istied = true
                tied = [best_node, node]
            end
        end
    end
    if istied
        return rand(rng, tied)
    else
        return best_node
    end
end

struct MaxQ end

function select_best(crit::MaxQ, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_v = tree.v[best_node]
    @assert !isnan(best_v)
    for node in tree.tried[h][2:end]
        if tree.v[node] >= best_v
            best_v = tree.v[node]
            best_node = node
        end
    end
    return best_node
end

struct MaxTries end

function select_best(crit::MaxTries, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_n = tree.n[best_node]
    @assert !isnan(best_n)
    for node in tree.tried[h][2:end]
        if tree.n[node] >= best_n
            best_n = tree.n[node]
            best_node = node
        end
    end
    return best_node
end
