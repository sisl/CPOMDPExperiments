# various utility functions

"""
Return a vector of action=>Q pairs ranked by Q.

For debugging/ checking purposes - this may be slow
"""
function ranked_actions(planner::AbstractCMCTSPlanner, state)
    @assert planner.tree != nothing "in ranked_actions, planner did not have a tree; use keep_tree=true when constructing the solver"
    ranked_actions(planner.tree, state)
end

function ranked_actions(tree::CMCTSTree, state)
    sanodes = children(StateNode(tree, tree.state_map[state]))
    sorted_sanodes = sort(collect(sanodes), by=q, rev=true)
    return [action(n)=>q(n) for n in sorted_sanodes]
end

function ranked_actions(tree::CDPWTree, state)
    snode = tree.s_lookup[state]
    sanodes = tree.children[snode]
    tuples = [(tree.a_labels[n], tree.q[n]) for n in sanodes]
    return sort!(tuples, by=last, rev=true)
end
