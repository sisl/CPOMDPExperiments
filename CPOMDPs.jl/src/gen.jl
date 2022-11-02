# generator interface 

function POMDPs.sorted_deppairs(m::Type{<:CPOMDPs.CMDP}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :r => [:s, :a, :sp],
                :c => [:s, :a, :sp],
                :info => Symbol[]
               )
    return sorted_deppairs(deps, symbols)
end

function POMDPs.sorted_deppairs(m::Type{<:CPOMDPs.CPOMDP}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :o => [:s, :a, :sp],
                :r => [:s, :a, :sp, :o],
                :c => [:s, :a, :sp, :o],
                :info => Symbol[]
               )
    return sorted_deppairs(deps, symbols)
end

POMDPs.node_expr(::Val{:c}, depargs) = :(costs(m, $(depargs...)))