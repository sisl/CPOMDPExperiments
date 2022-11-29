using Revise
using CPOMDPExperiments
using D3Trees
using Plots 
using Infiltrator
using LaTeXStrings
using JLD2, FileIO

p = CLightDarkNew(cost_budget=0.1)

base_kwargs = Dict(:tree_queries=>1e5, 
    :k_observation => 5., # 0.1,
    :alpha_observation => 1/15, #0.5,
    :enable_action_pw=>false,
    :max_depth => 10,)

c = 90.0
nu = 0.0
psolver_kwargs = Dict(
    :criterion=>CPOMDPExperiments.POMCPOW.MaxUCB(c), 
    :estimate_value=>zero_V,
    )
csolver_kwargs = Dict(
    :criterion=>CPOMDPExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    :alpha_schedule => CPOMDPExperiments.CPOMCPOW.ConstantAlphaSchedule(0.5),
    :estimate_value=>zeroV_trueC,
    )

lambdas = [0., 0.01, 0.03, 0.1, 0.3, .5, .7, 1, 1.5, 2, 3, 10, 30, 100]

le = run_lambda_experiments(lambdas,
    p, 
    CPOMDPExperiments.POMCPOWSolver, merge(base_kwargs,psolver_kwargs), 
    CPOMDPExperiments.CPOMCPOWSolver, merge(base_kwargs,csolver_kwargs),
    CPOMDPExperiments.CPOMCPOWBudgetUpdateWrapper;
    nsims=100,filter_size=Int(1e4),run_cpomdps=true)

function get_hull(x,y,p)
    p_hull=[p[1]]
    i_on = 1
    n = length(p)
    while true
        i_check = i_on+1
        max_slope = 0.
        i_add = nothing
        for i_check in (i_on+1):n
            slope = (y[p[i_check]] - y[p[i_on]]) /(x[p[i_check]] - x[p[i_on]])
            if slope > max_slope
                max_slope = slope
                i_add = i_check
            end
        end
        if !(i_add===nothing)
            push!(p_hull,p[i_add])
            i_on = i_add
        else
            break
        end
    end

    @assert length(p_hull)>1            
    return p_hull
end

function plot_lambdas(le::LambdaExperiments;target_cost::Union{Float64,Nothing}=nothing,
    saveloc::Union{String,Nothing}=nothing)

    x = [i.mean for i in le.Cs]
    x_stds = [i.std for i in le.Cs]
    y = [i.mean for i in le.Rs]
    y_stds = [i.std for i in le.Rs]
    p = sortperm(x)

    f=scatter(x[p],y[p],xerror=x_stds[p], yerror=y_stds[p], label="POMCPOW(λ)", legend=:bottomright) 
    if !(le.C_CPOMDP===nothing)
        scatter!([le.C_CPOMDP.mean],[le.R_CPOMDP.mean],
            xerror=[le.C_CPOMDP.std], yerror=[le.R_CPOMDP.std],label="CPOMDPOW")
    end
    
    title!("Constrained LightDark Pareto Frontier")
    xlabel!(L"V_{C}")
    ylabel!(L"V_{R}")

    if !(target_cost===nothing)
        vline!([target_cost],label="Cost Constraint",linecolor=:black, linestyle=:dash)
    end

    p_hull = get_hull(x,y,p)
    plot!(x[p_hull].-.01,y[p_hull].+ 1,linecolor=:red, linewidth=4,label="POMCPOW Pareto Frontier")
    
    if !(saveloc===nothing)
        savefig(f,saveloc)
    end
end

function save_le(le::LambdaExperiments,saveloc::String)
    d = Dict(
        "lambdas"=>le.λs,
        "POMDP_C_mean"=>[i.mean for i in le.Cs],
        "POMDP_C_std"=>[i.std for i in le.Cs],
        "POMDP_R_mean"=>[i.mean for i in le.Rs],
        "POMDP_R_std"=>[i.std for i in le.Rs] ,
        "CPOMDP_C_mean"=> !(le.C_CPOMDP===nothing) ? le.C_CPOMDP.mean : nothing,
        "CPOMDP_C_std"=> !(le.C_CPOMDP===nothing) ? le.C_CPOMDP.std : nothing,
        "CPOMDP_R_mean"=> !(le.C_CPOMDP===nothing) ? le.R_CPOMDP.mean : nothing,
        "CPOMDP_R_std"=> !(le.C_CPOMDP===nothing) ? le.R_CPOMDP.std : nothing,
        "CPOMDP_minC_C_mean"=> !(le.C_CPOMDP_minC===nothing) ? le.C_CPOMDP_minC.mean : nothing,
        "CPOMDP_minC_C_std"=> !(le.C_CPOMDP_minC===nothing) ? le.C_CPOMDP_minC.std : nothing,
        "CPOMDP_minC_R_mean"=> !(le.C_CPOMDP_minC===nothing) ? le.R_CPOMDP_minC.mean : nothing,
        "CPOMDP_minC_R_std"=> !(le.C_CPOMDP_minC===nothing) ? le.R_CPOMDP_minC.std : nothing,
        )

    save(saveloc,d)
end

plot_lambdas(le;target_cost=0.1,saveloc="figs/ld_lambdas.pdf")
   
save_le(le,"figs/pareto_info.jld2")