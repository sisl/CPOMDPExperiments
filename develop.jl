using Pkg
Pkg.activate(".")
dev_packages = [
    PackageSpec(url=joinpath(@__DIR__, "CPOMDPs.jl")),
    PackageSpec(url=joinpath(@__DIR__, "CPOMCP.jl")),
    PackageSpec(url=joinpath(@__DIR__, "CMCTS.jl")),
    PackageSpec(url=joinpath(@__DIR__, "CPOMCPOW.jl")),
]
println("Adding dev packages")
Pkg.develop(dev_packages)