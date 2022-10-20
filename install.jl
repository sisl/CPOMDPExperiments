using Pkg

packages = [
    # Misc github example
    # PackageSpec(url="https://github.com/sisl/AdversarialDriving.jl"),

    # [deps] CPOMDPs.jl
    PackageSpec(url=joinpath(@__DIR__, "CPOMDPs.jl")),

    # [deps] CPOMCP.jl
    PackageSpec(url=joinpath(@__DIR__, "CPOMCP.jl")),

    # [deps] CMCTS.jl
    PackageSpec(url=joinpath(@__DIR__, "CMCTS.jl")),

    # [deps] CPOMCPOW.jl
    PackageSpec(url=joinpath(@__DIR__, "CPOMCPOW.jl")),

    # [deps] CPOMDPExperiments.jl
    PackageSpec(url=joinpath(@__DIR__)),
]

ci = haskey(ENV, "CI") && ENV["CI"] == "true"

if ci
    # remove "own" package when on CI
    pop!(packages)
end

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)

# if ci
#    # pytorch does not work with 3.9
#    pkg"add Conda"
#    using Conda
#    Conda.add("python=3.7.5")
#    Conda.add("pytorch")
# end

Pkg.add("D3Trees")
#Pkg.add("BSON")
Pkg.add("Distributions")
#Pkg.add("Infiltrator")
#Pkg.add("POMDPModelTools")
#Pkg.add("POMDPPolicies")
#Pkg.add("POMDPs")
#Pkg.add("Parameters")
#Pkg.add("ProgressMeter")
#Pkg.add("POMDPSimulators")
#Pkg.add("OrderedCollections")
Pkg.add("POMDPGifs")
Pkg.add("Cairo")