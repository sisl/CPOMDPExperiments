using Pkg
Pkg.activate(".")
println("Running install file")

github_packages = [
    PackageSpec(url="https://github.com/jamgochiana/POMCPOW.jl", rev="1ba776a77eb887ab412bad1df85915283252b531"),
    PackageSpec(url="https://github.com/JuliaPOMDP/RockSample.jl", rev="4f8112b975d71c59ee6b67721bb1d41c06ad1334"),
    PackageSpec(url="https://github.com/jamgochiana/VDPTag2.jl", rev="53b08f5279b0d680c8741265ba1361aedaac1278"),
    PackageSpec(url="https://github.com/sisl/SpillpointPOMDP.jl", rev="976d171d370046f9991ce47d939c9256659a3404")
]
println("Adding external packages")
Pkg.add(github_packages)

include("develop.jl")