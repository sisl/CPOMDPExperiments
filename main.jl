using Pkg
Pkg.activate(".")
using Revise
Pkg.resolve()
using CPOMDPExperiments
#test("rocksample","pomcp")
test("lightdark1d","pft-ow")
#test("vdptag","pft-dpw")
#test("spillpoint", "pft-dpw")

test("lightdark1d","pomcpow-ow")
#test("vdptag","pomcpow)
test("spillpoint", "pomcpow")

#test("roomba","pft")
#run_all_tests()
