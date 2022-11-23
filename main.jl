using Pkg
Pkg.activate(".")
using Revise
Pkg.resolve()
using CPOMDPExperiments

#test("rocksample","pomcp")

#test("lightdark1d","pomcp-ow", test_pomdp=false)
#test("lightdark1d","pft-ow")
#test("vdptag", "pomcp-dpw", test_pomdp=false)
# test("vdptag","pft-dpw")
#test("vdptag","pomcpow)

#test("spillpoint", "pomcp-dpw", test_pomdp=false)
#test("spillpoint", "pft-dpw")
#test("spillpoint", "pomcpow")

#test("roomba","pft-dpw", test_cpomdp=false)
#run_all_tests()
