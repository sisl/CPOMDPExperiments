using Revise
using Pkg
Pkg.activate(".")
Pkg.resolve()
using CPOMDPExperiments
test("rocksample","pomcp")
test("vdptag","pomcpow")
#test("lightdark1d","pomcpow")
#test("roomba","pft")
#run_all_tests()
