using Revise
using Pkg
Pkg.activate(".")

using CPOMDPExperiments
test("rocksample","pomcp")
#test("lightdark1d","pomcpow")
#test("roomba","pft")
#run_all_tests()
