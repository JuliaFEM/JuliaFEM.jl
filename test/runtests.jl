using JuliaFEM
using Base.Test

# write your own tests here
@test 1 == JuliaFEM.test()


@test_approx_eq 1.0 1.0
