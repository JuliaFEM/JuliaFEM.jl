using JuliaFEM
using Base.Test

function test_create_model()
    m = new_model()
    fn = fieldnames(m)
    @assert :model in fn
    @assert :nodes in fn
    @assert :elements in fn
    @assert :element_nodes in fn
    @assert :element_gauss_points in fn
end

# write your own tests here
# @test 1 == JuliaFEM.test()
# @test_approx_eq 1.0 1.0
