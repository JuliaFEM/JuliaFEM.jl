# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

module RandomFieldTests

using JuliaFEM
using JuliaFEM: DiscreteField, Field, Increment, Quad4
using JuliaFEM.Test

type RandomField <: DiscreteField
    mu :: Float64
    std :: Float64
end

Base.first(field::RandomField) = Increment(randn(2, 4).*field.std^2 + field.mu)


function test_interpolate_in_time()
    r = RandomField(10.0, 0.0)
    f = Increment(ones(2, 4)*10.0)
    @test r(0.0) == f
    @test r(-Inf) == f
    @test r(+Inf) == f
    @test r(1.0) == f
end

function test_interpolate_in_spatial_domain()
    basis = Quad4([1, 2, 3, 4]).basis
    r = RandomField(10.0, 0.0)
    feval = basis(r(0.0), [0.0, 0.0])
    @test feval == [10.0, 10.0]
end

end
