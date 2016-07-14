# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "1d strain" begin
    X = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [1.0, 1.0, 1.0])
    u = Dict{Int64, Vector{Float64}}(
        1 => [0.0, 0.0, 0.0],
        2 => [1.0, 1.0, 1.0])
    element = Element(Seg2, [1, 2])
    update!(element, "geometry", X)
    detJ = element([0.0], 0.0, Val{:detJ})
    info("detJ = $detJ")
    @test isapprox(detJ, sqrt(3)/2)
    J = element([0.0], 0.0, Val{:Jacobian})
    info("J = $J")
    @test isapprox(J, [0.5 0.5 0.5])
    update!(element, "displacement", u)
    # FIXME
#   gradu = element("displacement", [0.0], 0.0, Val{:Grad})
#   info("1d bar: ∇u = $gradu")
end
