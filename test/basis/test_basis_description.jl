# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

struct _CoverageDummyBasis <: AbstractBasis end

@testset "Basis description API" begin
    desc = JuliaFEM.VandermondeBasisDescription(
        name="unit-test",
        description="coverage",
        topology=Triangle{3},
        family=Lagrange{1},
        ansatz=(),
    )
    @test JuliaFEM.basis_family(desc) === Lagrange{1}
    @test JuliaFEM.basis_topology(desc) === Triangle{3}
    @test JuliaFEM.basis_order(desc) === 1

    desc_s = JuliaFEM.VandermondeBasisDescription(
        name="ser",
        description="coverage",
        topology=Quadrilateral{4},
        family=Serendipity{2},
        ansatz=(),
    )
    @test JuliaFEM.basis_order(desc_s) === 2

    desc_bad = JuliaFEM.VandermondeBasisDescription(
        name="bad",
        description="coverage",
        topology=Triangle{3},
        family=_CoverageDummyBasis,
        ansatz=(),
    )
    @test_throws ErrorException JuliaFEM.basis_order(desc_bad)

    xi = Vec{2}((0.2, 0.25))
    tri = Triangle{3}()
    lag = Lagrange{1}()
    vals = get_basis_functions(tri, lag, xi)
    dvals = get_basis_derivatives(tri, lag, xi)
    @test get_basis_function(tri, lag, xi, 1) == vals[1]
    @test get_basis_derivative(tri, lag, xi, 2) == dvals[2]
end
