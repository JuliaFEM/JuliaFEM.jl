# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "Material field traits" begin
    E3 = required_material_fields(Elasticity{3}())
    E2 = required_material_fields(Elasticity{2}())
    @test E2 isa DataType
    z3 = create_zero_field(E3)
    @test z3.σ == zero(z3.σ)

    T3 = required_material_fields(Thermal{3}())
    zt = create_zero_field(T3)
    @test zt.q == zero(zt.q)

    mat = LinearElastic(E=210e9, ν=0.3)
    @test material_field_type(mat) === required_material_fields(Elasticity{3}())

    @test_throws ErrorException required_material_fields(Elasticity{1}())
    @test_throws ErrorException required_material_fields(Thermal{1}())
end
