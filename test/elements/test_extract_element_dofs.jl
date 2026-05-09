# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM
using Tensors

@testset "extract_element_dofs" begin
    mesh = create_structured_box_mesh(Hex8, nx=1, ny=1, nz=1)
    S = @DOFSet{T::DOF{Temperature, Vertex}, u::DOF{Displacement{3}, Vertex}}
    ET = Element{Hex8, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)
    u = zeros(Float64, handler.total_dofs)
    for i in eachindex(u)
        u[i] = Float64(i)
    end

    elem = elements[1]
    flat = extract_element_dofs(elem, u)
    @test flat.T isa NTuple{8,Float64}
    @test flat.u isa NTuple{24,Float64}
    @test flat.T[1] == u[elem.dof_indices[1]]

    structed = extract_element_dofs_structured(elem, u)
    @test structed.T isa NTuple{8,Float64}
    @test structed.u isa NTuple{8,Vec{3,Float64}}
    @test structed.u[1][1] == flat.u[1]
end
