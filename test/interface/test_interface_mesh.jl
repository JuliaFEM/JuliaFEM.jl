# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using Test

@testset "InterfaceMesh + InterfaceDOFHandler" begin
    nodes = [Vec(0.0, 0.0, 0.0), Vec(0.5, 0.0, 0.0), Vec(1.0, 0.0, 0.0)]
    conn = [(UInt32(1), UInt32(2)), (UInt32(2), UInt32(3))]
    coup = [
        InterfaceVolumeCoupling(UInt32(1), UInt32(1), UInt8(1), UInt32(2), UInt32(1), UInt8(1)),
        InterfaceVolumeCoupling(UInt32(1), UInt32(2), UInt8(1), UInt32(2), UInt32(2), UInt8(1)),
    ]
    im = InterfaceMesh(Seg2, nodes, conn, coup)

    @test topology_type(im) === Seg2
    @test interface_nnodes(im) == 3
    @test interface_nelements(im) == 2

    Sλ = @DOFSet{λ::DOF{Float64, Cell}}
    els_λ, h_λ = create_interface_elements!(im, Element{Seg2, Lagrange{1}, Sλ})
    @test h_λ.total_dofs == 2
    @test element_dofs(els_λ[1])[1] == 1
    @test element_dofs(els_λ[2])[1] == 2
    @test h_λ.dof_connectivity.n_total_dofs == 2

    Su = @DOFSet{u::DOF{Float64, Vertex}}
    els_u, h_u = create_interface_elements!(im, Element{Seg2, Lagrange{1}, Su})
    @test h_u.total_dofs == 3
    @test element_dofs(els_u[1]) == (UInt64(1), UInt64(2))
    @test element_dofs(els_u[2]) == (UInt64(2), UInt64(3))
    @test h_u.dof_connectivity.dof_to_elements[2][1].elem_id == 1
    @test h_u.dof_connectivity.dof_to_elements[2][2].elem_id == 2
end
