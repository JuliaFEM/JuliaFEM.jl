# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM.Testing

#=
@testset "find dofs given a set of nodes" begin
    nodes = [1, 3]
    dim = 3
    dofs = find_dofs_by_nodes(dim, nodes)
    @test dofs == [1, 2, 3, 7, 8, 9]
end

@testset "find nodes given a set of dofs" begin
    dofs = [2, 8, 9]
    dim = 3
    nodes = find_nodes_by_dofs(dim, dofs)
    @test nodes == [1, 3]

    dofs = [2, 12]
    dim = 2
    nodes = find_nodes_by_dofs(dim,dofs)
    @test nodes == [1, 6]

end
=#
