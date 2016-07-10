# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "renumber element nodes" begin
    mesh = Mesh()
    add_element!(mesh, 1, :Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mapping = Dict{Symbol, Vector{Int}}(
        :Tet10 => [1, 2, 4, 3, 5, 6, 7, 8, 9, 10])
    reorder_element_connectivity!(mesh, mapping)
    @test mesh.elements[1] == [1, 2, 4, 3, 5, 6, 7, 8, 9, 10]
    invmapping = Dict{Symbol, Vector{Int}}()
    invmapping[:Tet10] = invperm(mapping[:Tet10])
    reorder_element_connectivity!(mesh, invmapping)
    @test mesh.elements[1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
end

