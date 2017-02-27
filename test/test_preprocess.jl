# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

datadir = first(splitext(basename(@__FILE__)))

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

@testset "add_nodes! and add_elements!" begin
    mesh = Mesh()
    dic = Dict(1 => [1.,1.,1.], 2 => [2.,2.,2])
    add_nodes!(mesh, dic)
    @test mesh.nodes == dic

    vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    add_elements!(mesh,Dict(1=>(:Tet10,vec),
                            11=>(:Tet10,vec)))
    @test mesh.elements[1] == vec
    @test mesh.elements[11] == vec
end

@testset "find nearest nodes from mesh" begin
    meshfile = joinpath(datadir, "block_2d.med")
    mesh = aster_read_mesh(meshfile)
    create_node_set_from_element_set!(mesh, "LOWER_LEFT", "UPPER_BOTTOM")
    # nid 1 coords = (0.0, 0.5), nid 13 coords = (0.0, 0.5)
    nid = find_nearest_node(mesh, [0.0, 0.5]; node_set="LOWER_LEFT")
    @test first(nid) == 1
    nid = find_nearest_node(mesh, [0.0, 0.5]; node_set="UPPER_BOTTOM")
    @test first(nid) == 13
end
