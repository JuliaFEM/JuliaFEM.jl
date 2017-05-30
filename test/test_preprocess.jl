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

@testset "code aster / parse nodes" begin
    section = """
    N9  2.0 3.0 4.0
    COOR_3D
    N1          0.0 0.0 0.0
    N2          1.0 0.0 0.0
    N3          1.0 1.0 0.0
    N4          0.0 1.0 0.0
    N5          0.0 0.0 1.0
    N6          1.0 0.0 1.0
    N7          1.0 1.0 1.0
    N8          0.0 1.0 1.0
    FINSF
    absdflasdf
    N12 3.0 4.0 5.0 6.0
    N13 3.0 4.0 5.0
    """
    nodes = aster_parse_nodes(section)
    @test nodes[1] == Float64[0.0, 0.0, 0.0]
    @test nodes[8] == Float64[0.0, 1.0, 1.0]
    @test length(nodes) == 8
end

@testset "test reading aster .med file" begin
    meshfile = joinpath(datadir, "block_2d_1elem_quad4.med")
    mesh = aster_read_mesh(meshfile)
    @test length(mesh.element_sets) == 5
    @test length(mesh.node_sets) == 4
    @test length(mesh.elements) == 5
    @test length(mesh.nodes) == 4
    for elset in [:BLOCK, :TOP, :BOTTOM, :LEFT, :RIGHT]
        @test haskey(mesh.element_sets, elset)
        @test length(mesh.element_sets[elset]) == 1
    end
    for nset in [:TOP_LEFT, :TOP_RIGHT, :BOTTOM_LEFT, :BOTTOM_RIGHT]
        @test haskey(mesh.node_sets, nset)
        @test length(mesh.node_sets[nset]) == 1
    end
end

@testset "test filter by element set" begin
    mesh = aster_read_mesh(joinpath(datadir, "block_2d_1elem_quad4.med"))
    mesh2 = filter_by_element_set(mesh, :BLOCK)
    @test haskey(mesh2.element_sets, :BLOCK)
    @test length(mesh2.elements) == 1
end

function calculate_volume(mesh_name, eltype)
    mesh_file = joinpath(datadir, "primitives.med")
    mesh = aster_read_mesh(mesh_file, mesh_name)
    elements = create_elements(mesh; element_type=eltype)
    V = 0.0
    time = 0.0
    for element in elements
        for ip in get_integration_points(element)
            detJ = element(ip, time, Val{:detJ})
            detJ > 0 || warn("negative determinant for element $eltype !")
            V += ip.weight*detJ
        end
    end
    info("volume of $eltype is $V")
    return V
end

@testset "calculate volume for 1 element models" begin
    @test isapprox(calculate_volume("TRIANGLE_TRI3_1", :Tri3), 1/2)
    @test isapprox(calculate_volume("TRIANGLE_TRI6_1", :Tri6), 1/2)
    @test isapprox(calculate_volume("TRIANGLE_TRI7_1", :Tri7), 1/2)
    @test isapprox(calculate_volume("SQUARE_QUAD4_1", :Quad4), 2^2)
    @test isapprox(calculate_volume("SQUARE_QUAD8_1", :Quad8), 2^2)
    @test isapprox(calculate_volume("SQUARE_QUAD9_1", :Quad9), 2^2)
    @test isapprox(calculate_volume("TETRA_TET4_1", :Tet4), 1/6)
    @test isapprox(calculate_volume("TETRA_TET10_1", :Tet10), 1/6)
#   @test isapprox(calculate_volume("TETRA_TET14_1", :Tet14), 1/6)
    @test isapprox(calculate_volume("CUBE_HEX8_1", :Hex8), 2^3)
    @test isapprox(calculate_volume("CUBE_HEX20_1", :Hex20), 2^3)
    @test isapprox(calculate_volume("CUBE_HEX27_1", :Hex27), 2^3)
    @test isapprox(calculate_volume("WEDGE_WEDGE6_1", :Wedge6), 1)
#   @test isapprox(calculate_volume("WEDGE_WEDGE15_1", :Wedge15, 1/2))
#   @test isapprox(calculate_volume("PYRAMID_PYRAMID5_1", :Pyramid5, ?))
#   @test isapprox(calculate_volume("PYRAMID_PYRAMID13_1", :Pyramid13, ?))
end

@testset "read nodal field from code aster result file" begin
    rmedfile = joinpath(datadir, "rings.rmed")
    rmed = JuliaFEM.Preprocess.RMEDFile(rmedfile)
    temp = JuliaFEM.Preprocess.aster_read_data(rmed, "TEMP")
    @test isapprox(temp[15], 1.0)
    @test isapprox(temp[95], 2.0)
end

using JuliaFEM.Preprocess: MEDFile, get_element_sets

@testset "test read element sets from med file, issue #111" begin
    meshfile = joinpath(datadir, "hexmeshOverlappingGroups.med")
    med = MEDFile(meshfile)
    element_sets = get_element_sets(med, "Mesh_1")
    @test element_sets[-10] == ["halfhex", "mosthex"]
    @test element_sets[-11] == ["halfhex"]
end

using JuliaFEM.Preprocess: aster_read_mesh

@testset "test read overlapping ets, issue #111" begin
    mesh_file = joinpath(datadir, "hexmeshOverlappingGroups.med")
    mesh = aster_read_mesh(mesh_file, "Mesh_1")
    @test length(mesh.element_sets[:mosthex]) == 273
    @test length(mesh.element_sets[:halfhex]) == 147
end
