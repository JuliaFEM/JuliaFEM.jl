# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Testing

@testset "read ascii mesh" begin
    mesh = """
    COOR_2D
    N1          0.0 0.0
    N2          1.0 0.0
    N3          1.0 1.0
    N4          0.0 1.0
    FINSF

    QUAD4
    E1         N1 N2 N3 N4
    FINSF

    SEG2
    E2         N3 N4
    FINSF

    GROUP_NO NOM=NALL
    N1 N2
    FINSF

    GROUP_MA NOM=BODY1
    E1 E2
    FINSF

    FIN
    """

#   m = parse(mesh, Val{:CODE_ASTER_MAIL})
#   @test m["nodes"]["N1"] == [0.0, 0.0]
#   @test m["elements"]["E1"] == ["QUAD4", ["N1", "N2", "N3", "N4"]]
#   @test m["elements"]["E2"] == ["SEG2", ["N3", "N4"]]
#   @test m["elsets"]["BODY1"] == ["E1", "E2"]
#   @test m["nsets"]["NALL"] == ["N1", "N2"]
end

@testset "parse nodes" begin
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

function JuliaFEM.get_mesh(::Type{Val{Symbol("block_2d_1elem_quad4")}})
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d_1elem_quad4.med"
    mesh = aster_read_mesh(fn)
    return mesh
end

@testset "test reading aster .med file" begin
    mesh = get_mesh("block_2d_1elem_quad4")
    #=
    info("nodes")
    for (k, v) in mesh.nodes
        info("$k => $v")
    end
    info("node sets")
    for (k, v) in mesh.node_sets
        info("$k => $v")
    end
    info("elements")
    for (k, v) in mesh.elements
        info("$k => $v, type = $(mesh.element_types[k])")
    end
    info("element sets")
    for (k, v) in mesh.element_sets
        info("$k => $v")
    end
    =#
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
    mesh = get_mesh("block_2d_1elem_quad4")
    mesh2 = filter_by_element_set(mesh, :BLOCK)
    @test haskey(mesh2.element_sets, :BLOCK)
    @test length(mesh2.elements) == 1
end

function calculate_volume(mesh_name, eltype)
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(fn, mesh_name)
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

