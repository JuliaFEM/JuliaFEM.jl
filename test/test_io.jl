# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
using LightXML

@testset "create new Xdmf object" begin
    r = Xdmf()
    expected = "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.1\"/>"
    @test string(r.xml) == expected
end

@testset "put and get to Xdmf, low level" begin
    io = Xdmf()
    # h5
    write(io.hdf, "/Xdmf/Domain/Geometry", [1 2 3])
    @test isapprox(read(io.hdf, "/Xdmf/Domain/Geometry"), [1 2 3])
    # xml
    obj = new_child(io.xml, "Domain")
    set_attribute(obj, "Name", "Test Domain")
    obj2 = find_element(io.xml, "Domain")
    @test attribute(obj2, "Name") == "Test Domain"
end

@testset "put and get to xdmf" begin
    xdmf = Xdmf()
    domain = new_child(xdmf, "Domain")
    grid = new_child(domain, "Grid")
    grid["CollectionType"] = "Temporal"
    grid["GridType"] = "Collection"

    frame1 = new_child(grid, "Grid")
    new_child(frame1, "Time", "Value" => 0.0)
    X1 = new_child(frame1, "Geometry", Dict("Type" => "XY"))

    frame2 = new_child(grid, "Grid", "Name" => "Frame 2")
    new_child(frame2, "Time", "Value" => 1.0)
    X2 = new_child(frame2, "Geometry", "Type" => "XY")

    add_child(grid, frame1)
    add_child(grid, frame2)

    dataitem = new_dataitem(xdmf, "/Domain/Grid/Grid/2/Geometry", [1.0, 2.0])
    add_child(X2, dataitem)

    println(xdmf.xml)
    @test has_child(xdmf.xml, "Domain")
    @test isa(get_child(xdmf.xml, "Domain"), XMLElement)
    @test !has_attribute(xdmf.xml, "Domain")
    @test isapprox(read(xdmf, "/Domain/Grid/Grid/Time/Value"), 0.0)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[2]/Time/Value"), 1.0)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[end]/Time/Value"), 1.0)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[@Name=Frame 2]/Time/Value"), 1.0)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[2]/Geometry/DataItem"), [1.0, 2.0])
end

@testset "save results to disk" begin
    X = Dict{Int, Vector{Float64}}(
        1 => [0.0,0.0],
        2 => [1.0,0.0],
        3 => [1.0,1.0],
        4 => [0.0,1.0])
    element = Element(Quad4, [1, 2, 3, 4])
    update!(element, "geometry", X)
    update!(element, "temperature thermal conductivity", 6.0)
    update!(element, "temperature load", 0.0 => 12.0)
    update!(element, "temperature load", 1.0 => 18.0)
    problem = Problem(Heat, "one element heat problem", 1)
    problem.properties.formulation = "2D"
    push!(problem, element)
    boundary_element = Element(Seg2, [1, 2])
    update!(boundary_element, "geometry", X)
    update!(boundary_element, "temperature 1", 0.0)
    bc = Problem(Dirichlet, "fixed", 1, "temperature")
    push!(bc, boundary_element)
    solver = Solver(Linear, problem, bc)
    solver.xdmf = Xdmf()

    solver.time = 0.0
    solver()
    empty!(problem.assembly)
    solver.time = 1.0
    solver()

    info(solver("temperature", 0.0))
    info(solver("temperature", 1.0))
    info(element("temperature load", [0.0, 0.0], 0.0))
    info(element("temperature load", [0.0, 0.0], 1.0))

    xdmf = get(solver.xdmf)
    info("h5 file = $(h5file(xdmf))")
    E = read(xdmf.hdf, "/Topology/Quad4/Element IDs")
    C = read(xdmf.hdf, "/Topology/Quad4/Connectivity")
    N = read(xdmf.hdf, "/Node IDs")
    X = read(xdmf.hdf, "/Geometry")
    T1 = read(xdmf.hdf, "/Results/Time 0.0/Nodal Fields/Temperature")
    T2 = read(xdmf.hdf, "/Results/Time 1.0/Nodal Fields/Temperature")
    @test isapprox(E, [-1])
    @test isapprox(C, [0 1 2 3])
    @test isapprox(N, [1, 2, 3, 4])
    X_expected = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]'
    T1_expected = [0.0 0.0 1.0 1.0]
    T2_expected = [0.0 0.0 0.5 0.5]
    @test isapprox(X, X_expected)
    @test isapprox(T1, T1_expected)
    @test isapprox(T2, T2_expected)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid/Time/Value"), 0.0)
    @test read(xdmf, "/Domain/Grid/Grid/Geometry/Type") == "XY"
    @test isapprox(read(xdmf, "/Domain/Grid/Grid/Geometry/DataItem"), X_expected)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid/Topology/DataItem"), [0 1 2 3])
    @test isapprox(read(xdmf, "/Domain/Grid/Grid/Topology[@TopologyType=Polyline]/DataItem"), [0 1])
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[1]/Attribute[@Name=Temperature]/DataItem"), T1_expected)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[2]/Attribute[@Name=Temperature]/DataItem"), T2_expected)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[end]/Time/Value"), 1.0)
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[end]/Topology/DataItem"), [0 1 2 3])
end
