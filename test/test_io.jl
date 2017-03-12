# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
using LightXML

@testset "create new Xdmf object" begin
    r = Xdmf()
    expected = "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"3.0\"/>"
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

@testset "write data to HDF, automatically generate path" begin
    xdmf = Xdmf()
    di1 = new_dataitem(xdmf, [1 2 3])
    di2 = new_dataitem(xdmf, [4 5 6])
    @test contains(content(di1), "DataItem_1")
    @test contains(content(di2), "DataItem_2")
end

@testset "Xdmf filtering" begin
    grid1 = new_element("Grid")
    add_text(grid1, "I am first grid")
    grid2 = new_element("Grid")
    add_text(grid2, "I am second grid")
    set_attribute(grid2, "Name", "Frame 2")
    grid3 = new_element("Grid")
    add_text(grid3, "I am third grid")
    grids = [grid1, grid2, grid3]

    @test content(xdmf_filter(grids, "Grid")) == "I am first grid"
    @test content(xdmf_filter(grids, "Grid[1]")) == "I am first grid"
    @test content(xdmf_filter(grids, "Grid[2]")) == "I am second grid"
    @test content(xdmf_filter(grids, "Grid[3]")) == "I am third grid"
    @test content(xdmf_filter(grids, "Grid[end]")) == "I am third grid"
    @test content(xdmf_filter(grids, "Grid[@Name=Frame 2]")) == "I am second grid"
    @test xdmf_filter(grids, "Grid[0]") == nothing
    @test xdmf_filter(grids, "Grid[4]") == nothing
    @test xdmf_filter(grids, "Grid[@Name=Frame 3]") == nothing
    @test xdmf_filter(grids, "Domain/Grid[@Name=Frame 3]") == nothing
    @test xdmf_filter(grids, "Domain") == nothing
end

@testset "XML traverse" begin
    xdmf = Xdmf()
    domain = new_child(xdmf.xml, "Domain")
    grid = new_child(domain, "Grid")
    set_attribute(grid, "CollectionType", "Temporal")
    set_attribute(grid, "GridType", "Collection")

    frame1 = new_child(grid, "Grid")
    time1 = new_child(frame1, "Time")
    set_attribute(time1, "Value", 0.0)
    X1 = new_child(frame1, "Geometry")
    set_attribute(X1, "Type", "XY")

    frame2 = new_child(grid, "Grid")
    set_attribute(frame2, "Name", "Frame 2")
    time2 = new_child(frame2, "Time")
    set_attribute(time2, "Value", 1.0)
    X2 = new_child(frame2, "Geometry")
    set_attribute(X2, "Type", "XY")

    add_child(grid, frame1)
    add_child(grid, frame2)

    dataitem = new_dataitem(xdmf, "/Domain/Grid/Grid/2/Geometry", [1.0, 2.0])
    add_child(X2, dataitem)

    println(xdmf.xml)
    @test read(xdmf, "/Domain/Grid/Grid/Time/Value") == "0.0"
    @test read(xdmf, "/Domain/Grid/Grid[2]/Time/Value") == "1.0"
    @test read(xdmf, "/Domain/Grid/Grid[end]/Time/Value") == "1.0"
    @test read(xdmf, "/Domain/Grid/Grid[@Name=Frame 2]/Time/Value") == "1.0"
    @test isapprox(read(xdmf, "/Domain/Grid/Grid[2]/Geometry/DataItem"), [1.0, 2.0])
end
