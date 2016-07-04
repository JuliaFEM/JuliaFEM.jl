# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

function calc_model(mesh_name; with_volume_load=false, debug_print=false)
    meshfile = Pkg.dir("JuliaFEM")*"/test/testdata/3d_block.med"
    mesh = aster_read_mesh(meshfile, mesh_name)

    block = Problem(Elasticity, "BLOCK", 3)
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false
    block.elements = create_elements(mesh, "BLOCK")
    update!(block, "youngs modulus", 288.0)
    update!(block, "poissons ratio", 1/3)
    with_volume_load && update!(block, "displacement load 3", 576.0)

    traction = Problem(Elasticity, "traction force", 3)
    traction.properties.finite_strain = false
    traction.properties.geometric_stiffness = false
    traction.elements = create_elements(mesh, "LOAD")
    update!(traction, "displacement traction force 3", 288.0)

    bc = Problem(Dirichlet, "symmetry boundary condition", 3, "displacement")
    symyz = create_elements(mesh, "SYMYZ")
    symxz = create_elements(mesh, "SYMXZ")
    symxy = create_elements(mesh, "SYMXY")
    update!(symyz, "displacement 1", 0.0)
    update!(symxz, "displacement 2", 0.0)
    update!(symxy, "displacement 3", 0.0)
    push!(bc, symyz, symxz, symxy)

    solver = LinearSolver("Solver block problem")
    push!(solver, block, traction, bc)
    solver()

    max_u = maximum(block.assembly.u)
    nu = round(Int, length(block.assembly.u)/3)
    u = reshape(block.assembly.u, 3, nu)
    if debug_print
        f = reshape(full(block.assembly.f), 3, nu)
        dump(round(u', 5))
        dump(round(f', 5))
        info("max |u| = $max_u")
    end
    return block, u
end


@testset "test 3d block HEX8" begin
    block, u = calc_model("BLOCK_HEX8"; with_volume_load=true)
    @test isapprox(maximum(u), 2.0)
end

@testset "test 3d block TET4" begin
#   block, u = calc_model("BLOCK_TET4", :TE4, :TR3; with_volume_load=true)
#   @test isapprox(maximum(u), 2.1329516539440205)
    block, u = calc_model("BLOCK_TET4"; with_volume_load=false)
    @test isapprox(maximum(u), 1.0)
end

@testset "test 3d block TET10" begin
    block, u = calc_model("BLOCK_TET10"; with_volume_load=false)
#    @test isapprox(maximum(u), 2.13656216413056)
    @test isapprox(maximum(u), 1.0)
end

