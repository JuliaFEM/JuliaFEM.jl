# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Test

function get_model(fn, vol, sur)
    meshfile = Pkg.dir("JuliaFEM")*"/geometry/3d_blocks/BLOCK.med"
    mesh = parse_aster_med_file(meshfile, fn)

    block = Problem(Elasticity, fn, 3)
    block.properties.finite_strain = false

    elements = aster_create_elements(mesh, :BLOCK, vol)
    update!(elements, "youngs modulus", 288.0)
    update!(elements, "poissons ratio", 1/3)
    update!(elements, "displacement load 3", 576.0)
    push!(block, elements...)

    traction = aster_create_elements(mesh, :LOAD, sur)
    update!(traction, "displacement traction force 3", 288.0)
    push!(block, traction...)

    bc = Problem(Dirichlet, "symmetry boundary condition", 3, "displacement")
#   bc.properties.formulation = :incremental
    symyz = aster_create_elements(mesh, :SYMYZ, sur)
    symxz = aster_create_elements(mesh, :SYMXZ, sur)
    symxy = aster_create_elements(mesh, :SYMXY, sur)
    update!(symyz, "displacement 1", 0.0)
    update!(symxz, "displacement 2", 0.0)
    update!(symxy, "displacement 3", 0.0)
    push!(bc, symyz..., symxz..., symxy...)
    return block, bc, elements, traction, symyz, symxz, symxy
end

function calc_size(elements, dim)
    A = 0.0
    for element in elements
        Ael = 0.0
        size(element, 1) == dim || continue
        for (w, xi) in get_integration_points(element)
            detJ = element(xi, 0.0, Val{:detJ})
            Ael += w*detJ
        end
        for (i, X) in enumerate(element["geometry"](0.0))
            info("$i : $X")
        end
        info("Area / volume: $Ael")
        A += Ael
    end
    return A
end

#=
@testset "test 3d block hex8" begin
    block, bc = get_model("BLOCK_HEX8", :HE8, :QU4)
    V = calc_size(block.elements, 3)
    info("volume of block: $V")
    A = calc_size(bc.elements, 2)
    info("area of boundary condition: $A")
    @test isapprox(V, 1.0)
    @test isapprox(A, 3.0)
    solver = Solver("solver block problem")
    solver.is_linear_system = true
    push!(solver, block, bc)
    call(solver)
    max_u = maximum(abs(block.assembly.u))
    info("max |u| = $max_u")
    @test isapprox(max_u, 2.0)
end
=#

@testset "test 3d block TET4" begin
    block, bc, elements, traction, symyz, symxz, symxy = get_model("BLOCK_TET4", :TE4, :TR3)
#   block, bc = get_model("BLOCK_TET10", :T10, :TR6)
    V = calc_size(block.elements, 3)
    info("volume of block: $V")
    A = calc_size(bc.elements, 2)
    info("area of boundary condition: $A")
    At = calc_size(traction, 2)
    info("area of load surface: $At")
    @test isapprox(V, 1.0)
    @test isapprox(At, 1.0)
    @test isapprox(A, 3.0)
    solver = Solver("solver block problem")
    #solver.is_linear_system = true
    push!(solver, block, bc)
    call(solver)
    max_u = maximum(block.assembly.u)
    nu = round(Int, length(block.assembly.u)/3)
    u = reshape(block.assembly.u, 3, nu)
    f = reshape(full(block.assembly.f), 3, nu)
    dump(round(u', 5))
    dump(round(f', 5))
    info("max |u| = $max_u")
    @test isapprox(max_u, 2.0)
end

