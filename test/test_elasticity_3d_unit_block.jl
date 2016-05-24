# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

function get_model(fn, vol, sur)
    meshfile = Pkg.dir("JuliaFEM")*"/geometry/3d_blocks/BLOCK.med"
    mesh = parse_aster_med_file(meshfile, fn)
    info(mesh)

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

function xdmf_dump(all_elements, eltype, elsym, filename="/tmp/xdmf_result.xmf")
    info("$(length(all_elements)) elements.")
    xdoc, xmodel = xdmf_new_model()
    coll = xdmf_new_temporal_collection(xmodel)
    grid = xdmf_new_grid(coll; time=0.0)

    Xg = Dict{Int64, Vector{Float64}}()
    ug = Dict{Int64, Vector{Float64}}()
    nids = Dict{Int64, Int64}()
    for element in all_elements
        conn = get_connectivity(element)
        for (i, c) in enumerate(conn)
            nids[c] = c
        end
        X = element("geometry", 0.0)
        for (i, c) in enumerate(conn)
            Xg[c] = X[i]
        end
        haskey(element, "displacement") || continue
        u = element("displacement", 0.0)
        for (i, c) in enumerate(conn)
            ug[c] = u[i]
        end
    end
    perm = sort(collect(keys(Xg)))
    nodes = Vector{Float64}[Xg[i] for i in perm]
    disp = Vector{Float64}[ug[i] for i in perm]
    nids = Int[nids[i] for i in perm]
    inids = Dict{Int64, Int64}()
    for (i, nid) in enumerate(nids)
        inids[nid] = i
    end
    elements = []
    for element in all_elements
        isa(element, eltype) || continue
        conn = get_connectivity(element)
        nconn = [inids[i] for i in conn]
        push!(elements, (elsym, nconn))
    end

    xdmf_new_mesh!(grid, nodes, elements)
    xdmf_new_nodal_field!(grid, "displacement", disp)
    xdmf_save_model(xdoc, filename)
    info("model dumped to $filename")
end

function calc_model(model, volume_element, surface_element)
    block, bc, elements, traction, symyz, symxz, symxy = get_model(model, volume_element, surface_element)
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
    return block, u
end

@testset "test 3d block TET4" begin
    block, u = calc_model("BLOCK_TET4", :TE4, :TR3)
    @test isapprox(maximum(u), 2.1329516539440205)
end

@testset "test 3d block TET10" begin
    block, u = calc_model("BLOCK_TET10", :T10, :TR6)
    @test isapprox(maximum(u), 2.13656216413056)
end

@testset "test 3d block HEX8" begin
    block, u = calc_model("BLOCK_HEX8", :HE8, :QU4)
#   xdmf_dump(block.elements, Element{Hex8}, :Hex8, "/tmp/BLOCK_HEX8.xmf")
    @test isapprox(maximum(u), 2.0)
end
