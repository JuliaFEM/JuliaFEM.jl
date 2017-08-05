# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

using AsterReader: RMEDFile, aster_read_nodes, aster_read_data

#=
Two rings, RING1 is inner, RING2 is outer. Inner diameter is from 0.8 .. 0.9 and
outer ring is 0.9 .. 1.0. Contact surface pair is RING1_OUTER <- RING2_INNER.
Put constant temperature 1.0 for inner surface of inner ring and 2.0 for outer
surface of outer ring. We should expect constant temperature in contact surface.
This is conforming mesh so result should match to the conforming situation.
=#
@testset "test that curved interface transfers constant field without error, two rings problem" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "RINGS")

    ring1 = Problem(Heat, "RING1", 1)
    ring1.elements = create_elements(mesh, "RING1")
    update!(ring1.elements, "temperature thermal conductivity", 1.0)

    ring2 = Problem(Heat, "RING2", 1)
    ring2.elements = create_elements(mesh, "RING2")
    update!(ring2.elements, "temperature thermal conductivity", 1.0)

    bc_inner = Problem(Dirichlet, "INNER SURFACE", 1, "temperature")
    bc_inner.elements = create_elements(mesh, "RING1_INNER")
    update!(bc_inner, "temperature 1", 1.0)

    bc_outer = Problem(Dirichlet, "OUTER SURFACE", 1, "temperature")
    bc_outer.elements = create_elements(mesh, "RING2_OUTER")
    update!(bc_outer, "temperature 1", 2.0)

    interface = Problem(Mortar, "interface between rings", 1, "temperature")
    interface_slave = create_elements(mesh, "RING1_OUTER")
    interface_master = create_elements(mesh, "RING2_INNER")
    interface.elements = [interface_slave; interface_master]
    update!(interface_slave, "master elements", interface_master)

    solver = LinearSolver(ring1, ring2, bc_inner, bc_outer, interface)
    solver()

    fn = @__DIR__() * "/testdata/rings.rmed"
    results = RMEDFile(fn)
    nodes = aster_read_nodes(results)
    temp_ca = aster_read_data(results, "TEMP")

    passed = true
    for j in sort(collect(keys(nodes)))
        X = nodes[j]
        T1 = solver("temperature", X, 0.0)
        T2 = temp_ca[j]
        rtol = norm(T1-T2) / max(T1,T2)
        @printf "% 5i : %8.5f %8.5f %8.5f | %8.5f %8.5f | %8.5f\n" j X... T1 T2 rtol
        passed = passed && (rtol < 1.0e-12)
    end
    @test passed
end

