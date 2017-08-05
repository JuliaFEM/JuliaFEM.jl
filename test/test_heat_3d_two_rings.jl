# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing
using AsterReader: RMEDFile, aster_read_nodes, aster_read_data

#=
Two rings, RING1 = inner, RING2 = outer, RINGS combined mesh. Set T=1.0 for
inner ring and T=2.0 for outer ring, measure temperature from middle of ring.
Results are calculated using Code Aster for comparison.
=#
@testset "test 3d heat, two rings, and compare to CA solution" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "RINGS_UNION")

    rings = Problem(Heat, "RINGS", 1)
#   rings.elements = create_elements(mesh; element_type=:Tet4)
    rings.elements = create_elements(mesh, "RING1", "RING2")
    update!(rings.elements, "temperature thermal conductivity", 1.0)
    bc_inner = Problem(Dirichlet, "INNER SURFACE", 1, "temperature")
    bc_inner.elements = create_elements(mesh, "RING1_INNER")
    bc_outer = Problem(Dirichlet, "OUTER SURFACE", 1, "temperature")
    bc_outer.elements = create_elements(mesh, "RING2_OUTER")
    update!(bc_inner, "temperature 1", 1.0)
    update!(bc_outer, "temperature 1", 2.0)
    info("# of elements in RING1_INNER = ", length(bc_inner.elements))
    info("# of elements in RING2_OUTER = ", length(bc_outer.elements))
    solver = LinearSolver(rings, bc_inner, bc_outer)
    solver()

    temp_jf = rings("temperature", 0.0)

    fn = @__DIR__() * "/testdata/rings.rmed"
    results = RMEDFile(fn)
    nodes = aster_read_nodes(results)
    temp_ca = aster_read_data(results, "TEMP")

    passed = true
    for j in sort(collect(keys(temp_jf)))
        X = nodes[j]
        T1 = temp_jf[j]
        T2 = temp_ca[j]
        rtol = norm(T1-T2) / max(T1,T2)
        @printf "% 5i : %8.5f %8.5f %8.5f | %8.5f %8.5f | %8.5e\n" j X... T1 T2 rtol
        passed &= rtol < 1.0e-12
    end
    @test passed
end

