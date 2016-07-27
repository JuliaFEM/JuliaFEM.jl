# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing
using JuliaFEM.Preprocess

@testset "test projection" begin
    C = [
        2.0  1.0  0.0  0.0   0.0   0.0  0.0  0.0
        1.0  2.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  2.0  1.0  -1.0  -2.0  0.0  0.0
        0.0  0.0  1.0  2.0  -2.0  -1.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0
        0.0  0.0  0.0  0.0   0.0   0.0  0.0  0.0]
    g = [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    P, h = create_projection(sparse(C), g)
    P_expected = [
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0]
    h_expected = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test isapprox(full(P), P_expected)
    @test isapprox(full(h), h_expected)
    nz = [5, 6, 7, 8]
    info("is P'P positive definite? ", isposdef(P[:,nz]'P[:,nz]))
    @test isposdef(P[:,nz]'P[:,nz])
end

@testset "test creating projection matrix from invertible problem" begin
    # simple 3 element poisson problem
    k = [1.0 -1.0; -1.0 1.0]
    K = zeros(4, 4)
    K[1:2,1:2] += k
    K[2:3,2:3] += k
    K[3:4,3:4] += k
    # first dof homogeneous bc, last dof u₄ = 1
    C = zeros(4, 4)
    C[1,1] = 1.0
    C[4,4] = 1.0
    g = zeros(4)
    g[1] = 0.0
    g[4] = 1.0
    P, h = create_projection(sparse(C), g, Val{:invertible})
    info("P = ")
    dump(full(P))
    P_expected = zeros(4, 4)
    P_expected[2,2] = P_expected[3,3] = 1.0
    @test isapprox(full(P), P_expected)
    #h_expected = [0.5, 1.0]
    #@test isapprox(h, h_expected)
end

@testset "projection between surfaces" begin
    # FIXME: creating mortar projection takes very long time.
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/joint.med"
    isfile(meshfile) || return
    mesh = aster_read_mesh(meshfile, "JOINT")

    # top
    bc1 = Problem(Dirichlet, "fixed1", 3, "displacement")
    bc1.elements = create_elements(mesh, "FIXED1")
    update!(bc1.elements, "displacement 1", 0.0)
    update!(bc1.elements, "displacement 2", 0.0)
    update!(bc1.elements, "displacement 3", 0.0)

    # bottom
    bc2 = Problem(Dirichlet, "fixed2", 3, "displacement")
    bc2.elements = create_elements(mesh, "FIXED2")
    update!(bc2.elements, "displacement 1", 0.0)
    update!(bc2.elements, "displacement 2", 0.0)
    update!(bc2.elements, "displacement 3", 0.0)

    # joint
    contact = Problem(Mortar, "joint", 3, "displacement")
    master_elements = create_elements(mesh, "BODY1_TO_BODY2")
    slave_elements = create_elements(mesh, "BODY2_TO_BODY1")
    update!(slave_elements, "master elements", master_elements)
    contact.elements = [master_elements; slave_elements]

    solver = Solver(Modal)
    push!(solver, bc1, bc2, contact)
    assemble!(solver)
    Kb, C1, C2, D, fb, g = get_boundary_assembly(solver)
    P, h = create_projection(C1, g)
end

