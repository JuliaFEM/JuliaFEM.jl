# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

function JuliaFEM.get_mesh(::Type{Val{Symbol("curved 2d block splitted to upper and lower")}})
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d_curved.med"
    mesh = aster_read_mesh(meshfile)
end

function JuliaFEM.get_model(::Type{Val{Symbol("mesh tie with curved 2d block")}};
            dy=0.0, adjust=false, tolerance=0.0, rotate_normals=false, swap=false,
            dual_basis=false, use_forwarddiff=true, finite_strain=false,
            geometric_stiffness=false)

    mesh = get_mesh("curved 2d block splitted to upper and lower")

    upper = Problem(Elasticity, "upper", 2)
    upper.properties.formulation = :plane_stress
    upper.properties.finite_strain = finite_strain
    upper.properties.geometric_stiffness = geometric_stiffness
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper.elements, "youngs modulus", 96.0)
    update!(upper.elements, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 2)
    lower.properties.formulation = :plane_stress
    lower.properties.finite_strain = finite_strain
    lower.properties.geometric_stiffness = geometric_stiffness
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower.elements, "youngs modulus", 96.0)
    update!(lower.elements, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper.elements, "displacement 1", 0.0)
    update!(bc_upper.elements, "displacement 2", dy)

    bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower.elements, "displacement 1", 0.0)
    update!(bc_lower.elements, "displacement 2", 0.0)

    interface = Problem(Mortar, "interface between upper and lower block", 2, "displacement")
    interface_slave_elements = create_elements(mesh, "LOWER_TOP")
    interface_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    if swap
        interface_slave_elements, interface_master_elements = interface_master_elements, interface_slave_elements
    end
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_master_elements; interface_slave_elements]
    interface.properties.adjust = adjust
    interface.properties.distval = tolerance
    interface.properties.rotate_normals = rotate_normals
    interface.properties.dual_basis = dual_basis
    interface.properties.use_forwarddiff = use_forwarddiff
    interface.assembly.u = zeros(2*length(mesh.nodes))
    interface.assembly.la = zeros(2*length(mesh.nodes))

    solver = Solver(Nonlinear)
    push!(solver, upper, lower, bc_upper, bc_lower, interface)

    return solver

end


@testset "curved surface with adjust=true, standard lagrange, slave=lower surface, dy=0.0" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=false, tolerance=10, dy=-0.1, rotate_normals=true,
        dual_basis=true, use_forwarddiff=true, finite_strain=true,
        geometric_stiffness=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.11339715157447851)
end

#=

@testset "curved surface with adjust=true, dual lagrange, slave=lower surface, dy=0.0" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=0.0, rotate_normals=true,
        dual_basis=true, use_forwarddiff=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    # differs -- why?
    @test isapprox(norm(interface.assembly.u), 0.11660422877751599)
end

@testset "curved surface with adjust=true, standard lagrange, slave=lower surface, dy=-0.1" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=-0.1, rotate_normals=true,
        dual_basis=false, use_forwarddiff=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.34230262165505887)
end

@testset "curved surface, adjust=true, dual basis, slave=lower surface, dy=-0.1" begin
    # TODO: analytical solution now known, verify using other fem software
    solver = get_model("mesh tie with curved 2d block";
        adjust=true, tolerance=10, dy=-0.1, rotate_normals=true,
        dual_basis=true, use_forwarddiff=true)
    call(solver)
    interface = solver["interface between upper and lower block"]
    @test solver.properties.iteration == 2
    @test isapprox(norm(interface.assembly.u), 0.34318800698017704)
end

=#

function Base.isapprox(A::SparseMatrixCOO, B::SparseMatrixCOO)
    A2 = sparse(A)
    B2 = sparse(B, size(A2)...)
    return isapprox(A2, B2)
end

function Base.isapprox(a1::Assembly, a2::Assembly)
    T = isapprox(a1.K, a2.K)
    T &= isapprox(a1.C1, a2.C1)
    T &= isapprox(a1.C2, a2.C2)
    T &= isapprox(a1.D, a2.D)
    T &= isapprox(a1.f, a2.f)
    T &= isapprox(a1.g, a2.g)
    return T
end

@testset "compare forwarddiff solution to normal" begin
    X = Dict(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [0.0, 1.0],
        4 => [1.0, 1.0])
    u = Dict(
        1 => [0.0, 0.0],
        2 => [0.0, 0.0],
        3 => [0.0, 0.0],
        4 => [0.0, 0.0])
    sel1 = Element(Seg2, [1, 2])
    mel1 = Element(Seg2, [3, 4])
    update!([sel1, mel1], "geometry", X)
    update!([sel1, mel1], "displacement", u)
    update!(sel1, "master elements", [mel1])
    p1 = Problem(Mortar, "test 1", 2, "displacement")
    p2 = Problem(Mortar, "test 2", 2, "displacement")
    push!(p1, sel1, mel1)
    push!(p2, sel1, mel1)
    #p1.properties.adjust = true
    p2.properties.use_forwarddiff = true
    #p1.properties.dual_basis = true
    #p2.properties.dual_basis = true
    p2.assembly.u = zeros(8)
    p2.assembly.la = zeros(8)
    assemble!(p1, 0.0)
    assemble!(p2, 0.0)
    @test isapprox(p1.assembly, p2.assembly)

    empty!(p1.assembly)
    empty!(p2.assembly)
    p1.properties.adjust = true
    p2.properties.adjust = true
    assemble!(p1, 0.0)
    assemble!(p2, 0.0)
    C11 = full(p1.assembly.C1, 4, 8)
    C12 = full(p2.assembly.C1, 4, 8)
    C21 = full(p1.assembly.C2, 4, 8)
    C22 = full(p2.assembly.C2, 4, 8)
    D1 = full(p1.assembly.D)
    D2 = full(p2.assembly.D)
    g1 = full(p1.assembly.g, 4, 1)
    g2 = full(p2.assembly.g, 4, 1)
    println("C1")
    dump(C11)
    dump(C12)
    println("C2")
    dump(C21)
    dump(C22)
    println("D")
    dump(D1)
    dump(D2)
    println("g")
    dump(g1)
    dump(g2)
    @test isapprox(p1.assembly, p2.assembly)
end

