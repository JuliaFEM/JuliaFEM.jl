# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

#= this has nothing to do here
@testset "calculate cross-sectional properties" begin
    mesh_file = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "CYLINDER_20_TET4")
    # calculate cross-sectional properties A and Iₓ
    fixed1 = Problem(Dirichlet, "left support", 3, "displacement")
    fixed1.elements = create_elements(mesh, "FACE1")
    A = calculate_area(fixed1)
    info("cross-section area: $A")
    # real area is π
    @test isapprox(A, pi; rtol=0.1)
    Xc = calculate_center_of_mass(fixed1)
    info("center of mass: $Xc")
    @test isapprox(Xc, [0.0, 0.0, 0.0]; atol=1.0e-12)
    I = calculate_second_moment_of_mass(fixed1)
    info("moments:")
    info(I)
    I_expected = zeros(3, 3)
    I_expected[2,2] = I_expected[3,3] = pi/4
    rtol = norm(I[2,2]-I_expected[2,2]) / max(I[2,2],I_expected[2,2])
    info("I rtol = $rtol")
    @test isapprox(I, I_expected; rtol = 0.2)
end
=#

#=
test subjects:
- calculate cross-sectional properties
- modal analysis with known solution

Fixed-fixed solution is ωᵢ = λᵢ²√(EI/ρA) , where λᵢ = cosh(λᵢℓ)cos(λᵢℓ)

1:  4.730040744862704
2:  7.853204624095838
3: 10.995607838001671

Youngs modulus is tuned such that lowest eigenfrequency matches 1.0

5 lowest eigenfrequencies using Code Aster and Tet4 elements:
numéro    fréquence (HZ)     norme d'erreur
    1       1.19789E+00        2.20137E-12
    2       1.20179E+00        1.99034E-12
    3       3.07391E+00        3.29226E-13
    4       3.08812E+00        2.91550E-13
    5       4.87370E+00        2.95986E-13

5 lowest eigenfrequencies using Code Aster and Tet10 elements:
numéro    fréquence (HZ)     norme d'erreur
    1       9.65942E-01        1.54950E-11
    2       9.66160E-01        1.62712E-11
    3       2.52127E+00        2.06544E-12
    4       2.52187E+00        1.77970E-12
    5       3.48584E+00        9.96170E-13


[1] De Silva, Clarence W. Vibration: fundamentals and practice. CRC press, 2006, p.355
=#

@testset "long rod natural frequencies" begin
    mesh_file = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "CYLINDER_20_TET10")
#   for (id, coords) in mesh.nodes
#       mesh.nodes[id][1] *= 5.0
#   end
    body = Problem(Elasticity, "rod", 3)
    body.elements = create_elements(mesh, "CYLINDER")
    #E = 50475.44814745859
    E = 50475.5
    rho = 1.0
    update!(body.elements, "youngs modulus", E)
    update!(body.elements, "poissons ratio", 0.3)
    update!(body.elements, "density", rho)
    # calculate cross-sectional properties A and Iₓ
    fixed1 = Problem(Dirichlet, "left support", 3, "displacement")
    fixed1.elements = create_elements(mesh, "FACE1")
    update!(fixed1.elements, "displacement 1", 0.0)
    update!(fixed1.elements, "displacement 2", 0.0)
    update!(fixed1.elements, "displacement 3", 0.0)
    fixed2 = Problem(Dirichlet, "right support", 3, "displacement")
    fixed2.elements = create_elements(mesh, "FACE2")
    update!(fixed2.elements, "displacement 1", 0.0)
    update!(fixed2.elements, "displacement 2", 0.0)
    update!(fixed2.elements, "displacement 3", 0.0)
    A = calculate_area(fixed1)
    info("cross-section area: $A")
    # using SALOME / SMESH, A = 2.82843
    # real area is π
    @test isapprox(A, pi; rtol=0.1)
    Xc = calculate_center_of_mass(fixed1)
    info("center of mass: $Xc")
    #@test isapprox(Xc, [0.0, 0.0, 0.0]; atol=1.0e-5)
    I = calculate_second_moment_of_mass(fixed1)
    info("moments:")
    info(I)
    I_expected = zeros(3, 3)
    r = 1.0
    I_expected[2,2] = I_expected[3,3] = pi/4*r^2
    rtol = norm(I[2,2]-I_expected[2,2]) / max(I[2,2],I_expected[2,2])
    info("I rtol = $rtol")
    @test isapprox(I, I_expected; rtol = 0.2)
#=
    # apply transform Tx + b, in this case move cross-section to
    # xy-plane from yz-plane, i.e.
    # x₁ = y₂
    # y₁ = z₂
    T = [
        0.0 1.0 0.0
        0.0 0.0 1.0]
    b = [0.0, 0.0]
    X 1 = first(cross_section)("geometry", [1/3, 1/3], 0.0)
    apply_affine_transform!(cross_section, T, b)
    X2 = first(cross_section)("geometry", [1/3, 1/3], 0.0)
    info("X1 = $X1, X2 = $X2")
    @test isapprox(T*X1+b, X2)
=#
    c = sqrt(E*I[2,2]/(rho*A))
    info("c = $c")


    # analytical solution is
    l = 20.0
    r = 1.0
    la = 4.730040744862704/l

    # semi-analytical (c numerical)
    freq_sa = (c*la^2)/(2*pi)
    info("freq_sa = $freq_sa")

    A = pi*r^2
    I = pi/4*r^4
    c = sqrt(E*I/(rho*A))
    info("c analytical = $c")
    freq_a = (c*la^2)/(2*pi)

    info("freq_a = $freq_a")

    solver = Solver(Modal, body, fixed1, fixed2)
    solver.properties.nev = 5
    solver.xdmf = Xdmf()
    solver()
    freqs_jf = sqrt(solver.properties.eigvals)/(2.0*pi)
    # with Tet4 elements
    #freqs_ca = [1.19789E+00, 1.20179E+00, 3.07391E+00, 3.08813E+00, 4.87370E+00]
    # with Tet10 elements
    freqs_ca = [9.65942E-01, 9.66160E-01, 2.52127E+00, 2.52187E+00, 3.48584E+00]

    # looks that juliafem results are more close to 1.0, maybe different integration order
    rtol1 = norm(freq_sa - freqs_jf[1])/max(freq_sa, freqs_jf[1])
    rtol2 = norm(freq_a - freqs_jf[1])/max(freq_a, freqs_jf[1])
    info("rtol 1 = $rtol1, rtol 2 = $rtol2")
    passed = true
    for (f1, f2) in zip(freqs_jf, freqs_ca)
        rtol = norm(f1-f2) / max(f1,f2)
        @printf "JF: %8.5e | CA: %8.5e | rtol: %8.5e\n" f1 f2 rtol
        passed &= (rtol < 3.0e-2)
    end
    @test rtol2 < 3.5e-2
    @test passed

#=
    result = XDMF()
    for (i, freq) in enumerate(freqs)
        isapprox(freq, 0.0) && continue
        info("$i freq: $freq")
        xdmf_new_result!(result, body, freq)
        xdmf_save_field!(result, body, freq, "displacement"; field_type="Vector")
    end
    xdmf_save!(result, "/tmp/rod_nf.xmf")
=#

end

@testset "eigenvalues of cube (tet4)" begin
    meshfile = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(meshfile, "CUBE_TET4")
    cube = Problem(mesh, Elasticity, "CUBE", 3)
    update!(cube.elements, "youngs modulus", 10000.0)
    update!(cube.elements, "poissons ratio", 0.3)
    update!(cube.elements, "density", 10.0)
    sym23 = create_elements(mesh, "FACE231")
    update!(sym23, "displacement 1", 0.0)
    sym13 = create_elements(mesh, "FACE131")
    update!(sym13, "displacement 2", 0.0)
    sym12 = create_elements(mesh, "FACE121")
    update!(sym12, "displacement 3", 0.0)
    bcs = Problem(Dirichlet, "bcs", 3, "displacement")
    bcs.elements = [sym23; sym13; sym12]
    solver = Solver(Modal)
    solver.properties.nev = 5
    push!(solver, cube, bcs)
    solver()
    freqs_jf = sqrt(solver.properties.eigvals)/(2.0*pi)
    freqs_ca = [3.73724E+00, 3.73724E+00, 4.93519E+00, 6.59406E+00, 7.65105E+00]
    for (f1, f2) in zip(freqs_jf, freqs_ca)
        rtol = norm(f1-f2) / max(f1,f2)
        @printf "JF: %8.5e | CA: %8.5e | rtol: %8.5e\n" f1 f2 rtol
        @test rtol < 1.0e-5
    end
end

