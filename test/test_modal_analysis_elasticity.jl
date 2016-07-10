# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

@testset "calculate cross-sectional properties" begin
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
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

#=
test subjects:
- calculate cross-sectional properties
- modal analysis with known solution

Fixed-fixed solution is ωᵢ = λᵢ²√(EI/ρA) , where λᵢ = cosh(λᵢℓ)cos(λᵢℓ)

1:  4.730040744862704
2:  7.853204624095838
3: 10.995607838001671

[1] De Silva, Clarence W. Vibration: fundamentals and practice. CRC press, 2006, p.355
=#

@testset "long rod under point load" begin
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "CYLINDER_20_TET10")
#   for (id, coords) in mesh.nodes
#       mesh.nodes[id][1] *= 5.0
#   end
    body = Problem(Elasticity, "rod", 3)
    body.elements = create_elements(mesh, "CYLINDER")
    E = 50475.44814745859
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
    @test isapprox(Xc, [0.0, 0.0, 0.0]; atol=1.0e-5)
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
    solver()
    freqs = keys(body["displacement"])

    rtol1 = norm(freq_sa - freqs[2])/max(freq_sa, freqs[2])
    rtol2 = norm(freq_a - freqs[2])/max(freq_a, freqs[2])
    info("rtol 1 = $rtol1, rtol 2 = $rtol2")
    @test rtol2 < 1.0e-2
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

