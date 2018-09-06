# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test, LinearAlgebra

#= this has nothing to do here
@testset "calculate cross-sectional properties" begin
    mesh_file = @__DIR__() * "/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "CYLINDER_20_TET4")
    # calculate cross-sectional properties A and Iₓ
    fixed1 = Problem(Dirichlet, "left support", 3, "displacement")
    fixed1.elements = create_elements(mesh, "FACE1")
    A = calculate_area(fixed1)
    @info("cross-section area: $A")
    # real area is π
    @test isapprox(A, pi; rtol=0.1)
    Xc = calculate_center_of_mass(fixed1)
    @info("center of mass: $Xc")
    @test isapprox(Xc, [0.0, 0.0, 0.0]; atol=1.0e-12)
    I = calculate_second_moment_of_mass(fixed1)
    @info("moments:")
    @info(I)
    I_expected = zeros(3, 3)
    I_expected[2,2] = I_expected[3,3] = pi/4
    rtol = norm(I[2,2]-I_expected[2,2]) / max(I[2,2],I_expected[2,2])
    @info("I rtol = $rtol")
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

# long rod natural frequencies
mesh_file = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(mesh_file, "CYLINDER_20_TET10")
body = Problem(Elasticity, "rod", 3)
body_elements = create_elements(mesh, "CYLINDER")
#E = 50475.44814745859
E = 50475.5
rho = 1.0
update!(body_elements, "youngs modulus", E)
update!(body_elements, "poissons ratio", 0.3)
update!(body_elements, "density", rho)
add_elements!(body, body_elements)

fixed1 = Problem(Dirichlet, "left support", 3, "displacement")
fixed1_elements = create_elements(mesh, "FACE1")
update!(fixed1_elements, "displacement 1", 0.0)
update!(fixed1_elements, "displacement 2", 0.0)
update!(fixed1_elements, "displacement 3", 0.0)
add_elements!(fixed1, fixed1_elements)

fixed2 = Problem(Dirichlet, "right support", 3, "displacement")
fixed2_elements = create_elements(mesh, "FACE2")
update!(fixed2_elements, "displacement 1", 0.0)
update!(fixed2_elements, "displacement 2", 0.0)
update!(fixed2_elements, "displacement 3", 0.0)
add_elements!(fixed2, fixed2_elements)

analysis = Analysis(Modal)
add_problems!(analysis, body, fixed1, fixed2)
analysis.properties.nev = 5
run!(analysis)
freqs_jf = sqrt.(analysis.properties.eigvals)/(2.0*pi)

# with Tet4 elements
#freqs_ca = [1.19789E+00, 1.20179E+00, 3.07391E+00, 3.08813E+00, 4.87370E+00]
# with Tet10 elements
freqs_ca = [9.65942E-01, 9.66160E-01, 2.52127E+00, 2.52187E+00, 3.48584E+00]
rtol = [norm(f1-f2) / max(f1,f2) for (f1, f2) in zip(freqs_jf, freqs_ca)]
@test maximum(rtol) < 3.0e-2
