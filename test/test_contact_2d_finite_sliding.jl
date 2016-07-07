using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

@testset "2d curved block with frictionless finite sliding contact using forwarddiff" begin
    # FIXME: needs verification of some other fem software
    meshfile = Pkg.dir("JuliaFEM") * "/test/testdata/block_2d_curved.med"
    mesh = aster_read_mesh(meshfile)

    upper = Problem(Elasticity, "upper", 2)
    upper.properties.formulation = :plane_stress
    upper.properties.finite_strain = true
    upper.properties.geometric_stiffness = true
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 96.0)
    update!(upper, "poissons ratio", 1/3)

    lower = Problem(Elasticity, "lower", 2)
    lower.properties.formulation = :plane_stress
    lower.properties.finite_strain = true
    lower.properties.geometric_stiffness = true
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 96.0)
    update!(lower, "poissons ratio", 1/3)

    bc_upper = Problem(Dirichlet, "upper boundary", 2, "displacement")
    bc_upper.elements = create_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 1", 0.0)
    update!(bc_upper, "displacement 2", -0.15)

    bc_lower = Problem(Dirichlet, "lower boundary", 2, "displacement")
    bc_lower.elements = create_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 1", 0.0)
    update!(bc_lower, "displacement 2", 0.0)

    contact = Problem(Contact, "contact between upper and lower block", 2, "displacement")
    contact.properties.rotate_normals = true
    contact.properties.finite_sliding = true
    contact.properties.friction = false
    contact.properties.use_forwarddiff = true
    contact_slave_elements = create_elements(mesh, "LOWER_TOP")
    contact_master_elements = create_elements(mesh, "UPPER_BOTTOM")
    update!(contact_slave_elements, "master elements", contact_master_elements)
    contact.elements = [contact_master_elements; contact_slave_elements]

    solver = NonlinearSolver(upper, lower, bc_upper, bc_lower, contact)
    solver()
    normu = norm(contact.assembly.u)
    info("displacement vector norm = $normu")

    # while accurate solution is unknown this is very close to linear solution
    # sqrt( ((Stress 11 - Stress 22)^2 + (Stress 22 - Stress 33)^2 + (Stress 33-Stress 11)^2 + 6*(Stress 12^2 + Stress 23^2 + Stress 13^2))/2 )
#   @test isapprox(normu, 0.49745873784105105)
    @test isapprox(normu, 0.49745872893844145)
end

#= TODO: Fix test, this is not converging
@testset "project from master to slave" begin
    el = Element(Seg2, [1, 2])
    x1 = DVTI(Vector{Float64}[
             [ 0.07406987526791842, 0.6628967239474994],
             [-0.24633092752656838, 0.4732606367688589]])
    n1 = DVTI(Vector{Float64}[
             [0.40398625635635355, 0.9147650543583191],
             [-0.5093430176405901, 0.860563588807229]])
    x2 = [0.5049198709043257, 0.27765317280577695]
    xi = project_from_master_to_slave(el, x1, n1, x2; debug=true)
    info("xi = $xi")
end
=#
