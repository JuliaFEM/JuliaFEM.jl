# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

datadir = first(splitext(basename(@__FILE__)))

@testset "test postprocessing of secondary fields" begin
    mesh = abaqus_read_mesh(joinpath(datadir, "tet4.inp"))
    
    upper = Problem(Elasticity, "UPPER", 3)
    upper.elements = create_elements(mesh, "UPPER")
    update!(upper, "youngs modulus", 3*288.0)
    update!(upper, "poissons ratio", 1/3)
    push!(upper.postprocess_fields, "strain", "stress")
    info("upper postprocess: $(upper.postprocess_fields)")

    lower = Problem(Elasticity, "LOWER", 3)
    lower.elements = create_elements(mesh, "LOWER")
    update!(lower, "youngs modulus", 288.0)
    update!(lower, "poissons ratio", 1/3)
    push!(lower.postprocess_fields, "strain", "stress")
    info("lower postprocess: $(lower.postprocess_fields)")

    bc_upper = Problem(Dirichlet, "UPPER_TOP", 3, "displacement")
    bc_upper.elements = create_surface_elements(mesh, "UPPER_TOP")
    update!(bc_upper, "displacement 3", -0.4)

    bc_lower = Problem(Dirichlet, "LOWER_BOTTOM", 3, "displacement")
    bc_lower.elements = create_surface_elements(mesh, "LOWER_BOTTOM")
    update!(bc_lower, "displacement 3", 0.0)
    
    # point-wise boundary conditions to prevent free body move
    nid1 = find_nearest_nodes(mesh, [0.0, 0.0, 0.0])[1]
    nid2 = find_nearest_nodes(mesh, [1.0, 0.0, 0.0])[1]
    nid3 = find_nearest_nodes(mesh, [0.0, 1.0, 0.0])[1]
    nid4 = find_nearest_nodes(mesh, [0.0, 0.0, 1.0])[1]
    nid5 = find_nearest_nodes(mesh, [1.0, 0.0, 1.0])[1]
    nid6 = find_nearest_nodes(mesh, [0.0, 1.0, 1.0])[1]

    bc_sym13 = Problem(Dirichlet, "SYM13", 3, "displacement")
    # nodes in X2=0 plane
    bc_sym13.elements = [Element(Poi1, [j]) for j in [nid1, nid2, nid4, nid5]]
    update!(bc_sym13, "geometry", mesh.nodes)
    update!(bc_sym13, "displacement 2", 0.0)

    bc_sym23 = Problem(Dirichlet, "SYM23", 3, "displacement")
    # nodes in X1=0 plane
    bc_sym23.elements = [Element(Poi1, [j]) for j in [nid1, nid3, nid4, nid6]]
    update!(bc_sym23, "geometry", mesh.nodes)
    update!(bc_sym23, "displacement 1", 0.0)

    for bc in [bc_upper, bc_lower, bc_sym13, bc_sym23]
        push!(bc.postprocess_fields, "reaction force")
    end

    interface = Problem(Contact, "LOWER_TO_UPPER", 3, "displacement")
    interface_slave_elements = create_surface_elements(mesh, "LOWER_TO_UPPER")
    interface_master_elements = create_surface_elements(mesh, "UPPER_TO_LOWER")
    update!(interface_slave_elements, "master elements", interface_master_elements)
    interface.elements = [interface_slave_elements; interface_master_elements]
    interface.properties.dual_basis = true
    interface.properties.contact_state_in_first_iteration = :AUTO
    push!(interface.postprocess_fields, "contact pressure")

    solver = NonlinearSolver(upper, lower, bc_upper, bc_lower, bc_sym13, bc_sym23, interface)

    solver.xdmf = Xdmf("contact_two_blocks_postprocess"; overwrite=true)
    solver()

    node_ids, displacement = get_nodal_vector(interface.elements, "displacement", 0.0)
    node_ids, geometry = get_nodal_vector(interface.elements, "geometry", 0.0)
    # node_ids, pressure = get_nodal_vector(interface.elements, "contact pressure", 0.0)
    u3 = [u[3] for u in displacement]
    maxabsu3 = maximum(abs(u3))
    stdabsu3 = std(abs(u3))
    info("max(abs(u3)) = $maxabsu3, std(abs(u3)) = $stdabsu3")
    @test isapprox(stdabsu3, 0.0; atol=1.0e-12)
    upper_X = [0.5, 0.5, 0.75]
    lower_X = [0.5, 0.5, 0.25]
    strain_upper = upper("strain", upper_X)
    stress_upper = upper("stress", upper_X)
    strain_lower = lower("strain", lower_X)
    stress_lower = lower("stress", lower_X)
    info("strain at $upper_X = $strain_upper")
    info("stress at $upper_X = $stress_upper")
    info("strain at $lower_X = $strain_lower")
    info("stress at $lower_X = $stress_lower")
end

