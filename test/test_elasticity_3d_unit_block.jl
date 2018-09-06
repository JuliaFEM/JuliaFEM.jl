# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

function calc_model(mesh_name; with_volume_load=false)
    meshfile = @__DIR__()*"/testdata/3d_block.med"
    mesh = aster_read_mesh(meshfile, mesh_name)

    block = Problem(Elasticity, "BLOCK", 3)
    block.properties.finite_strain = false
    block.properties.geometric_stiffness = false

    block_elements = create_elements(mesh, "BLOCK")
    update!(block_elements, "youngs modulus", 288.0)
    update!(block_elements, "poissons ratio", 1/3)
    with_volume_load && update!(block_elements, "displacement load 3", 576.0)
    add_elements!(block, block_elements)

    traction_elements = create_elements(mesh, "LOAD")
    update!(traction_elements, "displacement traction force 3", 288.0)
    add_elements!(block, traction_elements)

    bc = Problem(Dirichlet, "symmetry boundary condition", 3, "displacement")
    symyz = create_elements(mesh, "SYMYZ")
    symxz = create_elements(mesh, "SYMXZ")
    symxy = create_elements(mesh, "SYMXY")
    update!(symyz, "displacement 1", 0.0)
    update!(symxz, "displacement 2", 0.0)
    update!(symxy, "displacement 3", 0.0)
    add_elements!(bc, symyz, symxz, symxy)

    analysis = Analysis(Linear)
    add_problems!(analysis, block, bc)
    run!(analysis)

    u = block("displacement", 0.0)
    max_u = maximum(maximum(ui for ui in values(u)))
    return max_u
end


# 3d block HEX8
max_u = calc_model("BLOCK_HEX8"; with_volume_load=true)
@test isapprox(max_u, 2.0)

# 3d block TET4
max_u = calc_model("BLOCK_TET4"; with_volume_load=false)
@test isapprox(max_u, 1.0)

# 3d block TET10
max_u = calc_model("BLOCK_TET10"; with_volume_load=false)
@test isapprox(max_u, 1.0)
