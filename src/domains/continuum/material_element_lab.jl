# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

"""
Single-element **material coupon** on a structured `Hex8` brick.

Typical use: one trilinear brick on ``[0,L]^3`` from [`create_structured_box_mesh`](@ref)
with `nx = ny = nz = 1`, symmetry-type kinematics on three faces meeting at the origin,
uniform axial displacement on the opposite `x = L` face, and **traction-free** lateral
faces at `y = L` and `z = L`. Under uniform deformation this reproduces (approximately,
for linear kinematics) **uniaxial tension/compression** with lateral contraction free,
so stress–strain curves can be read from one quadrature patch.

Default quadrature for linear `Hex8` is ``2 \\times 2 \\times 2`` Gauss (eight points), not
a single reduced-integration point. Single-IP reduced quadrature would require a dedicated
quadrature hook on the assembler cache.

# API

- [`hex8_symmetric_uniaxial_eliminated_dirichlet`](@ref)
- [`material_lab_single_hex8_brick`](@ref)
- [`material_lab_linear_elastic_uniaxial_solve`](@ref)
"""

"""
    material_lab_single_hex8_brick(; L = 1.0) -> mesh

Structured `1 \\times 1 \\times 1` `Hex8` mesh on ``[0,L]^3`` with standard face node sets
(`:xmin`, `:xmax`, …).
"""
function material_lab_single_hex8_brick(; L::Float64 = 1.0)
    L > 0 || throw(ArgumentError("brick edge length L must be positive"))
    return create_structured_box_mesh(
        Hex8;
        xmin = 0.0,
        xmax = L,
        ymin = 0.0,
        ymax = L,
        zmin = 0.0,
        zmax = L,
        nx = 1,
        ny = 1,
        nz = 1,
    )
end

function _merge_dirichlet_entries!(dict::Dict{Int,Float64}, dof::Int, val::Float64)
    if haskey(dict, dof)
        old = dict[dof]
        isapprox(old, val; rtol = 0.0, atol = 1e-12) ||
            throw(ArgumentError("conflicting Dirichlet value on dof $dof: $old vs $val"))
    else
        dict[dof] = val
    end
    return nothing
end

"""
    hex8_symmetric_uniaxial_eliminated_dirichlet(mesh, handler, δx) -> EliminatedDirichlet

Symmetric “coupon” kinematics for [`Displacement{3}`](@ref) vertex unknowns:

| Face | Condition |
|------|-----------|
| `:xmin` | ``u_x = 0`` |
| `:ymin` | ``u_y = 0`` |
| `:zmin` | ``u_z = 0`` |
| `:xmax` | ``u_x = \\delta_x`` |

Faces `:ymax` and `:zmax` are natural (traction-free). Requires node sets from
[`create_structured_box_mesh`](@ref).
"""
function hex8_symmetric_uniaxial_eliminated_dirichlet(
    mesh,
    handler::DOFHandler,
    δx::Float64,
)
    dmap = Dict{Int,Float64}()
    for nid in get_nodes_in_set(mesh, :xmin)
        d = get_node_dofs(handler, Int(nid))
        length(d) ≥ 1 || error("expected vertex displacement DOFs at node $nid")
        _merge_dirichlet_entries!(dmap, d[1], 0.0)
    end
    for nid in get_nodes_in_set(mesh, :ymin)
        d = get_node_dofs(handler, Int(nid))
        length(d) ≥ 2 || error("expected ≥2 displacement DOFs at node $nid")
        _merge_dirichlet_entries!(dmap, d[2], 0.0)
    end
    for nid in get_nodes_in_set(mesh, :zmin)
        d = get_node_dofs(handler, Int(nid))
        length(d) ≥ 3 || error("expected ≥3 displacement DOFs at node $nid")
        _merge_dirichlet_entries!(dmap, d[3], 0.0)
    end
    for nid in get_nodes_in_set(mesh, :xmax)
        d = get_node_dofs(handler, Int(nid))
        _merge_dirichlet_entries!(dmap, d[1], δx)
    end
    fixed_dofs = sort!(collect(keys(dmap)))
    vals = Float64[dmap[i] for i in fixed_dofs]
    return EliminatedDirichlet(fixed_dofs, vals)
end

"""
    material_lab_linear_elastic_uniaxial_solve(mesh, handler, elements, E, ν, δx; formulation = ContinuumFormulation{ThreeDimensional}())

Assemble `K`, apply [`hex8_symmetric_uniaxial_eliminated_dirichlet`](@ref), solve `K u = 0`
with elimination lift, and return `u`.

`elements` is the vector from [`create_elements!`](@ref); `handler` must match that call.
"""
function material_lab_linear_elastic_uniaxial_solve(
    mesh,
    handler::DOFHandler,
    elements,
    E::Float64,
    ν::Float64,
    δx::Float64;
    formulation = ContinuumFormulation{ThreeDimensional}(),
)
    material = LinearElastic(E = E, ν = ν)
    kernel = ContinuumKernel(formulation, material, Displacement{3}())
    asm = DOFBasedCOOAssembler()
    cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f = extract_system(cache)
    fill!(f, 0.0)
    bc = hex8_symmetric_uniaxial_eliminated_dirichlet(mesh, handler, δx)
    Kc = copy(K)
    apply_constraint!(Kc, f, bc)
    u = Kc \ f
    return u
end
