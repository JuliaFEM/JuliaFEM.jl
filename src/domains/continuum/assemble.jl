# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Assembly for 3D Continuum Mechanics using Generic Assemblers

This file provides the high-level `assemble!(physics)` method that:
1. Creates a continuum kernel from physics parameters
2. Selects an assembler strategy (COO by default)
3. Delegates to generic assembler framework
4. Applies boundary conditions

Replaces monolithic assembly with clean separation:
- **Kernel** (WHAT to assemble): `src/domains/continuum/kernel.jl`
- **Assembler** (HOW to assemble): `src/assemblers/`
"""

"""
    assemble!(physics::Physics{ContinuumFormulation{FullThreeD}, Displacement{3}, M, Mat})
        -> (K, f)

Assemble global system for 3D continuum mechanics.

# Algorithm

1. Create continuum kernel from physics parameters
2. Select assembler (COOAssembler by default, can configure)
3. Create cache (all allocations here)
4. Assemble using generic assembler framework (zero allocations)
5. Extract system (K, f)
6. Apply boundary conditions

# Arguments
- `physics`: Physics object with mesh, material, formulation, field, BCs

# Returns
- `(K, f)::Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}`

# Performance

COOAssembler (default):
- Time: ~9.7ms for 2500 Tet4 elements
- Memory: ~8MB
- Best for: Prototyping, debugging

To use faster CSCAssembler (4.1x speedup):
```julia
# TODO: Add assembler selection to Physics constructor
# physics = Physics(..., assembler=CSCAssembler())
```

# References
- Kernel: `src/domains/continuum/kernel.jl`
- Assemblers: `src/assemblers/`
- Original implementation: `src/domains/continuum/assemble_v1_backup.jl`
"""
function assemble!(
    physics::Physics{ContinuumFormulation{FullThreeD},
        Displacement{3},
        M,
        Mat}) where {M<:AbstractMesh,Mat<:AbstractMaterial}

    mesh = physics.mesh
    material = physics.material
    formulation = physics.formulation
    field = physics.field
    bc_dirichlet = physics.bc_dirichlet
    bc_neumann = physics.bc_neumann

    # Create continuum kernel
    kernel = ContinuumKernel(formulation, material, field)

    # Select assembler (COO by default)
    # TODO: Allow user to configure assembler choice
    assembler = COOAssembler()

    # Create cache (ALL allocations here!)
    cache = create_cache(assembler, mesh, kernel)

    # Assemble (ZERO allocations!)
    assemble!(cache, assembler, kernel, mesh)

    # Extract system
    K, f = extract_system(cache)

    # Apply Neumann BCs (add forces to f)
    apply_neumann_bcs!(f, bc_neumann, mesh, kernel)

    # Apply Dirichlet BCs (modify K and f)
    apply_dirichlet_bcs!(K, f, bc_dirichlet, mesh, kernel)

    return (K, f)
end

"""
    apply_neumann_bcs!(f, bc_neumann::NeumannBC, mesh, kernel) -> Nothing

Apply Neumann (natural) boundary conditions to force vector **in-place**.

For now, interprets `surface_ids` as node IDs (simplified).
TODO: Proper surface force integration over element faces.

# Arguments
- `f`: Global force vector (modified in-place)
- `bc_neumann`: Neumann BC data structure
- `mesh`: Finite element mesh
- `kernel`: Domain kernel (for DOF mapping)
"""
function apply_neumann_bcs!(
    f::Vector{Float64},
    bc_neumann::NeumannBC,
    mesh::AbstractMesh,
    kernel::AbstractKernel
)
    nnodes = nnodes_total(mesh)

    for (surf_id, force) in zip(bc_neumann.surface_ids, bc_neumann.values)
        # Simplified: treat surface_id as node_id
        # TODO: Implement proper surface integration
        node = surf_id
        if node <= nnodes
            for α in 1:3
                f[3*(node-1)+α] += force[α]
            end
        end
    end

    return nothing
end

"""
    apply_dirichlet_bcs!(K, f, bc_dirichlet::DirichletBC, mesh, kernel) -> Nothing

Apply Dirichlet (essential) boundary conditions **in-place**.

Uses elimination method:
1. Zero out row and column for constrained DOF
2. Set diagonal to 1.0
3. Set force vector entry to prescribed value

# Arguments
- `K`: Global stiffness matrix (modified in-place)
- `f`: Global force vector (modified in-place)
- `bc_dirichlet`: Dirichlet BC data structure
- `mesh`: Finite element mesh
- `kernel`: Domain kernel (for DOF mapping)
"""
function apply_dirichlet_bcs!(
    K::SparseMatrixCSC{Float64,Int},
    f::Vector{Float64},
    bc_dirichlet::DirichletBC,
    mesh::AbstractMesh,
    kernel::AbstractKernel
)
    nnodes = nnodes_total(mesh)
    ndofs = dofs_per_node(kernel) * nnodes

    for i in 1:length(bc_dirichlet.node_ids)
        node = bc_dirichlet.node_ids[i]
        components = bc_dirichlet.components[i]
        value = bc_dirichlet.values[i]

        for comp in components
            dof = 3 * (node - 1) + comp
            if dof <= ndofs  # Safety check
                # Elimination method
                K[dof, :] .= 0.0
                K[:, dof] .= 0.0
                K[dof, dof] = 1.0
                f[dof] = value
            end
        end
    end

    return nothing
end
