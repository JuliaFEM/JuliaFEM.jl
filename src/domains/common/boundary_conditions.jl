# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Generic boundary condition application for all domain types.

This file provides BC application functions that work with any kernel implementing
the generic kernel interface (continuum, beams, shells, plates, trusses, heat, etc.).

# Explicit Workflow (Recommended)

```julia
# Setup (example with continuum mechanics)
mesh = create_cantilever_mesh(50, 10, 10)
material = LinearElastic(E=210e9, ν=0.3)
kernel = ContinuumKernel(
    ContinuumFormulation{FullThreeD}(),
    material,
    Displacement{3}()
)

# Choose assembler explicitly
assembler = CSCAssembler()  # or COOAssembler()

# Create cache (reusable!)
cache = create_cache(assembler, mesh, kernel)

# Assembly and solve
assemble!(cache, assembler, kernel, mesh)
K, f = extract_system(cache)

# Apply BCs explicitly
apply_neumann_bcs!(f, kernel, mesh, bc_neumann)
apply_dirichlet_bcs!(K, f, kernel, mesh, bc_dirichlet)

# Solve
u = K \\ f
```

# Nonlinear Loop Example

```julia
cache = create_cache(CSCAssembler(), mesh, kernel)

for iter in 1:max_iter
    assemble!(cache, assembler, kernel, mesh)  # Zero allocations!
    K, f = extract_system(cache)
    apply_neumann_bcs!(f, kernel, mesh, bc_neumann)
    apply_dirichlet_bcs!(K, f, kernel, mesh, bc_dirichlet)

    Δu = K \\ f
    u .+= Δu

    if norm(Δu) < tol
        break
    end
end
```
"""

"""
    apply_neumann_bcs!(
        f::Vector{Float64},
        kernel::AbstractKernel,
        mesh::AbstractMesh,
        bc_neumann::NeumannBC
    ) -> Nothing

Apply Neumann (natural) boundary conditions to force vector **in-place**.

Works with any kernel type (continuum, beams, shells, heat, etc.) that implements
the `dofs_per_node(kernel)` interface.

For now, interprets `surface_ids` as node IDs (simplified).
TODO: Proper surface force integration over element faces.

# Arguments
- `f`: Global force vector (modified in-place)
- `kernel`: Domain kernel (provides dofs_per_node)
- `mesh`: Finite element mesh
- `bc_neumann`: Neumann BC data structure

# Usage

```julia
# Define Neumann BCs
bc_neumann = NeumannBC()
push!(bc_neumann.surface_ids, 100)  # Node ID (simplified)
push!(bc_neumann.values, Vec{3}((0.0, 0.0, -1000.0)))  # Force vector

# Apply to force vector
apply_neumann_bcs!(f, kernel, mesh, bc_neumann)
```

# See Also
- [`apply_dirichlet_bcs!`](@ref)
"""
function apply_neumann_bcs!(
    f::Vector{Float64},
    kernel::AbstractKernel,
    mesh::AbstractMesh,
    bc_neumann::NeumannBC
)
    nnodes = nnodes_total(mesh)
    ndofs_per_node = dofs_per_node(kernel)

    for (surf_id, force) in zip(bc_neumann.surface_ids, bc_neumann.values)
        # Simplified: treat surface_id as node_id
        # TODO: Implement proper surface integration
        node = surf_id
        if node <= nnodes
            for α in 1:ndofs_per_node
                f[ndofs_per_node*(node-1)+α] += force[α]
            end
        end
    end

    return nothing
end

"""
    apply_dirichlet_bcs!(
        K::SparseMatrixCSC{Float64,Int},
        f::Vector{Float64},
        kernel::AbstractKernel,
        mesh::AbstractMesh,
        bc_dirichlet::DirichletBC
    ) -> Nothing

Apply Dirichlet (essential) boundary conditions **in-place**.

Works with any kernel type (continuum, beams, shells, heat, etc.) that implements
the `dofs_per_node(kernel)` interface.

Uses elimination method:
1. Zero out row and column for constrained DOF
2. Set diagonal to 1.0
3. Set force vector entry to prescribed value

# Arguments
- `K`: Global stiffness matrix (modified in-place)
- `f`: Global force vector (modified in-place)
- `kernel`: Domain kernel (provides dofs_per_node)
- `mesh`: Finite element mesh
- `bc_dirichlet`: Dirichlet BC data structure

# Usage

```julia
# Define Dirichlet BCs (fix all DOFs at node 1)
bc_dirichlet = DirichletBC()
push!(bc_dirichlet.node_ids, 1)
push!(bc_dirichlet.components, [1, 2, 3])  # All components
push!(bc_dirichlet.values, 0.0)

# Apply to system
apply_dirichlet_bcs!(K, f, kernel, mesh, bc_dirichlet)
```

# See Also
- [`apply_neumann_bcs!`](@ref)
"""
function apply_dirichlet_bcs!(
    K::SparseMatrixCSC{Float64,Int},
    f::Vector{Float64},
    kernel::AbstractKernel,
    mesh::AbstractMesh,
    bc_dirichlet::DirichletBC
)
    nnodes = nnodes_total(mesh)
    ndofs_per_node = dofs_per_node(kernel)
    ndofs = ndofs_per_node * nnodes

    for i in 1:length(bc_dirichlet.node_ids)
        node = bc_dirichlet.node_ids[i]
        components = bc_dirichlet.components[i]
        value = bc_dirichlet.values[i]

        for comp in components
            dof = ndofs_per_node * (node - 1) + comp
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
