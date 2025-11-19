# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
Continuum mechanics assembly examples and documentation.

This file provides documentation for the explicit kernel API workflow.
Boundary condition functions have been moved to `src/domains/common/boundary_conditions.jl`
as they are generic and work with any domain type (continuum, beams, shells, heat, etc.).

# Explicit Workflow (Recommended)

```julia
# Setup
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

# Apply BCs explicitly (defined in domains/common/boundary_conditions.jl)
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

# See Also

- [`apply_neumann_bcs!`](@ref) - in `domains/common/boundary_conditions.jl`
- [`apply_dirichlet_bcs!`](@ref) - in `domains/common/boundary_conditions.jl`
- [`ContinuumKernel`](@ref) - in `domains/continuum/kernel.jl`
- [`CSCAssembler`](@ref), [`COOAssembler`](@ref) - in `assemblers/`
"""
