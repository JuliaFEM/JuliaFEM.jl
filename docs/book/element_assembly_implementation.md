---
title: "Traditional Element Assembly Implementation"
date: 2025-11-11
author: "JuliaFEM Team"
status: "Complete"
last_updated: 2025-11-11
tags: ["assembly", "element", "reference", "baseline"]
---

## Overview

This module provides the **traditional element-by-element assembly** approach for finite element analysis. It serves as:

1. **Reference implementation** for comparison with nodal assembly
2. **Baseline** for performance benchmarking
3. **Production-ready** assembly for CPU-based solvers

**Status:** Complete with comprehensive tests (47/47 passing ✓)

## Key Features

### ✅ Standard FEM Assembly

- Element stiffness matrices → Global sparse matrix
- Element force vectors → Global force vectors
- Residual computation: `r = f_int - f_ext`
- Dirichlet boundary conditions (penalty method)

### ✅ Sparse Matrix Assembly

- COO format during assembly (triplet lists)
- Automatic CSC conversion
- Overlapping elements handled correctly (summed)
- Symmetry preserved

### ✅ Production Features

- Reset for iterative/incremental solvers
- Matrix-vector product interface (for GMRES)
- Boundary condition application
- Assembly statistics printing

## Data Structures

### ElementAssemblyData

Global assembly storage:

```julia
struct ElementAssemblyData{T}
    K_global::SparseMatrixCSC{T}   # Global tangent stiffness
    r_global::Vector{T}             # Residual: f_int - f_ext
    f_int_global::Vector{T}         # Internal forces
    f_ext_global::Vector{T}         # External forces
    ndof::Int                       # Total DOF
end
```

**Usage:**

```julia
ndof = 3 * nnodes  # 3D problem
assembly = ElementAssemblyData(ndof, Float64)
```

### ElementContribution

Local element quantities before scattering:

```julia
struct ElementContribution{T}
    element_id::Int
    gdofs::Vector{Int}              # Global DOF indices
    K_local::Matrix{T}              # Element stiffness (e.g., 12×12 for Tet4)
    f_int_local::Vector{T}          # Element internal forces
    f_ext_local::Vector{T}          # Element external forces
end
```

**Example:**

```julia
# Tet4 element connecting nodes [5, 7, 12, 15]
conn = (5, 7, 12, 15)
gdofs = get_dof_indices(conn, 3)  # [13,14,15, 19,20,21, ...]
contrib = ElementContribution(elem_id, gdofs, Float64)

# Fill K_local, f_int_local, f_ext_local during element integration
# ... (loop over Gauss points)

# Scatter to global
scatter_to_global!(assembly, contrib)
```

## Assembly Workflow

### Standard FEM Assembly Loop

```julia
# 1. Initialize
ndof = 3 * nnodes
assembly = ElementAssemblyData(ndof)

# 2. Loop over elements
for (elem_id, element) in enumerate(elements)
    # Get connectivity
    conn = element.connectivity  # e.g., (1, 2, 3, 4)
    gdofs = get_dof_indices(conn, 3)
    
    # Allocate element contribution
    contrib = ElementContribution(elem_id, gdofs)
    
    # Compute element quantities (loop over Gauss points)
    for gp in gauss_points
        # Compute B matrix, material tangent, stress, etc.
        # Accumulate into contrib.K_local, contrib.f_int_local
    end
    
    # Scatter to global
    scatter_to_global!(assembly, contrib)
end

# 3. Compute residual
compute_residual!(assembly)

# 4. Apply boundary conditions
fixed_dofs = [1, 2, 3]  # Fix node 1
apply_dirichlet_bc!(assembly, fixed_dofs)

# 5. Solve (example: direct solver)
Δu = assembly.K_global \ (-assembly.r_global)
```

### Batch Assembly (for multiple elements)

```julia
# Compute all element contributions
contributions = ElementContribution{Float64}[]
for element in elements
    contrib = compute_element_contribution(element, u, time)
    push!(contributions, contrib)
end

# Assemble all at once
assemble_elements!(assembly, contributions)
```

## Key Operations

### 1. Scatter to Global

Adds element quantities to global system:

```julia
scatter_to_global!(assembly, contrib)
```

**What it does:**

- Adds `K_local[i,j]` to `K_global[gdofs[i], gdofs[j]]`
- Adds `f_int_local[i]` to `f_int_global[gdofs[i]]`
- Adds `f_ext_local[i]` to `f_ext_global[gdofs[i]]`

**Key property:** Multiple elements can contribute to same global DOF (summed automatically)

### 2. Residual Computation

```julia
compute_residual!(assembly)
```

Computes: `r = f_int - f_ext`

For Newton-Raphson: solve `K * Δu = -r`

### 3. Boundary Conditions

```julia
# Fix nodes 1 and 2 to zero displacement
fixed_dofs = [1,2,3, 4,5,6]
apply_dirichlet_bc!(assembly, fixed_dofs)

# Fix with prescribed values
prescribed_values = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]  # Node 2: u_x = 0.1
apply_dirichlet_bc!(assembly, fixed_dofs, prescribed_values)
```

Uses **penalty method**:

- Adds large stiffness to diagonal: `K[i,i] += penalty`
- Modifies RHS: `r[i] = penalty * u_prescribed`

### 4. Matrix-Vector Product

```julia
w = matrix_vector_product(assembly, v)
```

For matrix-free solvers (GMRES, CG):

```julia
function matvec(v)
    return matrix_vector_product(assembly, v)
end

Δu = gmres(matvec, -assembly.r_global, tol=1e-6)
```

## Testing

Comprehensive test suite (`test/test_element_assembly_structures.jl`):

### Test Coverage

- ✅ Data structure construction
- ✅ DOF indexing (`get_dof_indices`)
- ✅ Single element scatter
- ✅ Overlapping elements (accumulation)
- ✅ Residual computation
- ✅ Full assembly workflow
- ✅ Matrix-vector product
- ✅ Dirichlet BC application
- ✅ Symmetry preservation
- ✅ Reset functionality
- ✅ Statistics printing

**Results:** 47/47 tests passing ✓

### Example Test

```julia
@testset "Overlapping Elements" begin
    ndof = 15  # 5 nodes
    assembly = ElementAssemblyData(ndof)
    
    # Element 1: nodes [1,2,3,4]
    # Element 2: nodes [2,3,4,5]  ← shares nodes 2,3,4
    
    # ... create contributions ...
    scatter_to_global!(assembly, contrib1)
    scatter_to_global!(assembly, contrib2)
    
    # Node 2 diagonal: accumulated from both elements
    @test assembly.K_global[4,4] == K1[4,4] + K2[1,1]
end
```

## Advantages

### ✅ Well-Established

- Standard textbook algorithm
- Easy to understand and verify
- Decades of production use

### ✅ Flexible

- Works with any element type
- Handles complex meshes
- Supports all boundary conditions

### ✅ Sparse Matrix Support

- Leverages Julia's SparseArrays
- Efficient storage (only non-zeros)
- Fast direct solvers (UMFPACK, etc.)

## Disadvantages

### ❌ GPU Parallelization

**Problem:** Multiple elements write to same global DOF

```julia
# Thread 1 (element 5):
K_global[10, 10] += K_elem5[4, 4]  # Race!

# Thread 2 (element 7):
K_global[10, 10] += K_elem7[2, 2]  # Race!
```

**Solution:** Requires atomic operations → slow on GPU (10-100× slower)

### ❌ Memory Usage

For large problems (N DOF):

- **K_global**: Sparse but still O(N²) worst case
- **Memory**: Can be 100s of MB to GBs
- **Cache:** Scattered access pattern

### ❌ Matrix Formation

For matrix-free methods:

- Still need to form K explicitly
- Cannot avoid assembly cost
- Wastes work if only need K*v

## Comparison: Element vs Nodal Assembly

| Aspect | Element Assembly | Nodal Assembly |
|--------|-----------------|----------------|
| **Outer loop** | Elements | Nodes |
| **Parallelization** | Hard (atomics) | Easy (no conflicts) |
| **Memory** | Full K matrix | 3×3 blocks per thread |
| **Matrix-free** | Must form K | Natural K*v |
| **GPU** | Slow (atomics) | Fast (no atomics) |
| **CPU** | Fast (mature) | Comparable |
| **Complexity** | Simple | Moderate |

## Performance Characteristics

### Memory

```
Problem: 100K nodes (300K DOF)

Element Assembly:
- K_global: ~300K × 300K sparse
- Typical sparsity: 0.01%
- Memory: ~300K × 300K × 0.0001 × 8 bytes
         = ~720 MB (just for K!)

Nodal Assembly:
- No global K
- Per-thread: ~30 nodes × 3×3 blocks
- Memory: 30 × 9 × 8 bytes = 2.16 KB per thread
```

### Assembly Time

Typical: **O(n_elem × n_gauss × ndof_elem²)**

For 10K Tet4 elements:

- n_elem = 10,000
- n_gauss = 4 (typical)
- ndof_elem = 12 (4 nodes × 3 DOF)

Assembly: ~10K × 4 × 144 = 5.76M operations

**CPU:** ~5-50 ms (depending on material model)

## Usage Examples

### Example 1: Static Linear Elasticity

```julia
# Setup
nnodes = 1000
ndof = 3 * nnodes
assembly = ElementAssemblyData(ndof)

# Assemble
for element in elements
    contrib = compute_linear_elastic_element(element, coordinates, material)
    scatter_to_global!(assembly, contrib)
end

compute_residual!(assembly)

# Boundary conditions: fix base nodes
fixed_nodes = find_base_nodes(mesh)
fixed_dofs = vcat([3*(n-1) .+ (1:3) for n in fixed_nodes]...)
apply_dirichlet_bc!(assembly, fixed_dofs)

# Solve
u = assembly.K_global \ (-assembly.r_global)
```

### Example 2: Newton-Raphson Nonlinear

```julia
u = zeros(ndof)
for iter in 1:max_iterations
    # Assemble at current configuration
    reset!(assembly)
    for element in elements
        contrib = compute_element(element, u, material)
        scatter_to_global!(assembly, contrib)
    end
    compute_residual!(assembly)
    apply_dirichlet_bc!(assembly, fixed_dofs)
    
    # Check convergence
    if norm(assembly.r_global) < tol
        break
    end
    
    # Solve for increment
    Δu = assembly.K_global \ (-assembly.r_global)
    u += Δu
end
```

### Example 3: Matrix-Free GMRES

```julia
# Assemble once
assemble_elements!(assembly, contributions)

# Define matrix-free operator
function matvec(v)
    return matrix_vector_product(assembly, v)
end

# Solve with GMRES (Krylov.jl)
using Krylov
Δu, stats = gmres(matvec, -assembly.r_global, 
                  atol=1e-8, rtol=1e-6, itmax=100)

println("GMRES converged in $(stats.niter) iterations")
```

## Implementation Notes

### Sparse Matrix Assembly

Uses **triplet (COO) format** during assembly:

```julia
I_rows = Int[]
J_cols = Int[]
values = Float64[]

# Accumulate triplets
for element in elements
    for i in 1:ndofs_local, j in 1:ndofs_local
        push!(I_rows, gdofs[i])
        push!(J_cols, gdofs[j])
        push!(values, K_local[i,j])
    end
end

# Convert to CSC (automatic summing of duplicates)
K_global = sparse(I_rows, J_cols, values, ndof, ndof)
```

**Why:** Efficient for scattered writes, automatic duplicate handling

### DOF Ordering

Standard ordering: group by node

```
Node 1: DOFs [1, 2, 3]     (u_x, u_y, u_z)
Node 2: DOFs [4, 5, 6]     (u_x, u_y, u_z)
Node 3: DOFs [7, 8, 9]     (u_x, u_y, u_z)
...
```

Helper function:

```julia
gdofs = get_dof_indices(connectivity, dim)
```

## Files

- **Implementation:** `src/element_assembly_structures.jl` (356 lines)
- **Tests:** `test/test_element_assembly_structures.jl` (259 lines)
- **Documentation:** This file

## References

1. **Hughes:** "The Finite Element Method" - Chapter 4 (Assembly)
2. **Zienkiewicz & Taylor:** "The Finite Element Method" - Volume 1
3. **Bathe:** "Finite Element Procedures" - Chapter 6

## Next Steps

- ✅ Element assembly complete
- ✅ Nodal assembly complete
- 🔄 Performance comparison benchmarks
- 🔄 GPU implementation (nodal assembly)
- 🔄 Integration with material models

This traditional assembly serves as the **baseline** for validating nodal assembly correctness and measuring performance improvements!
