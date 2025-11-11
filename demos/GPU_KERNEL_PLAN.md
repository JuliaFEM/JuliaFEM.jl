# GPU Kernel Plan: Newton-Krylov-Anderson

## What We Learned from CPU Reference

The complete solver pipeline (see `newton_krylov_anderson_cpu.jl`) has these operations:

### Outer Loop: Newton Iterations
```julia
for iter in 1:max_newton_iter
    r = assemble_residual(u)              # ← GPU KERNEL 1
    du = gmres_solve(J, -r)              # ← GMRES loop (see below)
    u_new = u + α * du                    # ← GPU vector op
    u = anderson_step(u_new, r)          # ← CPU (small vectors)
end
```

### Middle Loop: GMRES Iterations
```julia
for j in 1:max_gmres_iter
    w = J * v_j                           # ← GPU KERNEL 1 (matvec via FD)
    # Arnoldi orthogonalization          # ← cuBLAS (dot products)
    # Givens rotations                   # ← CPU (small matrices)
end
```

### Inner Operation: Matrix-Free Matvec
```julia
function apply_jacobian(v)
    r_pert = assemble_residual(u + ε*v)  # ← GPU KERNEL 1
    return (r_pert - r) / ε
end
```

## GPU Kernel Requirements

### **KERNEL 1: Element Residual Assembly** (The Workhorse)

**What it does:**
```julia
for elem in elements                      # ← parallelized over GPU threads
    for gp in gauss_points
        # 1. Compute Jacobian
        J = Σ dN_i ⊗ X_i
        
        # 2. Physical derivatives
        dN_dx = J^-1 · dN_dξ
        
        # 3. Strain
        ε = sym(Σ dN_i ⊗ u_i)
        
        # 4. Stress with plasticity
        (σ, state_new) = return_mapping(ε, state_old)
        
        # 5. Nodal forces
        f_i = dN_i · σ
        
        # 6. Scatter to global (atomics!)
        atomic_add!(r_global[dofs], f_i)
    end
end
```

**Inputs:** (all on GPU)
- `nodes::CuArray{Float64, 2}` - Shape (3, n_nodes)
- `elements::CuArray{Int32, 2}` - Shape (4, n_elements) for Tet4
- `u::CuArray{Float64}` - Solution vector
- `states::CuArray{PlasticState}` - Plastic states per GP

**Outputs:** (all on GPU)
- `r::CuArray{Float64}` - Residual vector
- `states::CuArray{PlasticState}` - Updated states (in-place)

**Key challenge:** Manual indexing inside kernel (no Tensors.jl)

### **KERNEL 2: Vector Operations** (Trivial)

These are standard cuBLAS operations:
- `y = α*x + β*y` (axpy)
- `dot(x, y)`
- `norm(x)`

Already provided by CUDA.jl

## The GPU Implementation Strategy

### Phase 1: Single Kernel Test
```julia
# Test just the residual assembly kernel
r_cpu = assemble_residual_cpu(u)
r_gpu = assemble_residual_gpu(u)  # ← Implement this!
@test r_cpu ≈ r_gpu
```

### Phase 2: Matrix-Free Matvec Test
```julia
# Test Jacobian-vector product
Jv_cpu = apply_jacobian_cpu(u, v)
Jv_gpu = apply_jacobian_gpu(u, v)  # ← Just calls kernel twice
@test Jv_cpu ≈ Jv_gpu
```

### Phase 3: GMRES on GPU
```julia
# Krylov.jl supports CuArrays!
using Krylov, CUDA
du_gpu = CuVector(zeros(n))
gmres!(du_gpu, J_gpu, -r_gpu)  # ← Should work with CuArrays
```

### Phase 4: Complete Newton-Krylov-Anderson on GPU
```julia
u_gpu = CuVector(u)
states_gpu = CuArray(states)

for iter in 1:max_iter
    r_gpu = assemble_residual_gpu!(u_gpu, states_gpu)     # ← GPU
    gmres!(du_gpu, J_gpu, -r_gpu)                         # ← GPU
    u_gpu .+= du_gpu                                      # ← GPU
    u_cpu = Vector(u_gpu)                                 # ← Copy to CPU
    u_cpu = anderson_step(u_cpu, Vector(r_gpu))          # ← CPU
    u_gpu = CuVector(u_cpu)                               # ← Copy to GPU
end
```

**Note:** Anderson runs on CPU (small vectors, needs least-squares). This is fine!

## Key Implementation Details

### 1. Plastic State on GPU

```julia
# CPU version (Tensors.jl)
struct PlasticState
    ε_p::SymmetricTensor{2,3,Float64,6}
    α::Float64
end

# GPU version (manual indexing)
struct PlasticStateGPU
    ε_p::NTuple{6,Float64}  # Voigt notation: (ε11, ε22, ε33, ε12, ε13, ε23)
    α::Float64
end
```

### 2. Return Mapping on GPU

Must be **completely manual** - no Tensors.jl:
```julia
function return_mapping_gpu(ε_total, state_old, E, ν, σ_y)
    # All operations in Voigt notation
    # ε = (ε11, ε22, ε33, ε12, ε13, ε23)
    
    # Trial stress (manual Hooke's law)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    
    ε_e = ε_total .- state_old.ε_p
    tr_ε = ε_e[1] + ε_e[2] + ε_e[3]
    
    σ_trial = (
        λ * tr_ε + 2μ * ε_e[1],  # σ11
        λ * tr_ε + 2μ * ε_e[2],  # σ22
        λ * tr_ε + 2μ * ε_e[3],  # σ33
        2μ * ε_e[4],              # σ12
        2μ * ε_e[5],              # σ13
        2μ * ε_e[6]               # σ23
    )
    
    # Deviatoric stress (manual)
    p = (σ_trial[1] + σ_trial[2] + σ_trial[3]) / 3
    s = (σ_trial[1] - p, σ_trial[2] - p, σ_trial[3] - p,
         σ_trial[4], σ_trial[5], σ_trial[6])
    
    # von Mises stress
    σ_eq = sqrt(3/2 * (s[1]^2 + s[2]^2 + s[3]^2 + 2*(s[4]^2 + s[5]^2 + s[6]^2)))
    
    # Yield check
    f = σ_eq - σ_y
    
    if f <= 0
        return (σ_trial, state_old, false)
    else
        # Radial return
        Δγ = f / (3μ)
        β = 1 - 2μ * Δγ / σ_eq
        
        σ_new = (
            β * s[1] + p,
            β * s[2] + p,
            β * s[3] + p,
            β * s[4],
            β * s[5],
            β * s[6]
        )
        
        # Update plastic strain
        n = s ./ σ_eq
        Δε_p = Δγ .* n
        ε_p_new = state_old.ε_p .+ Δε_p
        α_new = state_old.α + Δγ
        
        state_new = PlasticStateGPU(ε_p_new, α_new)
        
        return (σ_new, state_new, true)
    end
end
```

### 3. Element Kernel Structure

```julia
function element_residual_kernel!(r_global, nodes, elements, u, states, E, ν, σ_y)
    elem_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if elem_idx <= size(elements, 2)
        # Extract element nodes
        n1, n2, n3, n4 = elements[:, elem_idx]
        
        X1 = (nodes[1, n1], nodes[2, n1], nodes[3, n1])
        X2 = (nodes[1, n2], nodes[2, n2], nodes[3, n2])
        X3 = (nodes[1, n3], nodes[2, n3], nodes[3, n3])
        X4 = (nodes[1, n4], nodes[2, n4], nodes[3, n4])
        
        # Extract displacements
        u1 = (u[3*n1-2], u[3*n1-1], u[3*n1])
        u2 = (u[3*n2-2], u[3*n2-1], u[3*n2])
        u3 = (u[3*n3-2], u[3*n3-1], u[3*n3])
        u4 = (u[3*n4-2], u[3*n4-1], u[3*n4])
        
        # Shape derivatives (constant for Tet4)
        dN1 = (-1.0, -1.0, -1.0)
        dN2 = (1.0, 0.0, 0.0)
        dN3 = (0.0, 1.0, 0.0)
        dN4 = (0.0, 0.0, 1.0)
        
        # Jacobian (manual tensor product and sum)
        J11 = dN1[1]*X1[1] + dN2[1]*X2[1] + dN3[1]*X3[1] + dN4[1]*X4[1]
        J12 = dN1[1]*X1[2] + dN2[1]*X2[2] + dN3[1]*X3[2] + dN4[1]*X4[2]
        # ... (all 9 components)
        
        # Inverse Jacobian (manual 3x3 inverse)
        detJ = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + ...
        invJ11 = (J22*J33 - J23*J32) / detJ
        # ... (all 9 components)
        
        # Physical derivatives (manual matrix-vector multiply)
        dN1_dx = (invJ11*dN1[1] + invJ12*dN1[2] + invJ13*dN1[3],
                  invJ21*dN1[1] + invJ22*dN1[2] + invJ23*dN1[3],
                  invJ31*dN1[1] + invJ32*dN1[2] + invJ33*dN1[3])
        # ... (for all 4 nodes)
        
        # Strain (manual tensor product, sum, symmetrize)
        gradu11 = dN1_dx[1]*u1[1] + dN2_dx[1]*u2[1] + dN3_dx[1]*u3[1] + dN4_dx[1]*u4[1]
        # ... (all 9 components)
        
        ε = (gradu11,
             gradu22,
             gradu33,
             (gradu12 + gradu21) / 2,
             (gradu13 + gradu31) / 2,
             (gradu23 + gradu32) / 2)
        
        # Plasticity
        state_old = states[elem_idx, 1]  # 1 GP for Tet4
        (σ, state_new, plastic) = return_mapping_gpu(ε, state_old, E, ν, σ_y)
        states[elem_idx, 1] = state_new
        
        # Forces (manual contraction)
        f1 = (dN1_dx[1]*σ[1] + dN1_dx[2]*σ[4] + dN1_dx[3]*σ[5],
              dN1_dx[1]*σ[4] + dN1_dx[2]*σ[2] + dN1_dx[3]*σ[6],
              dN1_dx[1]*σ[5] + dN1_dx[2]*σ[6] + dN1_dx[3]*σ[3])
        # ... (for all 4 nodes)
        
        # Gauss weight * detJ
        wdetJ = (1.0/6.0) * detJ
        
        # Scatter to global (ATOMIC!)
        CUDA.@atomic r_global[3*n1-2] += f1[1] * wdetJ
        CUDA.@atomic r_global[3*n1-1] += f1[2] * wdetJ
        CUDA.@atomic r_global[3*n1]   += f1[3] * wdetJ
        # ... (for all 4 nodes)
    end
    
    return nothing
end
```

## Next Steps

1. **Implement `element_residual_kernel!`** - The core GPU kernel (all manual indexing)
2. **Test kernel correctness** - Compare GPU vs CPU residual
3. **Wrap in matrix-free operator** - For GMRES
4. **Test GMRES with CuArrays** - Krylov.jl should handle it
5. **Complete Newton loop** - Anderson on CPU is fine

The key insight: **Everything except Anderson can stay on GPU!**
