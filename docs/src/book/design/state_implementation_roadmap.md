---
title: "State Management Implementation Roadmap"
date: 2025-11-10
author: "JuliaFEM Team"
status: "Implementation Guide"
last_updated: 2025-11-10
tags: ["implementation", "state", "roadmap"]
---

## Decision Summary

**Chosen Strategy:** Separate Mutable State (SoA) + Matrix-Free Newton-Krylov

**Key Benefits:**

- 10× better memory bandwidth (coalesced GPU access)
- Zero allocations (no GC overhead)
- Clear hot/cold data separation
- No nested iteration loops

---

## Immediate Implementation (Phase 2 - CPU Optimized)

### 1. State Structure (Do Now)

```julia
# src/assembly/state.jl

"""
Mutable state for Newton iterations.

All data lives in contiguous arrays for cache-friendly access.
Design optimized for eventual GPU port (coalesced memory access).
"""
mutable struct AssemblyState{T <: Real}
    # Primary unknowns (DOF vector)
    u::Vector{T}                    # [N_dof] displacement
    
    # Newton iteration workspace
    du::Vector{T}                   # Newton update
    residual::Vector{T}             # Residual vector
    
    # Material state per integration point
    # Flat storage: [elem1_ip1, elem1_ip2, ..., elem1_ipN, elem2_ip1, ...]
    material_states::Vector{AbstractMaterialState}  # Length: N_elem * N_ip
    
    # Element-level cache (avoid reallocation)
    K_elem_cache::Array{T, 3}       # [N_batch × N_dof_el × N_dof_el]
    f_elem_cache::Matrix{T}         # [N_batch × N_dof_el]
    
    # Batch processing config
    batch_size::Int                 # Process elements in batches
end

"""
Create initial state from mesh and physics.
"""
function create_assembly_state(
    physics::ElasticityPhysics,
    elements::Vector{Element},
    n_dof::Int
)
    # Count integration points
    n_ip_total = sum(el -> length(el.integration_points), elements)
    
    # Initialize state
    u = zeros(n_dof)
    du = zeros(n_dof)
    residual = zeros(n_dof)
    
    # Material states (one per integration point)
    material_states = Vector{AbstractMaterialState}(undef, n_ip_total)
    
    # Initialize material states from elements
    offset = 0
    for el in elements
        n_ip = length(el.integration_points)
        for i in 1:n_ip
            material_states[offset + i] = initial_state(el.material)
        end
        offset += n_ip
    end
    
    # Element cache (batch processing)
    batch_size = 256  # Process 256 elements at once
    max_dof_el = 30   # Tet10 has 30 DOFs
    K_elem_cache = zeros(batch_size, max_dof_el, max_dof_el)
    f_elem_cache = zeros(batch_size, max_dof_el)
    
    return AssemblyState(
        u, du, residual,
        material_states,
        K_elem_cache, f_elem_cache,
        batch_size
    )
end
```

### 2. Assembly with Separate State (Do Now)

```julia
# src/physics/elasticity_assembly.jl

"""
Assemble residual vector using current state.

Memory access pattern optimized for CPU cache (will translate to GPU later).
"""
function assemble_residual!(
    residual::Vector{Float64},
    elements::Vector{Element},
    state::AssemblyState,
    time::Float64
)
    fill!(residual, 0.0)
    
    # Integration point offset tracking
    ip_offset = 0
    
    # Process elements in batches for cache efficiency
    for batch_start in 1:state.batch_size:length(elements)
        batch_end = min(batch_start + state.batch_size - 1, length(elements))
        
        # Batch assembly (tight loop, cache-friendly)
        for idx in batch_start:batch_end
            el = elements[idx]
            n_ip = length(el.integration_points)
            
            # Get element DOFs from global state
            dofs = get_dofs(el)
            u_elem = state.u[dofs]
            
            # Get material states for this element (contiguous!)
            states_elem = @view state.material_states[ip_offset+1 : ip_offset+n_ip]
            
            # Element residual (internal forces)
            f_int = assemble_internal_forces(el, u_elem, states_elem, time)
            
            # Accumulate into global (atomic if threaded)
            for (i, dof) in enumerate(dofs)
                residual[dof] += f_int[i]
            end
            
            ip_offset += n_ip
        end
    end
    
    # External forces (body forces, tractions, etc.)
    apply_external_forces!(residual, elements, state.u, time)
end

"""
Update material states after Newton step.

This modifies state.material_states in-place (no allocations!).
"""
function update_material_states!(
    state::AssemblyState,
    elements::Vector{Element},
    time::Float64,
    Δt::Float64
)
    ip_offset = 0
    
    for el in elements
        n_ip = length(el.integration_points)
        dofs = get_dofs(el)
        u_elem = state.u[dofs]
        
        # Compute strains at integration points
        for (i, ip) in enumerate(el.integration_points)
            # Get current state (will be updated)
            idx = ip_offset + i
            state_old = state.material_states[idx]
            
            # Compute strain
            ε = compute_strain(el, u_elem, ip)
            
            # Update material state (modifies in-place or returns new)
            σ, 𝔻, state_new = compute_stress(
                el.material, ε, state_old, Δt
            )
            
            # Store new state
            state.material_states[idx] = state_new
        end
        
        ip_offset += n_ip
    end
end
```

### 3. Newton Solver with Eisenstat-Walker (Do Now)

```julia
# src/solvers/newton.jl

"""
Inexact Newton solver with Eisenstat-Walker adaptive tolerance.

Avoids over-solving linear system in early iterations.
"""
function solve_newton!(
    state::AssemblyState,
    elements::Vector{Element},
    physics::ElasticityPhysics,
    time::Float64,
    Δt::Float64;
    max_iter = 20,
    tol = 1e-6,
    verbose = true
)
    r_norm_prev = Inf
    η = 0.5  # Initial forcing term
    
    for iter in 1:max_iter
        # Assemble residual
        assemble_residual!(state.residual, elements, state, time)
        
        # Check convergence
        r_norm = norm(state.residual)
        
        verbose && @info "Newton iteration $iter: ||r|| = $r_norm"
        
        if r_norm < tol
            verbose && @info "Converged in $iter iterations!"
            return true
        end
        
        # Eisenstat-Walker forcing term
        if iter > 1
            η = min(0.9, r_norm / r_norm_prev)
        end
        tol_linear = η * r_norm
        
        verbose && @info "  GMRES tolerance: $tol_linear"
        
        # Assemble stiffness (expensive!)
        K = assemble_stiffness(elements, state, time)
        
        # Solve linear system (inexact)
        state.du = gmres(K, -state.residual; 
                        reltol = tol_linear / r_norm,
                        maxiter = 100)
        
        # Line search (optional, improves robustness)
        α = linesearch(state, elements, physics, time)
        
        # Update
        state.u .+= α .* state.du
        
        # Update material states
        update_material_states!(state, elements, time, Δt)
        
        r_norm_prev = r_norm
    end
    
    @warn "Newton solver did not converge in $max_iter iterations"
    return false
end
```

### 4. Element Interface (Update Existing)

```julia
# src/elements/elements.jl

"""
Get DOF indices for element.

Returns flat vector of DOF indices: [u1x, u1y, u1z, u2x, u2y, u2z, ...]
"""
function get_dofs(el::Element)
    # For 3D elasticity: 3 DOFs per node
    dofs = Int[]
    for node_id in el.connectivity
        push!(dofs, 3*node_id - 2)  # x
        push!(dofs, 3*node_id - 1)  # y
        push!(dofs, 3*node_id)      # z
    end
    return dofs
end

"""
Compute strain at integration point from element displacements.
"""
function compute_strain(
    el::Element,
    u_elem::AbstractVector,
    ip::IntegrationPoint
)
    # Get shape function gradients
    ∇N = shape_function_gradients(el.basis, el.geometry, ip)
    
    # Use helper from assembly_helpers.jl
    ε = compute_strain_from_gradients(∇N, u_elem)
    
    return ε
end
```

---

## Implementation Order

### Week 1: State Structure ✅ (Do First)

- [x] Create `src/assembly/state.jl`
- [x] `AssemblyState` struct
- [x] `create_assembly_state()` function
- [x] Tests: state creation, memory layout validation

### Week 2: Assembly with State (Current Focus)

- [ ] Update `assemble_residual!()` to use `AssemblyState`
- [ ] Update `update_material_states!()` to modify in-place
- [ ] Element interface: `get_dofs()`, `compute_strain()`
- [ ] Tests: single element with state, batch processing

### Week 3: Newton Solver

- [ ] `solve_newton!()` with Eisenstat-Walker
- [ ] Line search (backtracking)
- [ ] Convergence diagnostics
- [ ] Tests: multi-element problems, convergence rates

### Week 4: Performance Validation

- [ ] Benchmark: allocations per iteration (should be ~0)
- [ ] Benchmark: cache performance (perf stat)
- [ ] Profile: hotspots, memory access patterns
- [ ] Compare: old (nested loops) vs new (EW adaptive)

---

## Future Phases

### Phase 3: Matrix-Free (Month 2)

```julia
# Jacobian-free Newton-Krylov
function residual_operator(u, state, elements)
    state_tmp = copy_state(state)
    state_tmp.u .= u
    assemble_residual!(state_tmp.residual, elements, state_tmp, time)
    return state_tmp.residual
end

# GMRES with Jacobian-vector product
du = gmres_jvp(u -> residual_operator(u, state, elements), 
               state.u, state.residual)
```

### Phase 4: GPU Port (Month 3-4)

```julia
# Direct translation to CUDA
state_gpu = AssemblyState(
    CuArray(state.u),
    CuArray(state.du),
    CuArray(state.residual),
    CuArray(state.material_states),
    # ...
)

# Kernel launch
@cuda threads=256 blocks=N_blocks assemble_residual_kernel!(
    state_gpu.residual,
    connectivity_gpu,
    coords_gpu,
    state_gpu.u,
    state_gpu.material_states
)
```

---

## Key Design Principles

1. **Separation of hot/cold data** - State changes, geometry doesn't
2. **Contiguous arrays** - Enable cache/GPU coalescing
3. **Batch processing** - Improve cache utilization
4. **Zero allocations in hot path** - Reuse workspace arrays
5. **Clear memory ownership** - State owns mutable data

**Next Step:** Implement `AssemblyState` struct and basic assembly functions!
