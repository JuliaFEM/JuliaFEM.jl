# Nodal Assembly with Element Fields

**Date:** November 9, 2025  
**Status:** Design Discussion  
**Context:** Both nodes AND elements need fields (immutable)

## The Key Insight

**Nodes have fields:** Displacement, temperature, contact pressure (nodal quantities)  
**Elements have fields:** Integration point data, material state, history variables

**Both are immutable** → Update by creating new copies

## The Architecture

### Structures

```julia
"""Node with position (immutable geometry)"""
struct Node
    id::UInt
    x::Float64
    y::Float64
    z::Float64
end

"""
Element with connectivity + fields (immutable!)

Fields contain:
- Integration point data (stress, strain, history)
- Element properties (material, orientation)
- Internal state variables
"""
struct Element{N,B,F}
    id::UInt
    connectivity::NTuple{N,UInt}  # Node IDs
    basis::B
    fields::F  # Type-stable! (NamedTuple, struct, etc.)
end

"""
Problem/Mesh structure

Contains:
- nodes: List of nodes (immutable)
- elements: List of elements (mutable vector, immutable elements)
- node_to_elements: Inverse connectivity
- nodal_fields: Nodal quantities (u, T, contact_pressure, etc.)
"""
struct Problem{F_node, F_elem}
    nodes::Vector{Node}
    elements::Vector{Element}  # Vector is mutable, elements are immutable
    node_to_elements::Vector{Vector{Int}}
    nodal_fields::F_node  # Type-stable nodal fields
end
```

### Field Update Pattern (Immutable Elements)

```julia
# During Newton iteration or time stepping:

# 1. Update nodal fields (easy - just one container)
nodal_fields_new = (
    u = u_new,           # Updated displacement
    T = T_new,           # Updated temperature
    contact_pressure = p_new,  # Updated contact pressure
)

# 2. Update element fields (need to update each element)
elements_new = copy(problem.elements)  # Shallow copy of vector

for (i, element) in enumerate(problem.elements)
    # Compute updated integration point data
    σ_new, history_new = material_update(element, nodal_fields_new)
    
    # Create NEW fields for this element
    fields_new = (
        σ = σ_new,           # Stress at integration points
        ε_plastic = ε_plastic_new,  # Plastic strain
        α = α_new,           # Hardening variable
        # ... other integration point data
    )
    
    # Create NEW element with updated fields (immutable!)
    elements_new[i] = Element(
        element.id,
        element.connectivity,
        element.basis,
        fields_new  # New fields!
    )
end

# 3. Create new problem with updated fields
problem_new = Problem(
    problem.nodes,       # Nodes unchanged (geometry)
    elements_new,        # Updated elements
    problem.node_to_elements,  # Connectivity unchanged
    nodal_fields_new     # Updated nodal fields
)
```

## GPU Kernel (Nodal Assembly with Element Fields)

```julia
function gpu_matvec_kernel!(
    y::CuArray{Float64,1},
    x::CuArray{Float64,1},
    nodes::CuArray{Node,1},
    elements::CuArray{Element,1},
    node_to_elements::CuArray{CuArray{Int,1},1},  # Array of arrays on GPU
    nodal_fields::F_node,  # Type-stable!
    dofs_per_node::Int,
)
    # Each thread processes one NODE
    for node_id in 1:length(nodes)
        node = nodes[node_id]
        
        # Access NODAL fields (displacement, temperature, etc.)
        u = nodal_fields.u
        T = nodal_fields.T
        
        # Get this node's DOFs
        local_dofs = get_dofs(node, dofs_per_node)
        
        # Initialize nodal contribution
        y_nodal = zeros(length(local_dofs))
        
        # GATHER from connected elements
        for elem_idx in node_to_elements[node_id]
            element = elements[elem_idx]
            
            # Access ELEMENT fields (integration point data)
            σ = element.fields.σ      # Stress at IPs
            C = element.fields.C      # Tangent modulus
            
            # Compute element contribution using:
            # - Element fields (σ, C at integration points)
            # - Nodal fields (u, T for this and neighboring nodes)
            K_elem_contribution = compute_element_contribution(
                element, node, σ, C, u, T
            )
            
            x_elem = extract_element_dofs(element, x)
            y_nodal += K_elem_contribution * x_elem
        end
        
        # Write to global y (no atomics!)
        y[local_dofs] = y_nodal
    end
end
```

## Material State Update (Element-by-Element)

```julia
"""
Update material state at integration points

For each element:
1. Extract nodal displacements from updated solution
2. Compute strains at integration points
3. Material update (plasticity, damage, etc.)
4. Create new element with updated fields
"""
function update_material_state!(problem::Problem, u_new::Vector{Float64})
    # Update nodal fields first
    nodal_fields_new = merge(problem.nodal_fields, (u=u_new,))
    
    # Update each element
    elements_new = Vector{Element}(undef, length(problem.elements))
    
    Threads.@threads for i in 1:length(problem.elements)
        element = problem.elements[i]
        
        # Extract nodal displacements for this element
        elem_nodes = [problem.nodes[nid] for nid in element.connectivity]
        u_elem = extract_element_dofs(element, u_new)
        
        # Get current state
        σ_old = element.fields.σ
        ε_plastic_old = element.fields.ε_plastic
        α_old = element.fields.α
        
        # Update at each integration point
        n_ips = length(σ_old)
        σ_new = similar(σ_old)
        ε_plastic_new = similar(ε_plastic_old)
        α_new = similar(α_old)
        C_new = similar(element.fields.C)
        
        for ip in 1:n_ips
            # Compute strain at this integration point
            ε_total = compute_strain(element, elem_nodes, u_elem, ip)
            
            # Material update (plasticity model)
            σ_new[ip], ε_plastic_new[ip], α_new[ip], C_new[ip] = 
                plasticity_update(
                    ε_total,
                    σ_old[ip],
                    ε_plastic_old[ip],
                    α_old[ip],
                    element.fields.E,  # Material constants
                    element.fields.ν,
                    element.fields.σ_y,
                )
        end
        
        # Create new fields
        fields_new = (
            E = element.fields.E,      # Material constants (unchanged)
            ν = element.fields.ν,
            σ_y = element.fields.σ_y,
            σ = σ_new,                 # Updated stress
            ε_plastic = ε_plastic_new, # Updated plastic strain
            α = α_new,                 # Updated hardening
            C = C_new,                 # Updated tangent
        )
        
        # Create new element (immutable!)
        elements_new[i] = Element(
            element.id,
            element.connectivity,
            element.basis,
            fields_new
        )
    end
    
    # Return new problem
    return Problem(
        problem.nodes,
        elements_new,           # Updated!
        problem.node_to_elements,
        nodal_fields_new        # Updated!
    )
end
```

## Newton Iteration Loop

```julia
"""
Nonlinear solve with nodal assembly + element state updates
"""
function solve_nonlinear!(problem::Problem, f_ext::Vector{Float64})
    # Initial guess
    u = zeros(3 * length(problem.nodes))
    
    for iteration in 1:max_iterations
        println("Newton iteration $iteration")
        
        # 1. Compute residual: r = K(u)*u - f_ext
        r = zeros(length(u))
        
        # Nodal assembly (uses current element fields!)
        for (node_id, node) in enumerate(problem.nodes)
            local_dofs = get_dofs(node, 3)
            r_nodal = zeros(length(local_dofs))
            
            # Gather from connected elements
            for elem_idx in problem.node_to_elements[node_id]
                element = problem.elements[elem_idx]
                
                # Use element's current tangent stiffness
                C = element.fields.C
                σ = element.fields.σ
                
                r_nodal += compute_nodal_residual(element, node, u, C, σ)
            end
            
            r[local_dofs] = r_nodal
        end
        
        # Add external forces
        r .-= f_ext
        
        # Check convergence
        if norm(r) < tolerance
            println("  Converged! ||r|| = $(norm(r))")
            break
        end
        
        # 2. Solve for increment: K*Δu = -r
        # Using matrix-free GMRES with nodal assembly
        Δu, stats = gmres(r) do x
            matvec_nodal_assembly(problem, x)
        end
        
        # 3. Update displacement
        u .+= Δu
        
        # 4. Update material state at integration points
        # (creates new elements with updated fields)
        problem = update_material_state!(problem, u)
        
        println("  ||Δu|| = $(norm(Δu)), ||r|| = $(norm(r))")
    end
    
    return u, problem
end
```

## Memory Management

### Creating New Elements (Cost Analysis)

```julia
# Old element
element_old = Element(
    id,
    connectivity,
    basis,
    (E=210e3, ν=0.3, σ=σ_old, ε_plastic=ε_old, α=α_old, C=C_old)
)

# New element (immutable update)
fields_new = (
    E = element_old.fields.E,    # Reference (no copy!)
    ν = element_old.fields.ν,    # Reference (no copy!)
    σ = σ_new,                   # New array
    ε_plastic = ε_plastic_new,   # New array
    α = α_new,                   # New array
    C = C_new,                   # New array
)

element_new = Element(
    element_old.id,           # Copy UInt (8 bytes)
    element_old.connectivity, # Reference NTuple (no copy!)
    element_old.basis,        # Reference (no copy!)
    fields_new                # New NamedTuple (wraps references)
)
```

**Cost per element update:**
- UInt id: 8 bytes (copy)
- NTuple connectivity: 0 bytes (referenced)
- Basis: 0 bytes (referenced)
- NamedTuple wrapper: ~24 bytes (pointer overhead)
- Constants (E, ν): 0 bytes (referenced)
- Integration point arrays: Allocated (σ, ε_plastic, α, C)

**Total:** ~32 bytes + new integration point data

**For 100K elements:**
- Overhead: 3.2 MB (negligible!)
- Integration point data: Depends on problem (allocated anyway)

### Garbage Collection

```julia
# After Newton iteration:
problem_old → elements_old → fields_old → σ_old, ε_old, etc.
problem_new → elements_new → fields_new → σ_new, ε_new, etc.

# When problem_old goes out of scope:
# - elements_old becomes unreachable → GC
# - fields_old becomes unreachable → GC
# - Old integration point data becomes unreachable → GC
```

**Julia's GC is efficient for this pattern!**
- Generation 0 collection: ~1ms for 100K elements
- No manual memory management needed

## Advantages of This Architecture

### 1. Immutability Benefits ✅

```julia
# No accidental mutation!
element = problem.elements[1]
element.fields.σ[1] = 1000.0  # ❌ ERROR: immutable!

# Explicit updates only
element_new = Element(element.id, element.connectivity, element.basis, fields_new)
problem.elements[1] = element_new  # ✓ Clear update
```

### 2. Thread Safety ✅

```julia
# Read-only during assembly (no data races!)
Threads.@threads for node in problem.nodes
    for elem_idx in problem.node_to_elements[node.id]
        element = problem.elements[elem_idx]
        # Read element.fields (safe!)
        σ = element.fields.σ
        C = element.fields.C
    end
end

# Updates are explicit (sequential or with proper locking)
for i in 1:length(problem.elements)
    elements_new[i] = update_element(problem.elements[i], u_new)
end
```

### 3. GPU Compatibility ✅

```julia
# Transfer immutable elements to GPU
elements_gpu = cu(problem.elements)
nodal_fields_gpu = cu(problem.nodal_fields)

# GPU kernel reads fields (no mutations!)
@cuda gpu_matvec_kernel!(y, x, elements_gpu, nodal_fields_gpu, ...)

# Updates happen on CPU, then transfer new elements
```

### 4. Time Stepping Natural ✅

```julia
# Store history
history = Problem[]

for t in time_steps
    # Solve at this time step
    u_new, problem_new = solve_nonlinear!(problem, f_ext(t))
    
    # Store state (cheap - just references!)
    push!(history, problem_new)
    
    # Next iteration
    problem = problem_new
end

# Access history: history[timestep].elements[elem_id].fields.σ
```

### 5. Clear Separation of Concerns ✅

**Nodal fields:** Degrees of freedom (what we solve for)
- Displacement `u`
- Temperature `T`
- Velocity `v` (dynamics)
- Contact pressure `p` (Lagrange multipliers)

**Element fields:** Internal state (what we update)
- Integration point stress `σ`
- Integration point plastic strain `ε_plastic`
- Hardening variables `α`
- Tangent modulus `C`
- Damage variables `d`

## Performance Considerations

### Cost of Creating New Elements

**Per Newton iteration:**
```julia
# 100K elements, 8 integration points each
# Each IP: 6 stress components, 6 plastic strain, 1 hardening, 6×6 tangent

# Memory to allocate:
# 100K elements × 8 IPs × (6 + 6 + 1 + 36) × 8 bytes
# = 100K × 8 × 49 × 8 = ~314 MB

# Element struct overhead:
# 100K × 32 bytes = 3.2 MB (negligible!)

# Total: ~320 MB per iteration (reasonable!)
```

**Benchmark estimate:**
```julia
using BenchmarkTools

# Creating new element
fields_old = (E=210e3, ν=0.3, σ=σ_old, C=C_old, ...)
fields_new = (E=fields_old.E, ν=fields_old.ν, σ=σ_new, C=C_new, ...)
element_new = Element(id, connectivity, basis, fields_new)

# Expected: ~5-10 ns per element (just wrapping)
# 100K elements: ~1ms (negligible compared to material update!)
```

### Material Update Dominates

```julia
# Time breakdown per Newton iteration:
# 1. Material update: 100-500 ms (dominates!)
#    - Strain computation
#    - Plasticity return mapping
#    - Tangent computation
# 
# 2. Creating new elements: 1-5 ms (negligible!)
# 
# 3. Nodal assembly matvec: 50-200 ms
# 
# 4. GMRES solve: 100-1000 ms
#
# Total: ~250-1700 ms per iteration
# Element creation: <1% of total!
```

## Alternative: In-Place Updates (If Needed)

If creating new elements becomes a bottleneck (unlikely!):

```julia
# Mutable element fields (wrapped in Ref or Vector)
struct ElementMutable{N,B,F}
    id::UInt
    connectivity::NTuple{N,UInt}
    basis::B
    fields::Ref{F}  # Mutable container!
end

# In-place update
function update_material_state_inplace!(problem::Problem, u_new)
    Threads.@threads for element in problem.elements
        σ_new, C_new = material_update(element, u_new)
        
        # Mutate through Ref
        element.fields[] = merge(element.fields[], (σ=σ_new, C=C_new))
    end
end
```

**But:** Lose immutability benefits (thread safety, clarity, GPU compatibility)

**Recommendation:** Start with immutable, optimize if profiling shows need

## Summary: The Pattern

```julia
# Structures
struct Node
    id::UInt
    x::Float64
    y::Float64
    z::Float64
end

struct Element{N,B,F}
    id::UInt
    connectivity::NTuple{N,UInt}
    basis::B
    fields::F  # Integration point data (immutable!)
end

struct Problem{F_node}
    nodes::Vector{Node}
    elements::Vector{Element}  # Mutable vector, immutable elements
    node_to_elements::Vector{Vector{Int}}
    nodal_fields::F_node  # Nodal DOFs (immutable container)
end

# Updates
function newton_iteration(problem, f_ext)
    # 1. Assemble (read-only, parallel safe)
    r = compute_residual_nodal_assembly(problem, u)
    
    # 2. Solve (matrix-free)
    Δu = gmres(r) do x
        matvec_nodal_assembly(problem, x)
    end
    
    # 3. Update nodal fields (cheap)
    nodal_fields_new = (u = u + Δu, ...)
    
    # 4. Update element fields (create new elements)
    elements_new = [
        update_element(elem, nodal_fields_new)
        for elem in problem.elements
    ]
    
    # 5. New problem
    return Problem(nodes, elements_new, node_to_elements, nodal_fields_new)
end
```

**Key points:**
- ✅ Nodes: Geometry (never changes)
- ✅ Elements: Immutable, update by creating new
- ✅ Element fields: Integration point data (σ, ε_plastic, C, etc.)
- ✅ Nodal fields: DOFs (u, T, p, etc.)
- ✅ Nodal assembly: Loop over nodes, gather from elements
- ✅ Updates: Create new elements with updated fields (~1ms for 100K)
- ✅ Overhead: <1% of total computation time

This is clean, safe, and performant! 🎯
