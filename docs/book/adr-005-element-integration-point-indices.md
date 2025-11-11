---
title: "ADR-005: Integration Point Indices Instead of Data"
date: 2025-11-11
author: "Jukka Aho"
status: "Accepted"
tags: ["architecture", "elements", "integration-points", "material-state"]
---

## Status

**Accepted** (November 11, 2025)

## Context

Current `Element` struct stores integration points as data:

```julia
struct Element{N,NIP,F,B} <: AbstractElement{F,B}
    id::UInt
    connectivity::NTuple{N,UInt}           # Node IDs
    integration_points::NTuple{NIP,IP}     # ← Integration point DATA
    fields::F
    basis::B
end
```

**Problems with this approach:**

1. **Redundancy**: All elements of the same type have identical integration points
   - Seg2 elements all use same 2-point Gauss quadrature
   - Tri3 elements all use same 3-point triangle quadrature
   - Storing this in every element wastes memory

2. **Inconsistent with connectivity pattern**: Connectivity stores **indices** (node IDs), not node data
   - We don't store node coordinates in element
   - We store node **IDs** and look up coordinates elsewhere

3. **Inconsistent with material state design**: Material state is stored separately
   - Material state per integration point stored in global vector
   - Element needs **indices** to find its material state
   - Current design has no way to link element to material state

4. **Assembly workflow needs indices**:

   ```julia
   for element in elements
       node_ids = element.connectivity  # Get node indices
       node_coords = X[node_ids]        # Look up coordinates
       
       # What about material state?
       # Need: ip_ids = element.ip_indices  # Get IP indices
       #       mat_state = material_states[ip_ids]  # Look up state
   ```

## Decision

**Store integration point INDICES instead of DATA in elements.**

### New Element Structure

```julia
struct Element{N,NIP,F,B} <: AbstractElement{F,B}
    id::UInt
    connectivity::NTuple{N,UInt}        # Node IDs (indices)
    ip_indices::NTuple{NIP,UInt}        # ← Integration point IDs (indices)
    fields::F
    basis::B
end
```

### Global Integration Point Registry

Integration points are computed once per topology+order combination:

```julia
# Precompute integration points for each topology type
const INTEGRATION_POINTS = Dict{Tuple{Type,Int}, Vector{Tuple{Float64,Vec}}}(
    (Triangle, 1) => get_gauss_points!(Triangle, Gauss{1}),
    (Triangle, 2) => get_gauss_points!(Triangle, Gauss{2}),
    (Quadrilateral, 2) => get_gauss_points!(Quadrilateral, Gauss{2}),
    # ... etc
)

# Material state stored separately (one entry per integration point)
struct MaterialState
    stress::SymmetricTensor{2,3}
    strain::SymmetricTensor{2,3}
    plastic_strain::SymmetricTensor{2,3}
    # ... etc
end

material_states::Vector{MaterialState} = [MaterialState(...) for _ in 1:n_total_ips]
```

### Assembly Workflow

```julia
# Element creation (during mesh setup)
element_id = 1
node_ids = (1, 2, 3)  # Triangle nodes
ip_start = 1000       # First IP for this element
ip_indices = (ip_start, ip_start+1, ip_start+2)  # 3 IPs for Tri3
element = Element(Triangle, node_ids, ip_indices=ip_indices)

# Assembly loop
for element in elements
    # Get node data
    node_ids = element.connectivity
    X = node_coords[node_ids]      # Coordinates
    u = displacements[node_ids]    # Displacements
    
    # Get integration points (from global registry)
    topology = typeof(element.basis).parameters[1]  # Triangle
    order = typeof(element.basis).parameters[2]     # 1
    ips = INTEGRATION_POINTS[(topology, order)]
    
    # Get material state (from global vector)
    ip_ids = element.ip_indices
    mat_states = material_states[ip_ids]
    
    # Assembly loop over integration points
    for (i, (w, ξ)) in enumerate(ips)
        mat_state = mat_states[i]
        
        # Compute strain from displacements
        ε = compute_strain(element, u, ξ)
        
        # Compute stress from material state
        σ = compute_stress(mat_state, ε)
        
        # Assemble...
    end
end
```

## Consequences

### Positive

1. **Memory efficiency**: 
   - Old: 1M Tet10 elements × 4 IPs × 32 bytes = 128 MB for IP data
   - New: 1 global table × 4 IPs × 32 bytes = 128 bytes (1000× reduction!)

2. **Consistency**: Same pattern for all element data
   - Nodes: Store indices, look up data
   - Integration points: Store indices, look up data
   - Material state: Store indices, look up data

3. **Clear data ownership**:
   - Node coordinates: Owned by mesh/node array
   - Integration points: Owned by global registry
   - Material state: Owned by material state vector
   - Element: Only owns **indices** to these

4. **GPU-friendly**:
   - All material states in one contiguous array (easy to transfer)
   - All integration point data in global tables (transfer once)
   - Elements are small (just indices)

5. **Natural partitioning**:
   - Element ownership → node ownership (already clear)
   - Integration point ownership → element ownership (via indices)
   - Clear for domain decomposition / MPI

### Negative

1. **API change**: Existing code expects `element.integration_points`
   - **Mitigation**: Provide compatibility function `get_integration_points(element)`

2. **Index management**: Need to assign IP indices during mesh setup
   - **Mitigation**: Automatic during element creation

3. **Lookup overhead**: Extra indirection to get IP data
   - **Mitigation**: Negligible (one array lookup), data still cache-friendly

## Implementation Plan

### Phase 1: Add ip_indices field (keep integration_points for compatibility)

```julia
struct Element{N,NIP,F,B}
    id::UInt
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}  # DEPRECATED, keep temporarily
    ip_indices::NTuple{NIP,UInt}        # NEW
    fields::F
    basis::B
end
```

### Phase 2: Update assembly code to use indices

- Update `assemble!` functions to use `element.ip_indices`
- Look up IP data from global registry
- Look up material state from global vector

### Phase 3: Remove integration_points field

```julia
struct Element{N,NIP,F,B}
    id::UInt
    connectivity::NTuple{N,UInt}
    ip_indices::NTuple{NIP,UInt}  # Only indices remain
    fields::F
    basis::B
end
```

## Rationale

**Core principle**: Elements should store **relationships** (indices), not **data**.

Just as we don't store node coordinates in elements, we shouldn't store integration point data. The element's job is to define **which** nodes and **which** integration points participate in its stiffness matrix, not to own their data.

This aligns perfectly with the nodal assembly approach documented in `docs/src/book/multigpu_nodal_assembly.md`:
- Nodes own displacement DOFs
- Integration points own material state
- Elements define relationships between them

## References

- **Golden Standard**: `docs/src/book/multigpu_nodal_assembly.md` - Nodal assembly architecture
- **ADR-004**: Integration Points API Design
- **Material State Design**: `llm/FIELDS_DESIGN.md` - Field system redesign

## Notes

This decision was made during test migration (November 11, 2025) when updating Dirichlet tests to use immutable Element API. The question arose: "Why does each element store integration points when they're identical for all elements of the same type?"

The answer: **They shouldn't.** This ADR documents the correct architecture.
