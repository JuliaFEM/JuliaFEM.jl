---
title: "Element Field Architecture: Practical Design Choices"
description: "Should fields be a type parameter? What about immutability vs GPU performance?"
date: "November 9, 2025"
author: "Jukka Aho"
categories: ["architecture", "design", "performance"]
keywords: ["Element", "fields", "type-parameters", "GPU", "immutability"]
audience: "contributors"
level: "advanced"
type: "design-discussion"
series: "The JuliaFEM Book"
status: "active-discussion"
---

## The Question

**User asks:** "Should we add another parameter for our element model describing the field system? Does it have to be a NamedTuple, or can it be some custom struct?"

**Context:** Benchmarks show NamedTuples give 9-92× speedup, but they're immutable. We need immutability for GPU, but it feels restrictive.

---

## Current Architecture (v0.5.1)

```julia
struct Element{N,NIP,M,B} <: AbstractElement{M,B}
    id::UInt
    connectivity::NTuple{N,UInt}  
    integration_points::NTuple{NIP,IP}
    dfields::Dict{Symbol,AbstractField}  # ❌ Type-unstable!
    sfields::M  # Static fields (type-stable)
    properties::B  # Basis type
end
```

**Problems:**

- `dfields::Dict{Symbol,AbstractField}` is type-unstable (9-92× slower)
- `M` type parameter already exists for static fields (`AbstractFieldSet{N}`)
- Two field systems (dynamic + static) is confusing

---

## Design Options

### Option 1: Fields as Type Parameter (Current Style)

```julia
# Keep M parameter, make it the primary field storage
struct Element{N,NIP,F,B} <: AbstractElement{F,B}
    id::UInt
    connectivity::NTuple{N,UInt}  
    integration_points::NTuple{NIP,IP}
    fields::F  # Could be NamedTuple, struct, anything type-stable
    basis::B
end

# Example usage:
fields = (E = 210e3, ν = 0.3, u = zeros(3, 8))
element = Element{8, 4, typeof(fields), Lagrange{Triangle,1}}(
    UInt(1), (1,2,3,4,5,6,7,8), ips, fields, Lagrange{Triangle,1}()
)

# Access (type-stable!):
E = element.fields.E  # Float64
```

**Pros:**
- Full type stability (compiler knows `F` at compile time)
- Maximum performance (zero runtime overhead)
- GPU-compatible (immutable if F is immutable)
- Flexibility: `F` can be NamedTuple, custom struct, anything

**Cons:**
- Different field types = different element types
- Cannot have heterogeneous vectors `Vector{Element}` with different field types
- Type parameters proliferate: `Element{8,4,MyFields,Lagrange{Triangle,1}}`

### Option 2: Fields NOT in Element (ElementSet Pattern)

```julia
# Element has NO fields
struct Element{N,NIP,B}
    id::UInt
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}
    basis::B
end

# Fields belong to ElementSet
struct ElementSet{E,F}
    name::String
    elements::Vector{E}
    fields::F  # Shared by all elements in set
end

# Example:
elements = [Element(...), Element(...), ...]
fields = (E = 210e3, ν = 0.3, u = zeros(3, n_nodes))
element_set = ElementSet("steel_body", elements, fields)

# Access during assembly:
for element in element_set.elements
    E = element_set.fields.E  # Type-stable
    K_local = assemble(element, E, ν, cache)
end
```

**Pros:**
- Elements are homogeneous (same type regardless of fields)
- Matches physical reality (material properties per set, not per element)
- Cleaner separation of concerns (geometry vs properties)
- Easy to have `Vector{Element}` with different topologies

**Cons:**
- Fields are external to element (need to pass around)
- API changes: can't do `element.fields.E` anymore
- More complex data flow (element + fields, not just element)

### Option 3: Hybrid (Type Parameter + External Storage)

```julia
# Element has type parameter F, but fields can be external
struct Element{N,NIP,F,B}
    id::UInt
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}
    fields::F  # Can be actual fields OR reference to external storage
    basis::B
end

# Option A: Element-local fields
fields_local = (E = 210e3, ν = 0.3)
element1 = Element{8,4,typeof(fields_local),Tri3}(1, conn, ips, fields_local, basis)

# Option B: Reference to external storage
struct FieldRef{T}
    set_id::Int
end
field_ref = FieldRef{MyFieldType}(1)
element2 = Element{8,4,typeof(field_ref),Tri3}(2, conn, ips, field_ref, basis)

# Accessor handles both:
get_field(element::Element{N,NIP,F,B}, name) where {F<:NamedTuple} = getproperty(element.fields, name)
get_field(element::Element{N,NIP,F,B}, name) where {F<:FieldRef} = getproperty(global_fields[F.set_id], name)
```

**Pros:**
- Flexibility: supports both patterns
- Type-stable in both cases
- Can migrate gradually

**Cons:**
- Complex dispatch logic
- Two code paths to maintain
- Harder to understand

---

## The Immutability "Problem" (Not Actually a Problem)

**User concern:** "NamedTuple is immutable, but we need to update fields during simulation."

**Reality check:** We DON'T actually need mutable fields in the hot path!

### What Actually Happens During Simulation

```julia
# Time stepping loop:
for time in timesteps
    # 1. Assemble (reads fields, doesn't modify)
    K = assemble(elements, fields, cache)  # ← Hot path, needs immutability!
    
    # 2. Solve (creates NEW displacement field)
    u_new = solve(K, f)  # ← Returns new array
    
    # 3. Update fields (OUTSIDE hot path)
    fields = (
        E = fields.E,  # Keep old
        ν = fields.ν,  # Keep old  
        u = u_new,     # New displacement
    )
    # OR if fields is a struct:
    fields = FieldData(fields.E, fields.ν, u_new)
end
```

**Key insight:** We create NEW field containers between time steps, not mutate old ones!

### What About Material State (Plasticity)?

```julia
# Material state IS mutable, but stored separately:
struct MaterialState
    stress::Matrix{Float64}      # Mutable!
    plastic_strain::Matrix{Float64}
    damage::Vector{Float64}
end

# Fields contain PARAMETERS (immutable)
fields = (
    E = 210e3,    # Young's modulus (constant)
    ν = 0.3,      # Poisson's ratio (constant)
    yield = 250.0, # Yield stress (constant)
)

# State updated separately (outside hot assembly loop)
for element in elements
    for ip in integration_points
        # Read parameters (immutable, fast)
        params = fields
        
        # Read/write state (mutable, but not in type-unstable way)
        state = material_state[element.id, ip.id]
        
        # Update state
        stress_new, state_new = material_model(params, state, strain)
        material_state[element.id, ip.id] = state_new
    end
end
```

**Pattern:** 
- **Parameters** (E, ν, yield stress): Immutable, in type-stable container
- **State** (stress, plastic strain): Mutable, in separate typed arrays
- **Solution** (u, T): Immutable (replaced between time steps)

---

## Recommendation: Option 1 + ElementSet Pattern

**Proposal:**

```julia
# 1. Element with field type parameter
struct Element{N,NIP,B}  # Note: NO field parameter yet
    id::UInt
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}
    basis::B
end

# 2. ElementSet groups elements + fields
struct ElementSet{E<:Element,F}
    name::String
    elements::Vector{E}
    fields::F  # Type-stable! Can be NamedTuple, custom struct, anything
end

# 3. Assembly uses element set
function assemble!(K, f, element_set::ElementSet, cache)
    # Access fields once (type-stable)
    fields = element_set.fields
    
    # Loop over elements
    for element in element_set.elements
        # All field accesses are type-stable!
        K_local = assemble_element!(cache, element, fields)
        add_to_global!(K, f, element, K_local)
    end
end

# 4. Field types can be anything type-stable
const BasicFields = @NamedTuple{E::Float64, ν::Float64}
const ElasticityFields = @NamedTuple{E::Float64, ν::Float64, u::Matrix{Float64}}

# Or custom struct:
struct MyCustomFields
    E::Float64
    ν::Float64
    u::Matrix{Float64}
    temperature::Vector{Float64}
end

# Both work! Type stability is what matters.
```

**Why this works:**

1. **Type stability:** `F` is known at compile time for each ElementSet
2. **Immutability:** Fields can be immutable (create new containers between steps)
3. **Flexibility:** `F` can be NamedTuple, custom struct, whatever you want
4. **GPU-ready:** Immutable access patterns, zero allocations
5. **Physical meaning:** Matches how engineers think (material per set)

---

## FAQ

### Q: Can't I just make fields mutable?

**A:** You can, but you lose GPU compatibility and threading safety:

```julia
# Mutable struct (works on CPU, breaks on GPU)
mutable struct MutableFields
    E::Float64
    u::Matrix{Float64}
end

# Problem: GPU kernel can't modify host memory
function gpu_kernel(element, fields::MutableFields)  # ❌ Won't work
    fields.E = 220e3  # Error: Can't modify host data from GPU
end

# Immutable pattern works everywhere:
function gpu_kernel(element, E::Float64, u::Matrix{Float64})  # ✅ Works
    # E and u are copied to GPU, read-only access
    stress = compute_stress(E, u)
end
```

### Q: What about time-dependent fields?

**A:** Store time series, interpolate at access:

```julia
struct TimeField{T}
    times::Vector{Float64}
    values::Vector{T}
end

function interpolate(field::TimeField, t::Float64)
    # Linear interpolation
    idx = searchsortedfirst(field.times, t)
    if idx == 1
        return field.values[1]
    elseif idx > length(field.times)
        return field.values[end]
    else
        t0, t1 = field.times[idx-1], field.times[idx]
        v0, v1 = field.values[idx-1], field.values[idx]
        α = (t - t0) / (t1 - t0)
        return (1-α) * v0 + α * v1
    end
end

# Usage in fields:
fields = (
    E = 210e3,  # Constant
    temperature = TimeField([0.0, 1.0, 2.0], [20.0, 100.0, 50.0]),  # Time-dependent
)

# Access at specific time:
T = interpolate(fields.temperature, 0.5)  # Returns 60.0
```

### Q: Do I have to use NamedTuple?

**A:** No! Any type-stable container works:

```julia
# Option 1: NamedTuple (simplest)
fields1 = (E = 210e3, ν = 0.3)

# Option 2: Custom struct (more control)
struct ElasticityFields
    E::Float64
    ν::Float64
end
fields2 = ElasticityFields(210e3, 0.3)

# Option 3: Macro-generated struct
@fields MyFields begin
    @constant E::Float64
    @constant ν::Float64
    @nodal u::Vec3
end
fields3 = MyFields(210e3, 0.3, zeros(Vec3, n_nodes))

# All are type-stable! Pick what you like.
```

---

## Implementation Roadmap

### Phase 1: Remove Dict fields, add ElementSet

```julia
# 1. Update Element (remove dfields)
struct Element{N,NIP,B}
    id::UInt
    connectivity::NTuple{N,UInt}
    integration_points::NTuple{NIP,IP}
    basis::B
    # No fields here!
end

# 2. Create ElementSet
struct ElementSet{E,F}
    name::String
    elements::Vector{E}
    fields::F
end

# 3. Update assembly
function assemble!(problem::Problem, cache)
    K = spzeros(problem.ndofs, problem.ndofs)
    f = zeros(problem.ndofs)
    
    for element_set in problem.element_sets
        assemble!(K, f, element_set, cache)
    end
    
    return K, f
end
```

### Phase 2: Benchmark and validate

```julia
# Run benchmarks
julia> @btime assemble!($problem, $cache)
# Target: 0 allocations in loop

# Verify GPU compatibility
julia> using CUDA
julia> @cuda threads=256 blocks=ceil(Int, length(elements)/256) gpu_assembly_kernel(...)
# Should compile and run
```

### Phase 3: Add convenience wrappers

```julia
# Macro for defining field types
@fields ElasticityFields begin
    @constant youngs_modulus::Float64
    @constant poissons_ratio::Float64
    @nodal displacement::Vec3
    @nodal velocity::Vec3
end

# Auto-generates:
struct ElasticityFields
    youngs_modulus::Float64
    poissons_ratio::Float64
    displacement::Matrix{Vec3}  # N_nodes columns
    velocity::Matrix{Vec3}
end
```

---

## Decision

**Recommended approach:**

1. ✅ **Remove field type parameter from Element** (keep Element simple)
2. ✅ **Use ElementSet pattern** (fields belong to sets, not elements)
3. ✅ **Fields can be any type-stable container** (NamedTuple, struct, whatever)
4. ✅ **Embrace immutability** (create new containers, don't mutate)
5. ✅ **Separate state from parameters** (mutable state in arrays, immutable params in fields)

**Why:**
- Simple Element type (fewer type parameters)
- Matches physical reality (properties per set)
- Maximum type stability and GPU compatibility
- Flexibility in field container choice
- Clean separation of concerns

**Not recommended:**
- ❌ Keeping `Dict{Symbol,AbstractField}` (9-92× slower, GPU-incompatible)
- ❌ Adding field type parameter to Element (type proliferation)
- ❌ Mutable field containers (breaks GPU, threading unsafe)

---

**Status:** Ready for implementation  
**Next:** Update `src/elements/elements.jl` to implement ElementSet pattern  
**Validation:** Benchmarks must show 0 allocations in assembly loop
