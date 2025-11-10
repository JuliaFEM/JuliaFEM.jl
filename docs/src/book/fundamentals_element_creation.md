---
title: "Fundamentals: Element Creation"
date: 2025-11-10
author: "JuliaFEM Development Team"
status: "Authoritative - defines element creation API"
last_updated: 2025-11-10
tags: ["fundamentals", "elements", "API", "immutability"]
---

## Two Ways to Create Elements

JuliaFEM supports **two approaches** to element creation. Both are fully supported with no plans for deprecation.

### Modern API (Recommended) ✅

```julia
# Explicit topology and basis separation:
element = Element(Topology, Lagrange{Topology, Order}, connectivity; fields=(...))
```

**Example:**

```julia
el = Element(Triangle, Lagrange{Triangle, 1}, (1, 2, 3); fields=(E=210e3, ν=0.3))
```

**Advantages:**

- **Explicit and clear** - All parameters visible at construction
- **Type-stable** - Compiler knows everything at compile time
- **GPU-compatible** - Zero-allocation, immutable design
- **40-130× faster** - Field access performance ([see benchmarks](../blog/immutability_performance.md))
- **Future-proof** - Supports arbitrary basis types beyond Lagrange

**When to use:** New code, performance-critical applications, GPU computations, research

---

### Legacy API (Convenient) ✅

```julia
# Automatic basis inference from node count:
element = Element(Topology, connectivity; fields=(...))
```

**Example:**

```julia
el = Element(Triangle, (1, 2, 3); fields=(E=210e3, ν=0.3))  # → Infers Lagrange{Triangle,1}
```

**Advantages:**

- **Concise** - Less typing for simple cases
- **Backward compatible** - Works with all existing JuliaFEM code
- **Convenient** - Great for prototyping and educational examples
- **Automatic** - Infers basis order from number of nodes

**When to use:** Quick prototyping, backward compatibility, simple problems, teaching

---

**Important Notes:**

1. **Both APIs work correctly!** The legacy API internally converts to the modern architecture
2. **No deprecation planned** - Legacy API will remain supported indefinitely
3. **Internal consistency** - Both APIs create identical element structures
4. **Choose what fits** - Use modern for clarity, legacy for convenience

---

## Philosophy: Separation of Concerns

Both APIs respect the same fundamental principle: **separate topology, basis, connectivity, and fields**.

### The Four Concerns

1. **Topology** - Geometric shape (Segment, Triangle, Tetrahedron, etc.)
2. **Basis** - Interpolation scheme (Lagrange P1, P2, P3, etc.)
3. **Connectivity** - Which nodes form this element
4. **Fields** - Material properties, state variables (optional, type-stable)

### Why Separate?

**1. Clarity**
- Each concept has its own type/parameter
- Easier to understand what each part does
- Less cognitive load when reading code

**2. Reusability**
- Same topology can use different basis functions
- Topology and basis can be developed independently
- Share implementations across elements

**3. Type Stability**
- Compiler knows all types at compile time
- Enables aggressive optimizations
- **40-130× faster** field access vs Dict-based approach
- See [performance benchmarks](../blog/immutability_performance.md)

**4. GPU Compatibility**
- Immutable structures transfer efficiently to GPU
- Type-stable → GPU kernels can be specialized
- Zero-allocation → no garbage collection needed

**The Difference?**
- **Modern API**: Makes separation explicit in constructor
- **Legacy API**: Infers basis from node count, separation still exists internally

---

## Parameters Reference

### Topology Types

| Topology | Description | Linear Nodes | Quadratic Nodes |
|----------|-------------|--------------|-----------------|
| `Segment` | 1D line | 2 | 3 |
| `Triangle` | 2D triangle | 3 | 6 |
| `Quadrilateral` | 2D quad | 4 | 8 or 9 |
| `Tetrahedron` | 3D tet | 4 | 10 |
| `Hexahedron` | 3D hex | 8 | 20 or 27 |
| `Pyramid` | 3D pyramid | 5 | - |
| `Wedge` | 3D prism | 6 | 15 |

**Location:** `src/topology/`  
**Documentation:** [Element Architecture](element_architecture.md)

### Basis Types (Modern API)

```julia
Lagrange{Topology, Order}
```

**Order values:**

- `1` → Linear (P1) - corner nodes only
- `2` → Quadratic (P2) - corner + mid-edge nodes
- `3` → Cubic (P3) - corner + edge + face nodes (future)

**Examples:**

- `Lagrange{Segment, 1}` - Linear 1D
- `Lagrange{Triangle, 2}` - Quadratic 2D triangle
- `Lagrange{Tetrahedron, 1}` - Linear 3D tet

**Location:** `src/basis/`  
**Documentation:** [Lagrange Basis Functions](lagrange_basis_functions.md)

### Connectivity

Node IDs forming the element:

```julia
connectivity = (1, 2, 3)        # Tuple (preferred - zero allocation)
connectivity = [1, 2, 3]        # Vector (auto-converted to tuple)
```

**Convention:**

- Positive integers (converted to `UInt` internally)
- Order matters (defines element orientation)
- Tuple preferred for performance

### Fields (Optional)

Type-stable container for element properties:

```julia
fields = (E = 210e3, ν = 0.3, thickness = 0.01)
```

**Requirements:**

- **Type-stable**: NamedTuple or custom struct (no Dict!)
- **Immutable**: Cannot modify after creation
- **Optional**: Default is empty tuple `()`

**Common fields:**

- Material: `E`, `ν`, `G`, `K`, `ρ`
- Geometry: `thickness`, `area`, `volume`
- State: `temperature`, `displacement`, `stress`

---

## Examples: Side-by-Side Comparison

### 1D: Linear Segment (2 nodes)

```julia
# Modern API (explicit):
el = Element(Segment, Lagrange{Segment,1}, (1, 2))

# Legacy API (inferred):
el = Element(Segment, (1, 2))  # → Lagrange{Segment,1} automatically
```

### 1D: Quadratic Segment (3 nodes)

```julia
# Modern API (explicit):
el = Element(Segment, Lagrange{Segment,2}, (1, 2, 3))

# Legacy API (inferred):
el = Element(Segment, (1, 2, 3))  # → Lagrange{Segment,2} from node count
```

### 2D: Linear Triangle (3 nodes)

```julia
# Modern API (explicit):
el = Element(Triangle, Lagrange{Triangle,1}, (1, 2, 3);
             fields=(E=210e3, ν=0.3))

# Legacy API (inferred):
el = Element(Triangle, (1, 2, 3);
             fields=(E=210e3, ν=0.3))  # → Lagrange{Triangle,1}
```

### 2D: Quadratic Triangle (6 nodes)

```julia
# Modern API (explicit):
el = Element(Triangle, Lagrange{Triangle,2}, (1,2,3,4,5,6))

# Legacy API (inferred):
el = Element(Triangle, (1,2,3,4,5,6))  # → Lagrange{Triangle,2}
```

### 2D: Bilinear Quadrilateral (4 nodes)

```julia
# Modern API (explicit):
el = Element(Quadrilateral, Lagrange{Quadrilateral,1}, (1,2,3,4))

# Legacy API (inferred):
el = Element(Quadrilateral, (1,2,3,4))  # → Lagrange{Quadrilateral,1}
```

### 2D: Serendipity Quadrilateral (8 nodes)

```julia
# Modern API (explicit):
el = Element(Quadrilateral, Lagrange{Quadrilateral,2}, (1,2,3,4,5,6,7,8))

# Legacy API (inferred):
el = Element(Quadrilateral, (1,2,3,4,5,6,7,8))  # → Lagrange{Quadrilateral,2}
```

### 3D: Linear Tetrahedron (4 nodes)

```julia
# Modern API (explicit):
el = Element(Tetrahedron, Lagrange{Tetrahedron,1}, (1,2,3,4))

# Legacy API (inferred):
el = Element(Tetrahedron, (1,2,3,4))  # → Lagrange{Tetrahedron,1}
```

### 3D: Quadratic Tetrahedron (10 nodes)

```julia
# Modern API (explicit):
el = Element(Tetrahedron, Lagrange{Tetrahedron,2}, (1,2,3,4,5,6,7,8,9,10))

# Legacy API (inferred):
el = Element(Tetrahedron, (1,2,3,4,5,6,7,8,9,10))  # → Lagrange{Tetrahedron,2}
```

### 3D: Trilinear Hexahedron (8 nodes)

```julia
# Modern API (explicit):
el = Element(Hexahedron, Lagrange{Hexahedron,1}, (1,2,3,4,5,6,7,8))

# Legacy API (inferred):
el = Element(Hexahedron, (1,2,3,4,5,6,7,8))  # → Lagrange{Hexahedron,1}
```

---

## Updating Elements (Immutable API)

Elements are **immutable** for performance (40-130× faster field access). To "update" an element, create a new one with modified fields.

### Old API (Deprecated) ❌

```julia
element = Element(Triangle, (1,2,3); fields=(E=210e3,))
update!(element, "E", 200e3)  # DEPRECATED - mutates element
```

**Problems:**

- Mutation breaks type stability
- Incompatible with GPU
- 100× slower field access
- Not thread-safe

### New API (Immutable) ✅

```julia
element = Element(Triangle, (1,2,3); fields=(E=210e3,))
element = update(element, :E => 200e3)  # Returns NEW element
```

**Advantages:**

- Type-stable (40-130× faster)
- GPU-compatible
- Thread-safe
- Functional programming style

### Update Examples

```julia
# Single field:
el2 = update(el, :E => 200e3)

# Multiple fields:
el3 = update(el, :E => 200e3, :ν => 0.35)

# Keyword syntax:
el4 = update(el; E=200e3, ν=0.35)

# Add new field:
el5 = update(el, :temperature => 293.15)

# Original unchanged:
@assert el.fields.E == 210e3  # Original still 210e3
@assert el2.fields.E == 200e3  # New element has 200e3
```

**Performance:** Zero-allocation when field types match.

---

## Design Evolution & Rationale

### Historical Context

**2015-2019 (v0.5.1):** Mixed basis/topology types
- Used `Seg2`, `Tri3`, `Tet10` (basis+topology combined)
- Convenient but limiting
- Difficult to support multiple basis types

**2025 (v1.0 development):** Separated architecture
- Topology types: `Segment`, `Triangle`, `Tetrahedron`
- Basis types: `Lagrange{Topology, Order}`
- Both APIs supported for smooth transition

### Why the Change?

**Problem 1: Type Confusion**
```julia
# Old: What is Tri3?
el = Element(Tri3, (1,2,3))  # Topology? Basis? Both?
```

**Solution:**

```julia
# Modern: Clear separation
el = Element(Triangle, Lagrange{Triangle,1}, (1,2,3))  # Explicit!

# Legacy: Still works
el = Element(Triangle, (1,2,3))  # Infers basis, clear topology
```

**Problem 2: Limited Extensibility**
- Old: To add cubic basis, need `Tri10` (but quadratic uses 6 nodes, not 10!)
- New: Just add `Lagrange{Triangle, 3}` - systematic

**Problem 3: Performance**
- Old mutable API: Dict-based fields, 100× slower
- New immutable API: NamedTuple fields, 40-130× faster
- See [performance analysis](../blog/immutability_performance.md)

### Design Alternatives Considered

We evaluated three approaches:

**Alternative 1: Combined types (old way)**
```julia
Element(Tri3, (1,2,3))  # Tri3 means triangle + P1 basis
```
- ❌ Extensibility issues
- ❌ Type name confusion
- ✅ Very concise

**Alternative 2: Separate parameters (modern way)**
```julia
Element(Triangle, Lagrange{Triangle,1}, (1,2,3))
```
- ✅ Clear and explicit
- ✅ Extensible to any basis
- ✅ Type-stable
- ❌ More verbose

**Alternative 3: String-based dispatch**
```julia
Element("triangle", "lagrange", order=1, connectivity=(1,2,3))
```
- ❌ Not type-stable
- ❌ Runtime errors instead of compile-time
- ❌ Poor performance

**Decision:** Support both Alternative 1 (legacy) and Alternative 2 (modern) ✅

### Why Not Deprecate Legacy API?

**Reasons to keep legacy API:**

1. **Backward compatibility** - Thousands of lines of existing code
2. **Convenience** - Simple cases don't need explicit basis
3. **Teaching** - Easier for beginners to get started
4. **No cost** - Internally converts to modern architecture anyway
5. **Clear inference** - Node count uniquely determines basis order (for Lagrange)

**When legacy API is perfect:**

- Quick prototyping
- Educational examples
- Simple problems with standard Lagrange elements
- Porting code from other FEM libraries

**When modern API shines:**

- Production code (explicit is better)
- Performance-critical applications
- GPU computing
- Research with custom basis functions
- Large collaborative projects (clarity matters)

---

## Related Documentation

**Architecture & Design:**

- [Element Architecture](element_architecture.md) - Separation of concerns philosophy
- [ARCHITECTURE.md](../../llm/ARCHITECTURE.md) - System-wide architecture
- [TECHNICAL_VISION.md](../../llm/TECHNICAL_VISION.md) - Strategic lessons from v0.5.1

**Performance:**

- [Immutability Performance Analysis](../blog/immutability_performance.md) - 40-130× speedup data
- [Struct Size Scaling Benchmark](../../benchmarks/struct_size_scaling.jl) - Raw measurements

**Implementation:**

- [src/topology/](../../src/topology/) - Topology type definitions
- [src/basis/](../../src/basis/) - Basis function implementations
- [src/elements/elements.jl](../../src/elements/elements.jl) - Element constructors

**Design Documents:**

- [IMMUTABILITY.md](../design/IMMUTABILITY.md) - Why immutable elements?
- [FIELDS_DESIGN.md](../../llm/FIELDS_DESIGN.md) - Field system design (future)

---

## Common Pitfalls

### ❌ Pitfall 1: Using Old Type Names

```julia
# DON'T (old type names):
element = Element(Seg2, (1, 2))
element = Element(Tri3, (1, 2, 3))
element = Element(Tet10, (1,2,3,4,5,6,7,8,9,10))
```

**Why wrong?** Old names mixed topology + basis, causing confusion.

```julia
# DO (modern):
element = Element(Segment, Lagrange{Segment,1}, (1, 2))
element = Element(Triangle, Lagrange{Triangle,1}, (1, 2, 3))
element = Element(Tetrahedron, Lagrange{Tetrahedron,2}, (1,2,3,4,5,6,7,8,9,10))

# OR (legacy):
element = Element(Segment, (1, 2))
element = Element(Triangle, (1, 2, 3))
element = Element(Tetrahedron, (1,2,3,4,5,6,7,8,9,10))
```

### ❌ Pitfall 2: Trying to Mutate Elements

```julia
# DON'T (mutation):
element = Element(Triangle, (1,2,3); fields=(E=210e3,))
element.fields.E = 200e3  # ERROR: fields are immutable!
update!(element, "E", 200e3)  # DEPRECATED
```

**Why wrong?** Elements are immutable for performance.

```julia
# DO (immutable update):
element = Element(Triangle, (1,2,3); fields=(E=210e3,))
element = update(element, :E => 200e3)  # Returns NEW element
```

### ❌ Pitfall 3: Type-Unstable Fields

```julia
# DON'T (Dict - type unstable):
fields = Dict("E" => 210e3, "nu" => 0.3)
element = Element(Triangle, (1,2,3); fields=fields)  # 100× slower!
```

**Why wrong?** Dict loses type information → slow field access.

```julia
# DO (NamedTuple - type stable):
fields = (E = 210e3, ν = 0.3)
element = Element(Triangle, (1,2,3); fields=fields)  # 40-130× faster!
```

### ❌ Pitfall 4: Confusing Topology Order with Basis Order

```julia
# DON'T (confusion):
element = Element(Triangle, Lagrange{Triangle,6}, (1,2,3,4,5,6))
# Order 6? No! Quadratic has order 2, just 6 nodes
```

**Why wrong?** Node count ≠ basis order.

```julia
# DO (correct):
element = Element(Triangle, Lagrange{Triangle,2}, (1,2,3,4,5,6))
# Order 2 (quadratic), happens to have 6 nodes
```

**Node count vs Order:**

- Linear (P1): 3 nodes → Order 1
- Quadratic (P2): 6 nodes → Order 2
- Cubic (P3): 10 nodes → Order 3

---

## Summary

### Key Principles

1. **Two APIs, one architecture** - Modern (explicit) and Legacy (inferred) both supported
2. **Separation of concerns** - Topology, basis, connectivity, fields are independent
3. **Immutability for performance** - 40-130× faster than mutable Dict-based approach
4. **Type stability is critical** - NamedTuple fields, not Dict
5. **No deprecation** - Legacy API will remain supported indefinitely

### Quick Decision Guide

**Use Modern API when:**

- Writing production code
- Performance is critical
- Working with GPUs
- Using non-Lagrange basis functions
- Clarity and explicitness matter

**Use Legacy API when:**

- Prototyping quickly
- Teaching/learning FEM
- Backward compatibility needed
- Using standard Lagrange elements
- Brevity is valuable

### The Bottom Line

```julia
# Both create identical elements internally:
el1 = Element(Triangle, Lagrange{Triangle,1}, (1,2,3); fields=(E=210e3,))  # Modern
el2 = Element(Triangle, (1,2,3); fields=(E=210e3,))  # Legacy

# Both are fully supported ✅
# Both create type-stable, immutable elements ✅
# Both achieve same performance ✅
# Choose based on your needs ✅
```

---

**Questions?** See [Element Architecture](element_architecture.md) for deeper technical details or [TECHNICAL_VISION.md](../../llm/TECHNICAL_VISION.md) for the strategic rationale behind these decisions.
