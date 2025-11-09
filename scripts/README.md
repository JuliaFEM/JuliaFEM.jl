# JuliaFEM Scripts

This directory contains development and code generation scripts for JuliaFEM.

## Basis Function Generation

### `generate_lagrange_basis.jl`

**Purpose:** Pre-generate all Lagrange basis functions for standard finite elements.

**Why Pre-generate?**

- **Fast loading:** No symbolic math at package load time (100+ ms → 0 ms)
- **Full precompilation:** Remove `__precompile__(false)` restriction
- **Readable code:** Generated code is easy to debug and understand
- **Version control:** Changes to mathematics show up in git diffs
- **Reproducible:** Same input always produces same output

**When to Run:**

- Adding new element types (Seg2, Tri3, Hex20, etc.)
- Fixing bugs in generation logic
- Changing polynomial ansatz strategy
- After modifying `src/basis/lagrange_generator.jl`

**Usage:**

```bash
cd /path/to/JuliaFEM.jl
julia --project=. scripts/generate_lagrange_basis.jl
```

**Output:**

- `src/basis/lagrange_generated.jl` (commit this file!)

**Theory:**
See `docs/book/lagrange_basis_functions.md` for mathematical foundation.

**Architecture:**

```text
src/basis/lagrange_generator.jl
    │
    │ (symbolic engine - uses symbolic differentiation)
    │
    ↓
scripts/generate_lagrange_basis.jl
    │
    │ (orchestration - defines all element types)
    │
    ↓
src/basis/lagrange_generated.jl
    │
    │ (clean Julia code - no eval, fully precompilable)
    │
    ↓
src/JuliaFEM.jl includes generated file
```

**Generated Elements:**

| Dimension | Linear | Quadratic | Higher |
|-----------|--------|-----------|--------|
| 1D        | Seg2   | Seg3      | -      |
| 2D Tri    | Tri3   | Tri6      | -      |
| 2D Quad   | Quad4  | Quad8, Quad9 | -   |
| 3D Tet    | Tet4   | Tet10     | -      |
| 3D Hex    | Hex8   | Hex20, Hex27 | -   |
| 3D Pyramid| Pyr5   | -         | -      |
| 3D Wedge  | Wedge6 | Wedge15   | -      |

**Total:** 15 element types covering all standard Lagrange families.

**Performance Impact:**

- **Before:** 150+ ms at package load (symbolic math for each element)
- **After:** < 1 ms (just include pre-generated file)
- **Speedup:** ~150× faster package loading

**Workflow:**

1. Edit element catalog in `scripts/generate_lagrange_basis.jl`
2. Run generation script
3. Review `src/basis/lagrange_generated.jl`
4. Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
5. Commit both files: `git add scripts/ src/basis/lagrange_generated.jl`

**Example**: Adding Hex64 (Triquartic)

```julia
# In scripts/generate_lagrange_basis.jl, add to element catalog:
push!(elements, (
    name = "Hex64",
    description = "64-node triquartic hexahedral element",
    coordinates = [
        # ... 64 nodes (corners + edges + faces + volume)
    ],
    ansatz = [
        :(1), :(ξ), :(η), :(ζ),  # ... up to ξ³η³ζ³
    ]
))
```

Then regenerate:

```bash
julia --project=. scripts/generate_lagrange_basis.jl
```

The new `Hex64` type will be automatically available in JuliaFEM!

---

## Future Scripts (Planned)

### `benchmark_suite.jl`

Run comprehensive performance benchmarks.

### `validate_against_reference.jl`

Compare JuliaFEM results to Code Aster/ABAQUS.

### `generate_element_matrices.jl`

Pre-compute stiffness matrices for simple elements.

---

**See also:**

- `docs/book/lagrange_basis_functions.md` - Mathematical theory
- `src/basis/lagrange_generator.jl` - Symbolic generation engine
- `llm/VISION_2.0.md` - Overall project architecture
