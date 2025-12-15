# JuliaFEM Source Code Architecture

**Purpose:** This document defines the role and responsibility of each subdirectory in `src/`. Every directory must have a clear, single reason to exist.

---

## 🎯 Core Architecture Principle

**Clean separation of concerns:** Each module has ONE job. If a directory's purpose isn't clear, it should be merged or deleted.

---

## ✅ Active Modules (Clear Purpose, Keep)

### `topology/` - Element Geometry
**Purpose:** Reference element definitions (node positions, edges, faces)

**Responsibility:**
- Define topology types: `Segment`, `Triangle`, `Quadrilateral`, `Tetrahedron`, `Hexahedron`, `Pyramid`, `Wedge`
- Provide reference coordinates in parent space
- Define connectivity (edges, faces)
- **Zero-allocation design** (tuples, immutable)

**Files:** 9 files  
**Exports:** Topology types + aliases (Tri3, Quad4, Hex8, etc.)  
**Status:** ✅ Modern, complete (Nov 9, 2025)

**Key insight:** Topology ≠ Basis. `Triangle` is geometry. Node count comes from basis (Lagrange{Triangle,1} = 3 nodes, Lagrange{Triangle,2} = 6 nodes).

---

### `basis/` - Shape Functions
**Purpose:** Interpolation schemes for finite element approximation

**Responsibility:**
- Evaluate basis functions N(ξ,η,ζ) at reference points
- Evaluate basis derivatives ∂N/∂ξ, ∂N/∂η, ∂N/∂ζ
- Support Lagrange and Serendipity families
- Generate basis code from symbolic definitions

**Files:** 12 files (including generator)  
**Exports:** `AbstractBasis`, `Lagrange{T,P}`, `Serendipity{T,P}`, `eval_basis!`, `eval_dbasis!`  
**Status:** ✅ Modern, parametric types (Nov 20, 2025)

**Design:** `basis_generator.jl` creates `basis_generated.jl` with all Lagrange families. Run: `julia --project=. src/basis/basis_generator.jl`

---

### `quadrature/` - Numerical Integration
**Purpose:** Gauss quadrature rules for integration

**Responsibility:**
- Provide integration points (ξ, η, ζ, weight) for all topologies
- Support multiple orders (1-5 typically)
- Map high-level schemes (`Gauss{3}`) to low-level point tables
- **Zero-allocation API** (returns tuples)

**Files:** 10 files  
**Exports:** `Gauss{order}`, `integration_points(scheme, topology)`, `IntegrationPoint`  
**Status:** ✅ Modern API (Nov 20, 2025)

**Architecture:** `api.jl` defines interface, `gauss.jl` maps schemes, `gl_*.jl` contain point data.

---

### `mesh/` - Mesh Data Structures
**Purpose:** Mesh representation (nodes, connectivity, sets)

**Responsibility:**
- Define `Mesh{N,T}` parametric type (type-stable)
- Store node coordinates, element connectivity
- Manage element/node sets (for BCs and material regions)
- Provide inverse connectivity (node → elements) for nodal assembly
- Mesh creation (structured, circular, Gmsh interface)
- Mesh refinement (convergence studies)
- Graph ordering (RCM bandwidth minimization)

**Files:** 7 files  
**Exports:** `Mesh`, `AbstractMesh`, `create_structured_box_mesh`, `refine`, `LongestEdgeBisection`  
**Status:** ✅ Complete, documented (Nov 21, 2025)  
**README:** ✅ `mesh/README.md` (complete architecture doc)

---

### `materials/` - Constitutive Models
**Purpose:** Material behavior (stress-strain relationships)

**Responsibility:**
- Define material types: `LinearElastic`, `PerfectPlasticity`, `FiniteStrainPlasticity`
- Compute stress σ from strain ε
- Compute tangent stiffness 𝔻 (∂σ/∂ε)
- Manage internal state variables (plasticity, damage)
- Trait-based dispatch for different material behaviors

**Files:** 6 files  
**Exports:** `AbstractMaterial`, `LinearElastic`, `PerfectPlasticity`, `compute_stress`, `compute_tangent`  
**Status:** ✅ Modern trait-based API (Nov 19, 2025)

**Design:** Three material behavior traits:
- `StatelessConstantTangent` (linear elastic)
- `StatelessStrainDependent` (hyperelastic)
- `StatefulStrainDependent` (plasticity, damage)

---

### `assemblers/` - Assembly Strategies
**Purpose:** Build global matrices/vectors from element contributions

**Responsibility:**
- Provide assembly caches (COO, CSC, nodal)
- Scatter element blocks to global system
- Support symmetric and non-symmetric matrices
- Element-based and node-based assembly
- **Zero-allocation design** with pre-allocated caches

**Files:** 20 files  
**Exports:** `COOCache`, `CSCCache`, `NodalCache`, `assemble!`, `ElementBasedCOOAssembler`, `NodeBasedCOOAssembler`  
**Status:** ✅ Refactored into modular caches (Nov 20, 2025)

**Architecture:** Split from monolithic `caches.jl` into:
- `coo_cache.jl`, `csc_cache.jl`, `nodal_cache.jl` - Cache types
- `element_cache.jl`, `geometry_cache.jl`, `material_cache.jl` - Domain-specific caches
- `scatter_*.jl` - Scatter strategies (symmetric, direct, force vector)

---

### `domains/` - Physics Kernels
**Purpose:** Problem-specific formulations (continuum, beams, plates, shells, trusses)

**Responsibility:**
- Implement weak form integrals for different physics
- Provide kernel types that encapsulate formulation + material
- Handle domain-specific DOF mappings (displacement, rotation, etc.)
- Compute element stiffness and force contributions

**Subdirectories:**
- `continuum/` - 3D solid mechanics (displacement-based)
- `beams/` - 1D beam elements (Euler-Bernoulli, Timoshenko)
- `plates/` - 2D plate bending (Kirchhoff, Reissner-Mindlin, DKT)
- `shells/` - 2D shell elements (membrane + bending)
- `trusses/` - 1D truss elements (axial only)
- `common/` - Shared utilities

**Files:** 16 files across 6 subdirectories  
**Exports:** `ContinuumKernel`, `PlateKernel`, `BeamKernel`, etc.  
**Status:** ✅ Continuum complete with two-phase assembly (Nov 19, 2025)

**Key innovation:** Two-phase assembly architecture (Phase 1: material state, Phase 2: assembly) enables 4-64× performance improvement.

---

### `solvers/` - Linear System Solvers
**Purpose:** Solve Ku = f (direct, iterative, matrix-free)

**Responsibility:**
- Direct solvers (Cholesky, LU)
- Iterative solvers (CG, GMRES)
- Matrix-free operators (for contact, large deformations)
- Nonlinear solvers (Newton-Raphson, line search)

**Files:** 1 file (needs expansion)  
**Exports:** TBD  
**Status:** 🔄 Minimal (needs Krylov.jl integration)

**Roadmap:** Implement Newton-Krylov with GMRES per golden standard (`docs/book/multigpu_nodal_assembly.md`).

---

### `physics/` - High-Level Problem Definition
**Purpose:** User-facing problem setup (boundary conditions, loads, constraints)

**Responsibility:**
- Define `Physics` type (wraps kernel + mesh + BCs)
- Dirichlet boundary conditions (prescribed displacement)
- Neumann boundary conditions (surface tractions)
- Constraint equations (MPC, contact)
- Coordinate problem assembly and solve

**Files:** 7 files  
**Exports:** `Physics`, `DirichletBC`, `NeumannBC`, `apply_bc!`  
**Status:** ✅ Interface defined, implementations partial

---

### `fields/` - Field Storage on Elements
**Purpose:** Store and interpolate field variables (displacement, temperature, etc.)

**Responsibility:**
- Field definition and evaluation
- Interpolation at arbitrary points
- Time-dependent fields
- Field arithmetic

**Files:** 2 files  
**Exports:** `Field`, `eval_field`  
**Status:** ⚠️ Needs redesign (see `llm/FIELDS_DESIGN.md`)

**Blocker:** Old Dict-based design causes type instability. Must be resolved for v1.0.

---

### `elements/` - Legacy Element Interface
**Purpose:** Old element abstraction (Element type with fields)

**Responsibility:**
- Element struct with topology + connectivity + fields
- `update()` function (immutable API)
- Integration with old FEMBase API

**Files:** 3 files  
**Exports:** `Element`, `update`  
**Status:** ⚠️ Legacy compatibility, dual API (modern + old)

**Decision:** Keep for backward compatibility but encourage direct mesh/kernel usage.

---

### `geometry/` - Geometric Calculations
**Purpose:** Jacobian, coordinate transformations, strain computation

**Responsibility:**
- Compute Jacobian matrix J = ∂x/∂ξ
- Physical derivatives ∂N/∂x = J⁻¹ ∂N/∂ξ
- Strain tensor from displacement gradient
- Deformation gradient F

**Files:** 2 files  
**Exports:** `compute_jacobian`, `physical_derivatives`, `compute_strain`  
**Status:** ✅ Core functionality complete

---

### `backend/` - Execution Backend (CPU/GPU)
**Purpose:** Abstract backend for CPU vs GPU execution

**Responsibility:**
- Backend abstraction (`CPUBackend`, `CUDABackend`)
- Array type selection (Array vs CuArray)
- Backend-specific optimizations

**Files:** 2 files  
**Exports:** `AbstractBackend`, `CPUBackend`  
**Status:** ✅ Abstract interface defined, CPU complete, GPU planned

---

### `io/` - Mesh Import/Export
**Purpose:** Read/write mesh files (Abaqus, Code Aster, Gmsh)

**Responsibility:**
- Read Abaqus `.inp` files
- Read Code Aster `.med` files
- Read Gmsh `.msh` files
- Write VTK for visualization (planned)

**Files:** 4 files  
**Exports:** `abaqus_read_mesh`, `aster_read_mesh`  
**Status:** ✅ Basic readers implemented

---

### `sparse/` - Sparse Matrix Utilities
**Purpose:** Sparse matrix data structures and operations

**Responsibility:**
- Sparse matrix CSC format
- Dictionary-of-keys (DOK) format
- Sparse vector utilities
- Conversion between formats

**Files:** 3 files  
**Exports:** `SparseDOK`, sparse utilities  
**Status:** ✅ Utility module, may be replaceable by SparseArrays.jl

---

## 🔄 Transition/Legacy Modules (Needs Decision)

### `legacy/` - Old API Compatibility
**Purpose:** Deprecated FEMBase API for backward compatibility

**Responsibility:**
- Old `Problem` type (deprecated)
- Old `update!` mutable API (replaced by `update()`)
- Old assembly system (replaced by assemblers/)
- Deprecation warnings

**Files:** 18 files  
**Status:** ⚠️ Keep for compatibility, remove in v2.0

**Decision:** Maintain for existing code but discourage use. Document migration path.

---

### `readers/` - Legacy Mesh Readers
**Purpose:** Old mesh reading infrastructure from vendor packages

**Responsibility:**
- Duplicate of `io/` functionality
- More complex reader infrastructure
- Keyword parsing system

**Files:** 9 files  
**Status:** ⚠️ **DUPLICATES `io/`** - Should consolidate

**Action:** Merge best code into `io/`, delete `readers/`.

---

### `assembly/` - Old Assembly System
**Purpose:** Pre-refactor assembly code (element-based)

**Responsibility:**
- Element assembly (replaced by nodal assembly in `assemblers/`)
- Old problem-based API
- Framework code (superseded)

**Files:** 7 files  
**Status:** ❌ **NOT INCLUDED IN MODULE** - Dead code

**Evidence:** Not in `src/JuliaFEM.jl` includes, not in test suite.

**Action:** DELETE after confirming `domains/` and `assemblers/` supersede functionality.

---

### `prototype/` - Experimental Physics API
**Purpose:** Testing new physics abstraction designs

**Files:**
- `physics_modern.jl`
- `physics_modern_v2.jl`

**Status:** ❌ **EXPERIMENTAL CODE** - Not in module

**Action:** If superseded by `domains/` → DELETE. If still relevant → move to `llm/archive/prototypes/`.

---

## 🗑️ Empty/Dead Directories (Delete)

### `problems/`
**Files:** 0  
**Status:** ❌ Empty directory  
**Action:** DELETE (old Problem API is in `legacy/`)

---

### `utils/`
**Files:** 0  
**Status:** ❌ Empty directory  
**Action:** DELETE

---

## 🔀 Duplicate Functionality (Needs Consolidation)

### `plates/` vs `domains/plates/`
**Current state:**
- `plates/` - 3 files (api.jl, dkt.jl, test_dkt.jl)
- `domains/plates/` - Integrated domain

**Status:** ⚠️ **DUPLICATION**

**Investigation needed:**
1. Which has more recent code?
2. Is `plates/` superseded by `domains/plates/`?
3. If yes → delete `plates/`, if no → merge into `domains/plates/`

---

### `io/` vs `readers/`
**Current state:**
- `io/` - 4 files (simple readers)
- `readers/` - 9 files (complex reader infrastructure)

**Status:** ⚠️ **DUPLICATION**

**Action:** 
1. Compare functionality
2. Keep best implementation
3. Merge into single `io/` directory
4. Delete `readers/`

---

## 📋 Module Dependency Order

Understanding the include order in `src/JuliaFEM.jl`:

```
1. topology/          # Geometry (no dependencies)
2. quadrature/        # Integration points (needs topology)
3. basis/             # Shape functions (needs topology)
4. geometry/          # Jacobian (needs basis)
5. mesh/              # Mesh structure (needs topology)
6. materials/         # Constitutive models (standalone)
7. fields/            # Field storage (needs elements)
8. assemblers/        # Assembly (needs geometry, mesh)
9. domains/           # Physics kernels (needs materials, geometry, assemblers)
10. solvers/          # Linear solvers (needs assemblers)
11. physics/          # User API (needs everything)
12. backend/          # Execution backend (orthogonal)
13. io/               # Mesh I/O (needs mesh)
```

---

## 🎯 Cleanup Action Plan

### Phase 1: Delete Dead Code (Immediate)

```bash
# Empty directories
rm -rf src/problems/ src/utils/

# Dead experimental code (after verification)
rm -rf src/assembly/      # Verify superseded by assemblers/
rm -rf src/prototype/     # Move to llm/archive if needed
```

### Phase 2: Consolidate Duplicates (This Week)

**Task 1: io/ vs readers/**
1. Compare implementations
2. Merge best code into `io/`
3. Delete `readers/`
4. Update imports

**Task 2: plates/ vs domains/plates/**
1. Determine canonical version
2. Merge if needed
3. Delete duplicate
4. Update includes in `JuliaFEM.jl`

### Phase 3: Document Each Module (Ongoing)

Create `README.md` in each directory explaining:
- Purpose (one sentence)
- Key files and their roles
- Exports
- Usage examples
- Dependencies

**Priority order:**
1. ✅ `mesh/README.md` (done)
2. `topology/README.md`
3. `basis/README.md`
4. `quadrature/README.md`
5. `domains/README.md` (umbrella doc)
6. `assemblers/README.md`
7. `materials/README.md`

---

## 📊 Directory Health Status

| Directory | Status | Files | Action |
|-----------|--------|-------|--------|
| `topology/` | ✅ Excellent | 9 | Document |
| `basis/` | ✅ Good | 12 | Document |
| `quadrature/` | ✅ Excellent | 10 | Document |
| `mesh/` | ✅ Excellent | 7 | ✅ Documented |
| `materials/` | ✅ Good | 6 | Document |
| `assemblers/` | ✅ Good | 20 | Document |
| `domains/` | ✅ Good | 16 | Document |
| `solvers/` | 🔄 Minimal | 1 | Expand |
| `physics/` | 🔄 Partial | 7 | Complete |
| `fields/` | ⚠️ Needs redesign | 2 | Redesign (blocker) |
| `elements/` | ⚠️ Legacy | 3 | Keep for compat |
| `geometry/` | ✅ Good | 2 | Document |
| `backend/` | ✅ Good | 2 | Expand for GPU |
| `io/` | ✅ Good | 4 | Consolidate readers/ |
| `sparse/` | ✅ Utility | 3 | Review necessity |
| `legacy/` | ⚠️ Deprecated | 18 | Keep until v2.0 |
| `readers/` | ❌ Duplicate | 9 | **DELETE** (merge to io/) |
| `assembly/` | ❌ Dead | 7 | **DELETE** |
| `prototype/` | ❌ Experimental | 2 | **DELETE** (archive) |
| `plates/` | ⚠️ Duplicate? | 3 | **CONSOLIDATE** |
| `beams/` | 🔄 Stub | 1 | In `domains/beams/` |
| `shells/` | 🔄 Stub | 1 | In `domains/shells/` |
| `trusses/` | ⚠️ Duplicate? | 3 | In `domains/trusses/` |
| `problems/` | ❌ Empty | 0 | **DELETE** |
| `utils/` | ❌ Empty | 0 | **DELETE** |
| `formulations/` | 🔄 Stub | 1 | Merge to domains? |

---

## 🏆 Design Principles

1. **Single Responsibility:** Each directory has ONE clear purpose
2. **No Duplication:** Merge or delete duplicates
3. **Documentation:** Every directory has README.md
4. **Dependency Order:** Lower-level modules have fewer dependencies
5. **Type Stability:** Modern API throughout (no Dicts in hot paths)
6. **Zero Allocation:** Pre-allocated caches, tuple returns
7. **Testing:** Every module has corresponding `test/` subdirectory

---

## 🔮 Future Architecture (v2.0)

**Clean module structure:**
```
src/
├── topology/       # Element geometry
├── basis/          # Shape functions
├── quadrature/     # Integration
├── mesh/           # Mesh structures
├── materials/      # Constitutive models
├── geometry/       # Jacobian, strain
├── assemblers/     # Assembly strategies
├── domains/        # Physics kernels
│   ├── continuum/
│   ├── beams/
│   ├── plates/
│   ├── shells/
│   └── trusses/
├── solvers/        # Linear/nonlinear solvers
├── physics/        # User API (BCs, problems)
├── backend/        # CPU/GPU execution
└── io/             # Mesh I/O
```

**Removed:**
- `legacy/` (deleted in v2.0)
- `elements/` (merged into mesh/domains)
- `fields/` (redesigned and merged)
- All duplicate directories

---

**Maintainer:** JuliaFEM Team  
**Last Updated:** November 21, 2025  
**Next Review:** After consolidation phase (December 2025)

