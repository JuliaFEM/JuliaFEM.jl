# Mesh Module

**Purpose:** Mesh data structures, creation, manipulation, and optimization for finite element analysis.

## Overview

The mesh module provides the core `Mesh{N,T}` data structure and related functionality for representing finite element meshes. It handles node coordinates, element connectivity, named sets (for boundary conditions and material regions), and mesh optimization via graph algorithms.

## Key Concepts

### Type-Stable Parametric Mesh

```julia
Mesh{N, T<:AbstractTopology{N}}
```

The mesh is **parametrically typed** on topology for:
- **Type stability** (10× faster than abstract mesh)
- **GPU optimization** (fixed-size connectivity enables matrix reinterpretation)
- **Industrial workflows** (separate mesh per component in multi-body assemblies)

### Core Data
- **`nodes::Vector{Vec{3,Float64}}`** - Nodal coordinates (always 3D, 2D uses z=0)
- **`connectivity::Vector{NTuple{N,UInt32}}`** - Fixed-size element connectivity tuples
- **`element_sets::Dict{Symbol,Set{UInt32}}`** - Named element groups (e.g., `:body`, `:surface`)
- **`node_sets::Dict{Symbol,Set{UInt32}}`** - Named node groups (e.g., `:fixed`, `:loaded`)
- **`inverse_connectivity::Vector{Vector{Tuple{UInt32,UInt8}}}`** - Node-to-elements map (critical for nodal assembly)

### Advanced Features
- **Bandwidth optimization** via RCM (Reverse Cuthill-McKee) ordering
- **Named nodes/elements** for industrial CAE workflows (multi-part assemblies)
- **Parallel computing support** (node/element coloring, ghost data for MPI)

## Files

### Core Infrastructure

#### `api.jl`
**Purpose:** Abstract types, interfaces, and API definitions

Defines:
- `AbstractMesh` - Base type for all meshes
- `AbstractRefineStrategy` - Base type for refinement strategies
- Interface methods: `nnodes_total()`, `nelements()`, `get_node()`, `connectivity_matrix()`, etc.

**Read this first** to understand the mesh abstraction contract.

#### `mesh.jl`
**Purpose:** Concrete `Mesh{N,T}` implementation

The main mesh data structure with:
- Full field documentation (see struct docstring)
- Inner constructor with validation
- API method implementations
- Helper functions for connectivity matrices, inverse maps, etc.

**This is the workhorse file** - ~850 lines of core mesh functionality.

### Mesh Creation

#### `structured.jl`
**Purpose:** Generate simple structured meshes programmatically

Functions:
- `create_structured_box_mesh(Hex8; xmin, xmax, nx, ...)` - Create regular box mesh
- Automatically creates boundary node sets (`:xmin`, `:xmax`, `:ymin`, etc.)
- Perfect for testing, tutorials, and simple geometries

**Example:**
```julia
# 10×2×2 cantilever beam mesh
mesh = create_structured_box_mesh(Hex8, 
    xmin=0.0, xmax=10.0, nx=10,
    ymin=0.0, ymax=2.0, ny=2,
    zmin=0.0, zmax=2.0, nz=2)
```

#### `circular.jl`
**Purpose:** Generate circular/polar meshes for plates

Functions:
- `create_circular_plate_mesh(Tri3; radius, nr, nθ)` - Polar triangulation for circular plates
- Fan topology from center node - ideal for Kirchhoff plate elements (DKT)
- Creates `:center` and `:outer` node sets for boundary conditions

**Example:**
```julia
# Circular plate with 5 radial rings, 48 sectors
mesh = create_circular_plate_mesh(Tri3; radius=0.5, nr=5, nθ=48)
```

#### `gmsh_wrapper.jl`
**Purpose:** Interface to Gmsh mesh generator

Functions:
- `gmsh_initialize()` - Initialize Gmsh API (consolidated from Gmsh.jl)
- Provides access to `gmsh` module for complex geometries
- Use for industrial CAD-to-mesh workflows

**Note:** Requires `gmsh_jll` package for Gmsh binary.

### Mesh Manipulation

#### `refine.jl`
**Purpose:** Mesh refinement strategies for convergence studies

Strategies:
- `LongestEdgeBisection(levels)` - Adaptive octree-style refinement
  - Analyzes each element to find longest dimension
  - Splits along that direction (preserves geometry aspect ratio)
  - Creates 4 new nodes, 2 new elements per split
  - Element count grows as 2^level

**Example:**
```julia
# Refine 3 times: 1 → 2 → 4 → 8 elements
refined = refine(mesh, LongestEdgeBisection(3))
```

**Use cases:**
- Convergence studies (h-refinement)
- Creating dense meshes from simple coarse definitions
- Mesh sensitivity analysis

See [`README_REFINEMENT.md`](./README_REFINEMENT.md) for detailed documentation.

### Mesh Optimization

#### `graph_ordering.jl`
**Purpose:** Graph algorithms for bandwidth minimization

Functions:
- `symrcm(G, v)` - Sparse Reverse Cuthill-McKee ordering (consolidated from GraphOrdering.jl)
- `bandwidth(G)` - Calculate graph bandwidth
- Returns `GraphOrderingResult` with permutation vectors

**Purpose:** Minimize matrix bandwidth for direct solvers
- Reduces fill-in during factorization (Cholesky, LU)
- Improves cache locality
- Critical for large 3D problems with direct solvers

**Integration:** Mesh stores `node_permutation` and `element_permutation` fields.

## Usage Patterns

### Creating a Mesh

#### From Structured Grid
```julia
using JuliaFEM
using Tensors

# Simple box mesh
mesh = create_structured_box_mesh(Hex8, xmax=1.0, ymax=1.0, zmax=1.0, nx=4, ny=4, nz=4)

# Apply Dirichlet BC to xmin face
fixed_nodes = mesh.node_sets[:xmin]
```

#### From Gmsh
```julia
using JuliaFEM

gmsh_initialize()
# ... gmsh commands to create geometry and mesh ...
mesh = import_gmsh_mesh()  # TODO: implement import function
gmsh.finalize()
```

#### Manual Construction
```julia
# 4-node tetrahedron
nodes = [Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.0), 
         Vec(0.0, 1.0, 0.0), Vec(0.0, 0.0, 1.0)]
connectivity = [(UInt32(1), UInt32(2), UInt32(3), UInt32(4))]

mesh = Mesh{4, Tet4}(nodes, connectivity)
```

### Accessing Mesh Data

```julia
# Basic queries
n_nodes = nnodes_total(mesh)
n_elems = nelements(mesh)

# Node coordinates
X = get_node(mesh, 42)  # Vec{3}(x, y, z)

# Connectivity
conn = connectivity_matrix(mesh)  # Matrix{Int} - zero-copy view for GPU
elem_nodes = conn[elem_id, :]  # Node IDs for element

# Inverse connectivity (node → elements, critical for nodal assembly!)
elems_containing_node = get_elements_for_node(mesh, node_id)

# Named sets
fixed_nodes = get_node_set(mesh, :fixed)
body_elements = get_element_set(mesh, :body)
```

### Mesh Refinement

```julia
# Convergence study
coarse_mesh = create_structured_box_mesh(Hex8, nx=2, ny=2, nz=2)

for level in 0:4
    if level == 0
        mesh = coarse_mesh
    else
        mesh = refine(coarse_mesh, LongestEdgeBisection(level))
    end
    
    println("Level $level: $(nelements(mesh)) elements")
    # Run FEM analysis...
end
```

## Design Philosophy

### Separation of Concerns
- **Mesh owns topology** (nodes, connectivity, sets)
- **Physics references mesh** (does not own it)
- **Multiple physics can share one mesh** (multiphysics coupling)

### Type Stability for Performance
- `Mesh{8, Hex8}` is fully concrete (no abstract fields)
- Fixed-size `NTuple{8,UInt32}` connectivity (not `Vector{Int}`)
- Enables 10× performance gains vs abstract mesh types

### Industrial Workflows
- Named node/element IDs support multi-part assemblies
  - Part 1: node IDs 10,000,001 → 10,050,000
  - Part 2: node IDs 20,000,001 → 20,030,000
- Element/node sets for boundary conditions and material regions
- Multi-body assembly via separate meshes per component

### GPU and Parallel Computing
- `connectivity_matrix()` provides zero-copy `Matrix{Int}` for GPU transfer
- Node/element coloring for thread-safe assembly
- Ghost nodes/elements for MPI domain decomposition
- `inverse_connectivity` enables efficient nodal assembly (see `docs/book/multigpu_nodal_assembly.md`)

## Future Extensions

Planned features:
- **Mixed-topology meshes** (`MixedMesh` type)
- **Gmsh import function** (read `.msh` files)
- **Abaqus/Code Aster readers** (consolidate from vendor packages)
- **More refinement strategies** (uniform octree, red-green, adaptive)
- **Surface extraction** (generate boundary mesh for visualization/BCs)
- **Mesh quality metrics** (Jacobian determinant, aspect ratio)
- **RCM integration** (automatic bandwidth minimization)

## Testing

Run tests:
```bash
julia --project=. test/mesh/runtests.jl
```

Key test files:
- `test/mesh/test_structured.jl` - Structured mesh generation
- `test/mesh/test_refine.jl` - Refinement algorithms
- `test/mesh/test_connectivity.jl` - Connectivity and inverse maps

## Examples

See:
- `examples/structured_mesh_demo.jl` - Creating and using structured meshes
- `examples/mesh_refinement_demo.jl` - Convergence study with refinement
- `examples/cantilever_minimal_new_api.jl` - Complete FEM example with mesh

## Dependencies

- `Tensors.jl` - For `Vec{3,Float64}` node coordinates
- `gmsh_jll` (optional) - For Gmsh mesh generation

## Related Modules

- **`src/topology/`** - Element topology (reference coordinates, edges, faces)
- **`src/basis/`** - Shape functions (interpolation on mesh)
- **`src/assemblers/`** - Uses `inverse_connectivity` for nodal assembly
- **`src/domains/`** - Physics modules that operate on meshes

## Notes

- **Always 3D coordinates:** Even 2D problems use `Vec{3}` with z=0 (simplifies code)
- **UInt32 for indices:** Saves memory vs Int64, supports 4 billion nodes
- **Memory profiling artifacts:** `*.mem` files are git-ignored (profiler output)

## References

- Reverse Cuthill-McKee: [SIAM J. Numer. Anal. 13, 865 (1976)]
- Octree refinement: Standard technique in adaptive mesh refinement (AMR)
- Nodal assembly architecture: `docs/book/multigpu_nodal_assembly.md`

---

**Maintainer:** JuliaFEM Team  
**Last Updated:** November 21, 2025

