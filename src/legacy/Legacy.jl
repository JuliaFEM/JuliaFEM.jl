# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
    JuliaFEM.Legacy

Older pre-reset API surface. Loaded only when the environment variable
`JULIAFEM_ENABLE_LEGACY=1` is set at module-load time. Contains the
Dict-based field system (`DCTI`, `DVTI`, ...), the `Problem` /
`Assembly` / `Solver` / `Analysis` hierarchy, the `Element(Poi1, ...)`
constructor family, the Abaqus mesh-reader stack, and the various
`problems_*` files.

None of this is needed by the current default **0.x** code (`@DOFSet`,
`Element{K,P,S,N}`, `DOFHandler`, `DOFBasedCOOAssembler`, the matrix-free
operators, etc.). It is preserved here so that downstream users who
still depend on the old API can opt back in via the env variable, and
so that the contact/mortar reference implementations remain accessible
during the migration.

# Design notes

The legacy files reference each other with late-bound dispatch
(`update!`, `interpolate`, `field`, `Problem`, ...). They are therefore
included as a single block, in dependency order, after the new core
types are available in the parent `JuliaFEM` module.

To make sub-includes resolve relative to this directory, no path
prefix is needed; `include(\"foo.jl\")` from inside `Legacy.jl` resolves
to `src/legacy/foo.jl`.
"""
module Legacy

using ..JuliaFEM
# Names the legacy files reference at top level. `Mesh` and
# `AbstractMaterial` are deliberately NOT imported because the Abaqus
# parser defines its own (Dict-based) `Mesh` and `AbstractMaterial`
# types inside this submodule and would otherwise collide with the
# JuliaFEM bindings.
using ..JuliaFEM: AbstractField, AbstractElement, Element, AbstractMesh
using ..JuliaFEM: AbstractBasis, AbstractTopology
using ..JuliaFEM: ContinuumFormulation, FullThreeD, AbstractContinuumTheory
using ..JuliaFEM: AbstractFormulation
using ..JuliaFEM: dofs_per_node, get_field
using ..JuliaFEM: Seg2, Seg3, Tri3, Tri6, Tri7, Quad4, Quad8, Quad9
using ..JuliaFEM: Tet4, Tet10, Pyr5, Wedge6, Wedge15, Hex8, Hex20, Hex27
# Reuse the parent module's no-op `@timeit` stub instead of redefining it
# locally; the two used to drift independently.
using ..JuliaFEM: @timeit
import Base: getindex, setindex!, length, size, haskey, convert, ==

using SparseArrays
using LinearAlgebra
using Logging

# CSC-only sparse helpers used by the legacy linear-system / solver path.
# Not part of the modern public API; loaded with the rest of the legacy
# stack so downstream callers that opted in to `JULIAFEM_ENABLE_LEGACY=1`
# keep working.
include("sparse_helpers.jl")

# Legacy Dict-based field system: DCTI, DVTI, DCTV, DVTV, CCTI, CVTI, CCTV, CVTV, ...
include("dcti_dvti_fields.jl")

# Legacy element constructors using AbstractBasis{0} (Poi1) and Dict-based fields.
include("elements_lagrange.jl")

# LinearSystem (sparse-matrix + lambda + ...) used by `Solver`/`Analysis`.
include("linear_system.jl")

# Legacy core types (Node, IP, IntegrationPoint, Point).
include("core_types.jl")

# Legacy Problem hierarchy (FieldProblem / BoundaryProblem / Problem).
include("assembly_problems.jl")

# Analysis (Analysis, AbstractAnalysis, AbstractResultsWriter).
include("analysis.jl")

# Deprecated FEMBase methods (length / size / setindex! / update! warnings).
include("deprecated_fembase.jl")

# Boundary-condition problem (Dirichlet).
include("problems_dirichlet.jl")

# Solver hierarchy (Solver = Analysis, Linear, Nonlinear).
include("solvers.jl")

# Mesh I/O stack (Abaqus). `parse_mesh.jl` provides the working
# `abaqus_read_mesh`; `parse_model.jl` provides `abaqus_read_model`;
# `create_surface_elements.jl` is shared infrastructure used by the
# Dict-based mesh. All depend on the legacy `Element(Poi1, ...)`
# constructor and the Dict-based fields above. Aster `.med` reading
# requires HDF5 and is intentionally not wired in here.
#
# Loaded before `deprecations.jl` so the `Abaqus` sub-module's
# `import ..create_surface_elements` finds its target.
include("io/keyword_register.jl")
include("io/parse_mesh.jl")
include("io/parse_model.jl")
include("io/create_surface_elements.jl")
include("io/abaqus_download.jl")

# Tiny `assemble!` shims plus the `Abaqus` sub-module helper.
include("deprecations.jl")

end # module Legacy
