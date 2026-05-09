# Source of truth for the package Documenter home page "minimal elasticity" demo.
# The README.md block under "A modern minimal example" must match this assembly
# pipeline; `test/docs/runtests.jl` runs both this file and the README fence.
# Executed by `scripts/verify_docs_quickstart.jl`, `test/docs/runtests.jl`, and
# (via Documenter) shown on `docs/src/index.md` with `@literalinclude`.
using JuliaFEM
using SparseArrays

function minimal_elasticity_quickstart()
    mesh = create_structured_box_mesh(Hex8;
        xmin = 0.0, xmax = 1.0, nx = 4,
        ymin = 0.0, ymax = 1.0, ny = 4,
        zmin = 0.0, zmax = 1.0, nz = 4,
    )
    S  = @DOFSet{u::DOF{Displacement{3}, Vertex}}
    ET = Element{Hex8, Lagrange{1}, S}
    elements, handler = create_elements!(mesh, ET)

    material = LinearElastic(E = 210e9, ν = 0.3)
    kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                               material, Displacement{3}())

    asm   = DOFBasedCOOAssembler()
    cache = create_cache(asm, elements, handler, mesh, kernel)
    assemble!(cache, asm, kernel, mesh)
    K, f  = extract_system(cache)

    nd = size(K, 1)
    @assert nd > 0
    @assert size(K, 2) == nd
    @assert nnz(K) > 0
    @assert length(f) == nd
    return (; ndofs = nd, nnz_stiffness = nnz(K))
end
