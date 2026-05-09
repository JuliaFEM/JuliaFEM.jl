# src/assemblers/caches/

Pre-allocated per-element scratch and global COO/CSC caches reused by
every assembly strategy. Keeping them in one place makes it easy to see
what is allocated up-front and what is consumed by the hot loops.

## Files

- `geometry_cache.jl` — `GeometryCache`. Per-element shape values, derivatives, and Jacobian data sized for one quadrature rule. Parametric on storage so the same struct works heap-owned (element-based assemblers) or as a view into batched SoA storage (DOF-based assembler).
- `element_cache.jl` — `ElementCache`. Assembly-time per-element scratch (local stiffness, force, dof index buffers).
- `material_cache.jl` — `AssemblyMaterialWorkspace` plus the `create_material_cache` helper. Holds the per-IP material state buffers (stress, tangent, history) used by the kernels.
- `coo_cache.jl` — `COOCache`, the global I/J/V triplet workspace used by the element-based assembler.

The hot-path invariant for all of these is zero allocation per element
on a steady-state assemble. The regression test for that is
`test/assemblers/test_dof_based_zero_alloc.jl`.
