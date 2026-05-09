# test/materials/

Tests for the material trait system and the material/assembly cache
machinery used by the DOF-based assembler.

## What is here

State-variable and trait infrastructure:

- `test_state_variables.jl`
- `test_traits.jl`

Material cache and assembly workspace, including the zero-allocation
contract that the cantilever bench locks in:

- `test_global_material_cache.jl`
- `test_assembly_workspace_refactor.jl`
- `test_cantilever_material_cache_zero_allocations.jl`

End-to-end material-driven assembly:

- `test_plasticity_integration.jl`

All files in this directory are wired into `test/runtests.jl` and run
on every `Pkg.test()`.

## Related

- Source: `src/materials/`.
- Concrete model unit tests (`test_linear_elastic`, `test_neo_hookean`,
  `test_perfect_plasticity`, `test_finite_strain_plasticity`) and the
  `_new_api.jl` constitutive specifications were moved to
  `llm/design/legacy-tests/materials/` because they target removed APIs.
  The materials roadmap lives at `llm/design/materials-tdd/`.
