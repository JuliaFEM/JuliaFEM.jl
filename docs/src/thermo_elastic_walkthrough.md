# [Thermo-elasticity: multi-field walkthrough](@id thermo_elastic_walkthrough)

This page is a **narrative companion** to
[`test/domains/thermo_elastic/test_thermo_elastic_kernel.jl`](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/test/domains/thermo_elastic/test_thermo_elastic_kernel.jl),
which is the executable contract for the coupled `u` + `T` path through the
DOF-based assembler.

## What you assemble

`ThermoElasticKernel` couples vertex displacement `u` (3 DOFs per node) with
vertex temperature `T` (1 DOF per node). The global stiffness has blocks
`K_uu`, `K_TT`, and (when the coupling coefficient `β ≠ 0`) off-diagonal
`K_uT` / `K_Tu`. With `β = 0` the kernel is block-diagonal and matches
independent elasticity and heat solves on the same mesh.

## Minimal code (block-diagonal smoke)

The following uses the same material stack as the regression test, a small
`Hex8` box, and `DOFBasedCOOCache` (the multi-field kernel is wired through this
cache type rather than the narrow `create_cache` overload used for
`ContinuumKernel` alone).

```julia
using JuliaFEM
using SparseArrays

mesh = create_structured_box_mesh(Hex8;
    xmin = 0.0, xmax = 1.0, nx = 2,
    ymin = 0.0, ymax = 1.0, ny = 2,
    zmin = 0.0, zmax = 1.0, nz = 2,
)

mech  = LinearElastic(E = 210e9, ν = 0.3)
therm = HeatConductivity(k = 50.2)
β     = 0.0
kernel = ThermoElasticKernel(ContinuumFormulation{FullThreeD}(), mech, therm, β)

S = @DOFSet{u::DOF{Displacement{3}, Vertex},
            T::DOF{Temperature, Vertex}}
elements, handler = create_elements!(mesh, Element{Hex8, Lagrange{1}, S})

asm   = DOFBasedCOOAssembler()
cache = DOFBasedCOOCache(elements, handler, mesh, kernel)
assemble!(cache, asm, kernel, mesh)
K, f = extract_system(cache)

@assert size(K, 1) == size(K, 2) == 108
@assert nnz(K) == 5488
```

## Where to read next

- Kernel source and coupling equations: `src/domains/thermo_elastic/kernel.jl`.
- Compile-time DOF layout: `local_dof_layout(Element{Hex8, Lagrange{1}, S})`.
- Matrix-free parity with `K * x`: same test file as above.
