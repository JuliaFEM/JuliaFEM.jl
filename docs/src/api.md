# API Reference

This page lists the public API of `JuliaFEM`. Symbols are grouped by
topic; the underlying source lives in `src/<topic>/`. The reference is
generated from docstrings and only documents symbols that are exported
and have a docstring attached.

```@meta
DocTestSetup = quote
    using JuliaFEM
end
CurrentModule = JuliaFEM
```

## Topology

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "topology/api.jl",
    "topology/segments.jl",
    "topology/triangles.jl",
    "topology/quadrilaterals.jl",
    "topology/tetrahedra.jl",
    "topology/hexahedra.jl",
    "topology/pyramids.jl",
    "topology/wedges.jl",
]
Order   = [:type, :function]
```

## Basis functions

```@autodocs
Modules = [JuliaFEM]
Pages   = ["basis/api.jl", "basis/basis_generated.jl"]
Order   = [:type, :function]
```

## Quadrature

```@autodocs
Modules = [JuliaFEM]
Pages   = ["quadrature/api.jl", "quadrature/gauss.jl"]
Order   = [:type, :function]
```

## Mesh

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "mesh/api.jl",
    "mesh/mesh.jl",
    "mesh/structured.jl",
    "mesh/refine.jl",
]
Order   = [:type, :function]
```

## DOFs and elements

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "dofs/api.jl",
    "dofs/fields.jl",
    "dofs/dofs.jl",
    "dofs/dof_handler.jl",
    "dofs/dof_connectivity.jl",
    "elements/elements.jl",
    "elements/extract_element_dofs.jl",
    "elements/interpolate.jl",
]
Order   = [:type, :function, :macro]
```

## Materials

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "materials/api.jl",
    "materials/state_variables.jl",
    "materials/traits.jl",
    "materials/linear_elastic.jl",
    "materials/neo_hookean.jl",
    "materials/perfect_plasticity.jl",
    "materials/heat_conductivity.jl",
]
Order   = [:type, :function]
```

## Assemblers

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "assemblers/abstract.jl",
    "assemblers/caches/coo_cache.jl",
    "assemblers/caches/element_cache.jl",
    "assemblers/caches/geometry_cache.jl",
    "assemblers/caches/material_cache.jl",
    "assemblers/element_based/element_based_coo.jl",
    "assemblers/element_based/scatter_blocks_to_force.jl",
    "assemblers/element_based/scatter_blocks_to_triplets_symmetric_direct.jl",
    "assemblers/microkernel.jl",
    "assemblers/dof_based/dof_based_coo.jl",
    "assemblers/dof_based/dof_based_coo_ka.jl",
    "assemblers/matrix_free/operator.jl",
    "assemblers/matrix_free/dirichlet.jl",
    "assemblers/matrix_free/mpc.jl",
    "assemblers/matrix_free/preconditioners.jl",
    "assemblers/matrix_free/eigensolve.jl",
    "assemblers/matrix_free/loads.jl",
]
Order   = [:type, :function]
```

## Physics kernels

```@autodocs
Modules = [JuliaFEM]
Pages   = [
    "domains/continuum/kernel.jl",
    "domains/heat/kernel.jl",
    "domains/thermo_elastic/kernel.jl",
]
Order   = [:type, :function]
```

## Index

```@index
```
