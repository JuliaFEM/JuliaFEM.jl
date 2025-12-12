# API Reference

Complete API documentation for JuliaFEM.jl.

```@meta
DocTestSetup = quote
    using JuliaFEM
end
```

## Core Types

### Elements

```@docs
Element
topology_type
basis_type
dof_type
element_id
element_dofs
n_element_dofs
```

### Physics

```@docs
Physics
AbstractPhysics
DirichletBC
NeumannBC
Constraint
assemble!
solve!
add_dirichlet!
add_neumann!
```

### Fields

```@docs
AbstractField
Displacement
Temperature
DisplacementRotation
```

### Formulations

```@docs
AbstractFormulation
ContinuumFormulation
AbstractContinuumTheory
FullThreeD
PlaneStress
PlaneStrain
Axisymmetric
```

### Materials

```@docs
AbstractMaterial
AbstractElasticMaterial
AbstractPlasticMaterial
LinearElastic
NeoHookean
compute_stress
elasticity_tensor
```

### Mesh

```@docs
AbstractMesh
Mesh
nnodes_total
nelements
get_node
connectivity_matrix
get_element_set
get_node_set
```

### Topology

```@docs
AbstractTopology
Segment
Triangle
Quadrilateral
Tetrahedron
Hexahedron
Pyramid
Wedge
Seg2
Seg3
Tri3
Tri6
Tri7
Quad4
Quad8
Quad9
Tet4
Tet10
Hex8
Hex20
Hex27
Pyr5
Wedge6
Wedge15
```

### Basis Functions

```@docs
AbstractBasis
Lagrange
Serendipity
get_basis_functions
get_basis_derivatives
eval_basis!
eval_dbasis!
```

### DOF System

```@docs
AbstractDOF
DOFSet
@DOFSet
DOFManager
register_fields!
create_elements!
```

### Solvers

```@docs
Solver
Linear
Analysis
add_problems!
run!
```

## Index

```@index
```
