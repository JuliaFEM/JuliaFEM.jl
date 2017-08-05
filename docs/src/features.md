===============
FEATURES / TODO
===============

- Parallel design
- Sparse matrices
- Discontinuous Galerkin?
- 100% Tested
  - Automatic coverage (Travis CI)
- Doctest examples and tutorials
  - Intuitive to use
- Marketing
  - Engineering porn
- Field functions
- Multiphysics platform
  - Elasticity
  - Thermal implemented
- Transient and steady solvers
  - Implicit dynamics
- Nonlinearities
  - Geometrical
  - Material
  - Contact
    - Mortar
- Modular design
  - E.g. contact formulation can be altered by user
- Mesh and results format, Xdmf?
- No scalars but field variables, e.g. no constant 210GPa for steel, we interpolate variable from nodes.
  - (Constant is a special case of field variable).
