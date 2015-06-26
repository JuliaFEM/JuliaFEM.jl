# API-INDEX


## MODULE: JuliaFEM

---

## Methods [Exported]

[add_elements(model,  elements)](JuliaFEM.md#method__add_elements.1)  Add new elements to model.

[get_elements(model,  element_ids)](JuliaFEM.md#method__get_elements.1)  Get subset of elements from model.

[get_field(field_type,  field_name)](JuliaFEM.md#method__get_field.1)  Get field from model.

[new_model()](JuliaFEM.md#method__new_model.1)  Initialize empty model.

---

## Types [Exported]

[JuliaFEM.Model](JuliaFEM.md#type__model.1)  Basic model

## MODULE: JuliaFEM.elasticity_solver

---

## Methods [Internal]

[assemble!(fe,  eldofs_,  I,  V)](JuliaFEM.elasticity_solver.md#method__assemble.1)  Assemble global RHS to I,V ready for sparse format

[assemble!(ke,  eldofs_,  I,  J,  V)](JuliaFEM.elasticity_solver.md#method__assemble.2)  Assemble global stiffness matrix to I,J,V ready for sparse format

[calc_local_matrices!(X,  u,  R,  Kt,  N,  dNdchi,  lambda_,  mu_,  ipoints,  iweights)](JuliaFEM.elasticity_solver.md#method__calc_local_matrices.1)  Calculate local tangent stiffness matrix and residual force vector R = T - F

[eliminate_boundary_conditions(dirichletbc,  I,  J,  V)](JuliaFEM.elasticity_solver.md#method__eliminate_boundary_conditions.1)  Eliminate Dirichlet boundary conditions from matrix

[eliminate_boundary_conditions(dirichletbc,  I,  V)](JuliaFEM.elasticity_solver.md#method__eliminate_boundary_conditions.2)  Eliminate Dirichlet boundary conditions from vector

[interpolate{T<:Real}(field::Array{T<:Real, 1},  basis::Function,  ip)](JuliaFEM.elasticity_solver.md#method__interpolate.1)  Interpolate field variable using basis functions f for point ip.

[solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights)](JuliaFEM.elasticity_solver.md#method__solve_elasticity_increment.1)  Solve one increment of elasticity problem

