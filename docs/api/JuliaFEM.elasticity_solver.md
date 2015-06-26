# JuliaFEM.elasticity_solver

## Internal

---

<a id="method__assemble.1" class="lexicon_definition"></a>
#### assemble!(fe,  eldofs_,  I,  V) [¶](#method__assemble.1)
Assemble global RHS to I,V ready for sparse format

Parameters
----------
fe : local vector
eldofs_ : Array
  degrees of freedom
I,V : Arrays for sparse matrix

Notes
-----
eldofs can also be node ids for convenience. In that case dimension
is calculated and eldofs are "extended" to problem dimension.


*source:*
[JuliaFEM/src/elasticity_solver.jl:171](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L171)

---

<a id="method__assemble.2" class="lexicon_definition"></a>
#### assemble!(ke,  eldofs_,  I,  J,  V) [¶](#method__assemble.2)
Assemble global stiffness matrix to I,J,V ready for sparse format

Parameters
----------
ke : local matrix
eldofs_ : Array
  degrees of freedom
I,J,V : Arrays for sparse matrix

Notes
-----
eldofs can also be node ids for convenience. In that case dimension
is calculated and eldofs are "extended" to problem dimension.


*source:*
[JuliaFEM/src/elasticity_solver.jl:130](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L130)

---

<a id="method__calc_local_matrices.1" class="lexicon_definition"></a>
#### calc_local_matrices!(X,  u,  R,  Kt,  N,  dNdchi,  lambda_,  mu_,  ipoints,  iweights) [¶](#method__calc_local_matrices.1)
Calculate local tangent stiffness matrix and residual force vector R = T - F


*source:*
[JuliaFEM/src/elasticity_solver.jl:68](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L68)

---

<a id="method__eliminate_boundary_conditions.1" class="lexicon_definition"></a>
#### eliminate_boundary_conditions(dirichletbc,  I,  J,  V) [¶](#method__eliminate_boundary_conditions.1)
Eliminate Dirichlet boundary conditions from matrix

Parameters
----------
dirichletbc : array [dim x nnodes]
I, J, V : sparse matrix arrays

Returns
-------
I, J, V : boundary conditions removed

Notes
-----
pros:
- matrix assembly remains positive definite
cons:
- maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
- if u != 0 in dirichlet boundary requires extra care

Raises
------
Exception, if displacement boundary conditions given, i.e.
DX=2 for some node, for example.



*source:*
[JuliaFEM/src/elasticity_solver.jl:218](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L218)

---

<a id="method__eliminate_boundary_conditions.2" class="lexicon_definition"></a>
#### eliminate_boundary_conditions(dirichletbc,  I,  V) [¶](#method__eliminate_boundary_conditions.2)
Eliminate Dirichlet boundary conditions from vector

Parameters
----------
dirichletbc : array [dim x nnodes]
I, V : sparse vector arrays

Returns
-------
I, V : boundary conditions removed

Notes
-----
pros:
- matrix assembly remains positive definite
cons:
- maybe inefficient because of extra sparse matrix operations. (It's hard to remove stuff from sparse matrix.)
- if u != 0 in dirichlet boundary requires extra care

Raises
------
Exception, if displacement boundary conditions given, i.e.
DX=2 for some node, for example.


*source:*
[JuliaFEM/src/elasticity_solver.jl:257](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L257)

---

<a id="method__interpolate.1" class="lexicon_definition"></a>
#### interpolate{T<:Real}(field::Array{T<:Real, 1},  basis::Function,  ip) [¶](#method__interpolate.1)
Interpolate field variable using basis functions f for point ip.
This function tries to be as general as possible and allows interpolating
lot of different fields.

Parameters
----------
field :: Array{Number, dim}
  Field variable
basis :: Function
  Basis functions
ip :: Array{Number, 1}
  Point to interpolate


*source:*
[JuliaFEM/src/elasticity_solver.jl:30](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L30)

---

<a id="method__solve_elasticity_increment.1" class="lexicon_definition"></a>
#### solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights) [¶](#method__solve_elasticity_increment.1)
Solve one increment of elasticity problem


*source:*
[JuliaFEM/src/elasticity_solver.jl:278](https://github.com/JuliaFEM/JuliaFEM.jl/tree/92c5e6c15a1ffaea4c153cba2af3a62ba3b42ebe/src/elasticity_solver.jl#L278)

