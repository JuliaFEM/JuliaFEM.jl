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
[JuliaFEM/src/elasticity_solver.jl:174](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L174)

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
[JuliaFEM/src/elasticity_solver.jl:133](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L133)

---

<a id="method__calc_local_matrices.1" class="lexicon_definition"></a>
#### calc_local_matrices!(X,  u,  R,  K,  basis,  dbasis,  lambda_,  mu_,  ipoints,  iweights) [¶](#method__calc_local_matrices.1)
Calculate local tangent stiffness matrix and residual force vector
R = T - F for elasticity problem.

Parameters
----------
X : Element coordinates
u : Displacement field
R : Residual force vector
K : Tangent stiffness matrix
basis : Basis functions
dbasis : Derivative of basis functions
lambda : Material parameter
mu : Material parameter
ipoints : integration points
iweights : integration weights

Returns
-------
None

Notes
-----
If material parameters are given in list, they are interpolated to gauss
points using shape functions.


*source:*
[JuliaFEM/src/elasticity_solver.jl:76](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L76)

---

<a id="method__dummy.1" class="lexicon_definition"></a>
#### dummy(a) [¶](#method__dummy.1)
This is dummy function. Testing doctests and documentation.

Parameters
----------
x : Array{Float64, 1}

Returns
-------
Array{float64, 1}
  x + 1

Notes
-----
This is dummy function

Raises
------
Exception
  if things are not going right

Examples
--------
>>> a = [1.0, 2.0, 3.0]
>>> dummy(a)
[2.0, 3.0, 4.0]


*source:*
[JuliaFEM/src/elasticity_solver.jl:44](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L44)

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
[JuliaFEM/src/elasticity_solver.jl:221](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L221)

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
[JuliaFEM/src/elasticity_solver.jl:260](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L260)

---

<a id="method__solve_elasticity_increment.1" class="lexicon_definition"></a>
#### solve_elasticity_increment!(X,  u,  du,  elmap,  nodalloads,  dirichletbc,  lambda,  mu,  N,  dNdchi,  ipoints,  iweights) [¶](#method__solve_elasticity_increment.1)
Solve one increment of elasticity problem


*source:*
[JuliaFEM/src/elasticity_solver.jl:281](https://github.com/JuliaFEM/JuliaFEM.jl/tree/33a7fe664e9808c57564b507f0b8d5dcb451365a/src/elasticity_solver.jl#L281)

