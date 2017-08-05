# Simple usage examples

A simple example demonstrating the basic usage of package. Calculate a simple
one element model. Add pressure load on top and support block symmetrically.

```@example 1
using JuliaFEM # hide
X = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 1.0],
    4 => [0.0, 1.0])
```

```@example 1
element = Element(Quad4, [1, 2, 3, 4])
update!(element, "geometry", X)
update!(element, "youngs modulus", 288.0)
update!(element, "poissons ratio", 1/3)
```

First define a field problem and add element to it
```@example 1
body = Problem(Elasticity, "test problem", 2)
update!(body.properties,
    "formulation" => "plane_stress",
    "finite_strain" => "false",
    "geometric_stiffness" => "false")
body.elements = [element]
```

Then create element to carry on pressure
```@example 1
tr_el = Element(Seg2, [3, 4])
update!(tr_el, "geometry", X)
update!(tr_el, "displacement traction force 2", 288.0)
traction = Problem(Elasticity, "pressure on top of block", 2)
update!(traction.properties,
    "formulation" => "plane_stress",
    "finite_strain" => "false",
    "geometric_stiffness" => "false")
traction.elements = [tr_el]
```

Create boundary condition to support block at bottom and left
```@example 1
bc_el_1 = Element(Seg2, [1, 2])
bc_el_2 = Element(Seg2, [4, 1])
update!(bc_el_1, "displacement 2", 0.0)
update!(bc_el_2, "displacement 1", 0.0)
bc = Problem(Dirichlet, "add symmetry bc", 2, "displacement")
bc.elements = [bc_el_1, bc_el_2]
```

Last thing is to create a solver, push problem to solver and solve:
```@example 1
solver = Solver(Linear, body, traction, bc)
solver()
```

Displacement in node 3 is
```@example 1
solver("displacement", 0.0)[3]
```
