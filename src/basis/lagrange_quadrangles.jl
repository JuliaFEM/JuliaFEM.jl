# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

code = create_basis_and_eval(
    :Quad4,
    "4 node linear quadrangle element",
    [
     (-1.0, -1.0), # N1
     ( 1.0, -1.0), # N2
     ( 1.0,  1.0), # N3
     (-1.0,  1.0)  # N4
    ],
    :(1 + u + v + u*v),
   )

code = create_basis_and_eval(
    :Quad8,
    "8 node quadratic quadrangle element (Serendip)",
    [
     (-1.0, -1.0), # N1
     ( 1.0, -1.0), # N2
     ( 1.0,  1.0), # N3
     (-1.0,  1.0), # N4
     ( 0.0, -1.0), # N5
     ( 1.0,  0.0), # N6
     ( 0.0,  1.0), # N7
     (-1.0,  0.0)  # N8
    ],
    :(1 + u + v + u*v + u^2 + u^2*v + u*v^2 + v^2),
   )

code = create_basis_and_eval(
    :Quad9,
    "9 node quadratic quadrangle element",
    [
     (-1.0, -1.0), # N1
     ( 1.0, -1.0), # N2
     ( 1.0,  1.0), # N3
     (-1.0,  1.0), # N4
     ( 0.0, -1.0), # N5
     ( 1.0,  0.0), # N6
     ( 0.0,  1.0), # N7
     (-1.0,  0.0), # N8
     ( 0.0,  0.0)  # N9
    ],
    :(1 + u + v + u*v + u^2 + u^2*v + u*v^2 + v^2 + u^2*v^2),
   )
