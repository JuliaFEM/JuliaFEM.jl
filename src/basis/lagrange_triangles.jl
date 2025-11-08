# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

code = create_basis_and_eval(
    :Tri3,
    "3 node linear triangle element",
    [
     (0.0, 0.0), # N1
     (1.0, 0.0), # N2
     (0.0, 1.0), # N3
    ],
    :(1 + u + v),
   )

code = create_basis_and_eval(
    :Tri6,
    "6 node quadratic triangle element",
    [
     (0.0, 0.0), # N1
     (1.0, 0.0), # N2
     (0.0, 1.0), # N3
     (0.5, 0.0), # N4
     (0.5, 0.5), # N5
     (0.0, 0.5), # N6
    ],
    :(1 + u + v + u^2 + u*v + v^2),
   )

code = create_basis_and_eval(
    :Tri7,
    "7 node quadratic triangle element (has middle node)",
    [
     (0.0, 0.0), # N1
     (1.0, 0.0), # N2
     (0.0, 1.0), # N3
     (0.5, 0.0), # N4
     (0.5, 0.5), # N5
     (0.0, 0.5), # N6
     (1/3, 1/3), # N7
    ],
    :(1 + u + v + u^2 + u*v + v^2 + u^2*v^2),
   )
