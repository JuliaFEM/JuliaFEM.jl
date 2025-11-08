# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

code = create_basis_and_eval(
    :Tet4,
    "4 node linear tetrahedral element",
    [
     (0.0, 0.0, 0.0), # N1
     (1.0, 0.0, 0.0), # N2
     (0.0, 1.0, 0.0), # N3
     (0.0, 0.0, 1.0), # N4
    ],
    :(1 + u + v + w),
   )

code = create_basis_and_eval(
    :Tet10,
    "10 node quadratic tetrahedral element",
    [
     (0.0, 0.0, 0.0), # N1
     (1.0, 0.0, 0.0), # N2
     (0.0, 1.0, 0.0), # N3
     (0.0, 0.0, 1.0), # N4
     (0.5, 0.0, 0.0), # N5
     (0.5, 0.5, 0.0), # N6
     (0.0, 0.5, 0.0), # N7
     (0.0, 0.0, 0.5), # N8
     (0.5, 0.0, 0.5), # N9
     (0.0, 0.5, 0.5), # N10
    ],
    :(1 + u + v + w + u*v + v*w + w*u + u^2 + v^2 + w^2),
   )
