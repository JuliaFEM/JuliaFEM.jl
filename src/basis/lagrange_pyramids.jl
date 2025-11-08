# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

# Kaltenbacher, Manfred. Numerical simulation of mechatronic sensors and actuators: finite elements for computational multiphysics. Springer, 2015.
code = create_basis_and_eval(
    :Pyr5A,
    "5 node linear pyramid element",
    [
     ( 1.0,  1.0, 0.0), # N1
     ( 1.0, -1.0, 0.0), # N2
     (-1.0, -1.0, 0.0), # N3
     (-1.0,  1.0, 0.0), # N4
     ( 0.0,  0.0, 1.0), # N5
    ],
    [
     :(1/4*( (1+u)*(1+v) - w + u*v*w/(1-w) )),
     :(1/4*( (1+u)*(1-v) - w + u*v*w/(1-w) )),
     :(1/4*( (1-u)*(1-v) - w + u*v*w/(1-w) )),
     :(1/4*( (1-u)*(1+v) - w + u*v*w/(1-w) )),
     :(1.0*w),
    ],
   )

# source: Code Aster documentation?
code = create_basis_and_eval(
    :Pyr5,
    "5 node linear pyramid element",
    [
     (-1.0, -1.0, -1.0), # N1
     ( 1.0, -1.0, -1.0), # N2
     ( 1.0,  1.0, -1.0), # N3
     (-1.0,  1.0, -1.0), # N4
     ( 0.0,  0.0,  1.0), # N5
    ],
    [
     :(1/8 * (1-u) * (1-v) * (1-w)),
     :(1/8 * (1+u) * (1-v) * (1-w)),
     :(1/8 * (1+u) * (1+v) * (1-w)),
     :(1/8 * (1-u) * (1+v) * (1-w)),
     :(1/2 * (1+w)),
    ],
   )
