# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

# Kaltenbacher, Manfred. Numerical simulation of mechatronic sensors and actuators: finite elements for computational multiphysics. Springer, 2015.
create_basis_and_eval(
    :Wedge6,
    "6 node linear prismatic/wedge element",
    [
     (0.0, 0.0, -1.0), # N1
     (1.0, 0.0, -1.0), # N2
     (0.0, 1.0, -1.0), # N3
     (0.0, 0.0,  1.0), # N4
     (1.0, 0.0,  1.0), # N5
     (0.0, 1.0,  1.0), # N6
    ],
    [
     :(1/2 * (1-w) * (1-u-v)), # N1
     :(1/2 * (1-w) * u), # N2
     :(1/2 * (1-w) * v), # N3
     :(1/2 * (1+w) * (1-u-v)), # N4
     :(1/2 * (1+w) * u), # N5
     :(1/2 * (1+w) * v), # N6
    ],
   )

# Basis functions are from ABAQUS theory manual
create_basis_and_eval(
    :Wedge15,
    "15 node quadratic prismatic/wedge element",
    [
     (0.0, 0.0, -1.0), # N1
     (1.0, 0.0, -1.0), # N2
     (0.0, 1.0, -1.0), # N3
     (0.0, 0.0,  1.0), # N4
     (1.0, 0.0,  1.0), # N5
     (0.0, 1.0,  1.0), # N6
     (0.5, 0.0, -1.0), # N7
     (0.5, 0.5, -1.0), # N8
     (0.0, 0.5, -1.0), # N9
     (0.5, 0.0,  1.0), # N10
     (0.5, 0.5,  1.0), # N11
     (0.0, 0.5,  1.0), # N12
     (0.0, 0.0,  0.0), # N13
     (1.0, 0.0,  0.0), # N14
     (0.0, 1.0,  0.0), # N15
    ],
    [
     :(u^2*w^2 - u^2*w + 2*u*v*w^2 - 2*u*v*w - 3*u*w^2/2 + 3*u*w/2 + v^2*w^2 - v^2*w - 3*v*w^2/2 + 3*v*w/2 + w^2/2 - w/2),
     :(u^2*w^2 - u^2*w - u*w^2/2 + u*w/2),
     :(v^2*w^2 - v^2*w - v*w^2/2 + v*w/2),
     :(u^2*w^2 + u^2*w + 2*u*v*w^2 + 2*u*v*w - 3*u*w^2/2 - 3*u*w/2 + v^2*w^2 + v^2*w - 3*v*w^2/2 - 3*v*w/2 + w^2/2 + w/2),
     :(u^2*w^2 + u^2*w - u*w^2/2 - u*w/2),
     :(v^2*w^2 + v^2*w - v*w^2/2 - v*w/2),
     :(-2*u^2*w^2 + 2*u^2*w - 2*u*v*w^2 + 2*u*v*w + 2*u*w^2 - 2*u*w),
     :(2*u*v*w^2 - 2*u*v*w),
     :(-2*u*v*w^2 + 2*u*v*w - 2*v^2*w^2 + 2*v^2*w + 2*v*w^2 - 2*v*w),
     :(2*u^2*w^2 - 2*u^2*w - 4*u^2 + 2*u*v*w^2 - 2*u*v*w - 4*u*v - 2*u*w^2 + 2*u*w + 4*u),
     :(-2*u*v*w^2 + 2*u*v*w + 4*u*v),
     :(2*u*v*w^2 - 2*u*v*w - 4*u*v + 2*v^2*w^2 - 2*v^2*w - 4*v^2 - 2*v*w^2 + 2*v*w + 4*v),
     :(-2*u^2*w^2 + 2*u^2 - 4*u*v*w^2 + 4*u*v + 3*u*w^2 - 3*u - 2*v^2*w^2 + 2*v^2 + 3*v*w^2 - 3*v - w^2 + 1),
     :(-2*u^2*w^2 + 2*u^2 + u*w^2 - u),
     :(-2*v^2*w^2 + 2*v^2 + v*w^2 - v),
    ],
   )
