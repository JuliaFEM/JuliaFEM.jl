# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

code = create_basis_and_eval(
    :Seg2,
    "2 node linear segment/line element",
    [
     (-1.0,),
     ( 1.0,),
    ],
    :(1 + u))

code = create_basis_and_eval(
    :Seg3,
    "3 node quadratic segment/line element",
    [
     (-1.0,),
     ( 1.0,),
     ( 0.0,),
    ],
    :(1 + u + u^2))
