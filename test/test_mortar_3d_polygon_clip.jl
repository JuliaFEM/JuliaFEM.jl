# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test
using JuliaFEM: get_polygon_clip, calculate_polygon_area

## polygon clipping

S = [[0.375, 0.000, 0.500], [0.600, 0.000, 0.500], [0.500, 0.250, 0.500]]
M = [[0.500, 0.000, 0.500], [0.250, 0.000, 0.500], [0.375, 0.250, 0.500]]
n0 = [0.0, 0.0, 1.0]
P = get_polygon_clip(S, M, n0)
@test length(P) == 3
@test isapprox(calculate_polygon_area(P), 1/128)

S = [[0.250, 0.000, 0.500], [0.750, 0.000, 0.500], [0.500, 0.250, 0.500]]
M = [[0.500, 0.000, 0.500], [0.250, 0.000, 0.500], [0.375, 0.250, 0.500]]
n0 = [0.0, 0.0, 1.0]
P = get_polygon_clip(S, M, n0)
@test length(P) == 3
@test isapprox(calculate_polygon_area(P), 1/48)

# visually inspected
Xs = [[ 0.00, 0.00, 0.50], [1.00,  0.00, 0.50], [0.00, 1.00, 0.50]]
Xm = [[-0.25, 0.50, 0.50], [0.50, -0.25, 0.50], [0.75, 0.75, 0.50]]
P_ = [[0.65, 0.35, 0.00], [0.5625, 0.0,    0.0], [0.25, 0.00, 0.0],
      [0.00, 0.25, 0.00], [0.0,    0.5625, 0.0], [0.35, 0.65, 0.0]]
P = get_polygon_clip(Xs, Xm, [0.0, 0.0, 1.0])
@test length(P) == length(P_)
