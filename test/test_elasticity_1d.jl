# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# 1d strain

X = Dict(1 => [0.0, 0.0, 0.0], 2 => [1.0, 1.0, 1.0])
u = Dict(1 => [0.0, 0.0, 0.0], 2 => [1.0, 1.0, 1.0])
element = Element(Seg2, (1, 2))
update!(element, "geometry", X)
update!(element, "displacement", u)
xi, time = (0.0,), 0.0
detJ = element(xi, time, Val{:detJ})
J = element(xi, time, Val{:Jacobian})
# gradu = element("displacement", xi, time, Val{:Grad})
@debug("1d seg2 info", xi ,time, detJ, J)
@test isapprox(detJ, sqrt(3)/2)
@test isapprox(J, [0.5 0.5 0.5])
