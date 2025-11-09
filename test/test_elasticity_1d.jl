# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

# 1d strain

# Global node data (Dict format for backward compatibility in tests)
X_global = Dict(1 => [0.0, 0.0, 0.0], 2 => [1.0, 1.0, 1.0])
u_global = Dict(1 => [0.0, 0.0, 0.0], 2 => [1.0, 1.0, 1.0])

# Convert to element-local format (extract data for element nodes)
connectivity = (1, 2)
X = tuple([X_global[i] for i in connectivity]...)
u = tuple([u_global[i] for i in connectivity]...)

# Wrap in field objects (DVTI = Discrete, Variable, Time-Invariant)
X_field = JuliaFEM.DVTI(X)
u_field = JuliaFEM.DVTI(u)

# Create element with fields at construction (immutable pattern)
element = Element(Seg2, connectivity; fields=(geometry=X_field, displacement=u_field))

xi, time = (0.0,), 0.0
detJ = element(xi, time, Val{:detJ})
J = element(xi, time, Val{:Jacobian})
# gradu = element("displacement", xi, time, Val{:Grad})
@debug("1d seg2 info", xi, time, detJ, J)
@test isapprox(detJ, sqrt(3) / 2)
# Jacobian is 3×1 (physical_dim × parametric_dim) for 1D element in 3D
@test isapprox(J, [0.5; 0.5; 0.5])  # column vector
