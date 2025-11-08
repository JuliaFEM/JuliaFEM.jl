# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

import Base: size, length

function NURBS(i, p, u, t)
    p == 0 && return t[i] <= u <= t[i+1] ? 1.0 : 0.0
    anom = u-t[i]
    adenom = t[i+p]-t[i]
    a = isapprox(adenom, 0.0) ? 0.0 : anom/adenom
    bnom = t[i+p+1]-u
    bdenom = t[i+p+1]-t[i+1]
    b = isapprox(bdenom, 0.0) ? 0.0 : bnom/bdenom
    result = a*NURBS(i,p-1,u,t) + b*NURBS(i+1,p-1,u,t)
    return result
end
