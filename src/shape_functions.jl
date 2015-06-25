# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function test(a)
     return a*2
end


function C2D4(xx)
     theta = 1
     chi = xx[1]
     eta = xx[2]
     return = [(1 - chi) * (1 - eta),
               chi * (1 - eta),
               chi * eta,
               (1 - chi) * eta]
end

