function test(a)
     return a*2
end


function C2D4(xx)
     chi = xx[1]
     eta = xx[2]
     return = [(1 - chi) * (1 - eta),
               chi * (1 - eta),
               chi * eta,
               (1 - chi) * eta]
end

