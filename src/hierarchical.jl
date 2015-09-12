# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# some preliminary code for constructing hierarchical elements

"""
Return Legendgre polynomial of order n to inverval ξ ∈  [-1, 1]

Maybe slow version?
"""
function get_legendre_polynomial_2(n)
    bin(n, k) = prod([(n + 1 - i)/i for i=1:k])
    P(xi) = sum([2^n*xi.^k*bin(n, k)*bin(1/2*(n+k-1), n) for k=0:n])
    P
end

"""
Return Legendgre polynomial of order n to inverval ξ ∈ [1, 1].

Parameters
----------
n :: Int
    order of polynomial

Returns
-------
function
    Legendgre polynomial of order n in interval ξ ∈ [-1, 1]

Notes
-----
Uses Bonnet's recursion formula. See
https://en.wikipedia.org/wiki/Legendre_polynomials
"""
function get_legendre_polynomial(n)
    if n == 0
        P(xi) = 1
    elseif n == 1
        P(xi) = xi
    else
        Pm1 = get_legendre_polynomial(n-1)
        Pm2 = get_legendre_polynomial(n-2)
        P(xi) = 1/n*((2*n-1)*xi*Pm1(xi) - (n-1)*Pm2(xi))
    end
    return P
end

"""
Return derivative of Legendgre polynomial of order n to inverval ξ ∈  [-1, 1]
"""
function get_legendre_polynomial_derivative(n)
    if n == 0
        P(xi) = 0*xi
    elseif n == 1
        P(xi) = 0*xi + 1
    else
        Pm1 = get_legendre_polynomial_derivative(n-1)
        Pm2 = get_legendre_polynomial_derivative(n-2)
        P(xi) = 1/(n-1)*( (2*(n-1)+1)*xi.*Pm1(xi) - (n+1-1)*Pm2(xi))
    end
    return P
end

"""
Return hierarchical shape function of order N
"""
function get_hierarchial_basis(n)
    if n == 1
        N(xi) = 1/2*(1 - xi)
    elseif n == 2
        N(xi) = 1/2*(1 + xi)
    else
        j = n-1
        Pj = get_legendre_polynomial(j)
        Pjm2 = get_legendre_polynomial(j-2)
        N(xi) = 1/sqrt(2*(2*j-1))*(Pj(xi) - Pjm2(xi))
    end
    return N
end

"""
Return derivative of hierarchical shape function of order N
"""
function get_hierarchial_basis_derivative(n)
    if n == 1
        dN(xi) = 1/2*(0*xi - 1)
    elseif n == 2
        dN(xi) = 1/2*(0*xi + 1)
    else
        j = n-1
        Pj = get_legendre_polynomial_derivative(j)
        Pjm2 = get_legendre_polynomial_derivative(j-2)
        dN(xi) = 1/sqrt(2*(2*j-1))*(Pj(xi) - Pjm2(xi))
    end
    return dN
end

