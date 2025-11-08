# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

"""
    vandermonde_matrix(polynomial, coordinates)

Given some polynomial and coordinates points (1-3 dimensions), create a Vandermonde
matrix.

# Example

To genererate a Vandermonde matrix for a reference quadrangle `[-1.0, 1.0]^2` for
polynomial `p(u,v) = 1 + u + v + u*v`, one writes:

```julia
polynomial = :(1 + u + v + u*v)
coordinates = [(-1.0,-1.0), (1.0,-1.0), (1.0,1.0), (-1.0,1.0)]
V = vandermonde_matrix(polynomial, coordinates)

# output

[
1.0 -1.0 -1.0  1.0
1.0  1.0 -1.0 -1.0
1.0  1.0  1.0  1.0
1.0 -1.0  1.0 -1.0
]

```

# References
- Wikipedia contributors. (2018, August 1). Vandermonde matrix. In Wikipedia, The Free Encyclopedia. Retrieved 10:00, August 20, 2018, from https://en.wikipedia.org/w/index.php?title=Vandermonde_matrix&oldid=852930962
"""
function vandermonde_matrix(polynomial::Expr, coordinates::Vector{NTuple{D, T}}) where {D, T<:Number}
    N = length(coordinates)
    A = zeros(N, N)
    first(polynomial.args) == :+ || error("Use only summation between terms of polynomial")
    args = polynomial.args[2:end]
    for i in 1:N
        X = coordinates[i]
        if D == 1
            data = (:u => X[1],)
        elseif D == 2
            data = (:u => X[1], :v => X[2])
        elseif D == 3
            data = (:u => X[1], :v => X[2], :w => X[3])
        end
        for (j, term) in enumerate(args)
            A[i,j] = subs(term, data)
        end
    end
    return A
end
