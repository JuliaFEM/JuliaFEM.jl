# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMBasis.jl/blob/master/LICENSE

__precompile__(false)

function get_reference_element_coordinates end
function eval_basis! end
function eval_dbasis! end

function calculate_interpolation_polynomials(p, V)
    basis = []
    first(p.args) == :+ || error("Use only summation between terms of polynomial")
    args = p.args[2:end]
    n = size(V, 1)
    b = zeros(n)
    for i in 1:n
        fill!(b, 0.0)
        b[i] = 1.0
        # TODO: Think about numerical stability with
        # inverting Vandermonde matrix?
        solution = V \ b
        N = Expr(:call, :+)
        for (ai, bi) in zip(solution, args)
            isapprox(ai, 0.0) && continue
            push!(N.args, Calculus.simplify( :( $ai * $bi ) ))
        end
        push!(basis, N)
    end
    return basis
end

function calculate_interpolation_polynomial_derivatives(basis, D)
    vars = [:u, :v, :w]
    dbasis = Matrix(undef, D, length(basis))
    for (i, N) in enumerate(basis)
        partial_derivatives = []
        for j in 1:D
            dbasis[j, i] = Calculus.simplify(Calculus.differentiate(N, vars[j]))
        end
    end
    return dbasis
end

function create_basis(name, description, X::Vector{<:Vecish{D}}, p::Expr) where D
    @debug "create basis given antsatz polynomial" name description X p
    V = vandermonde_matrix(p, X)
    basis = calculate_interpolation_polynomials(p, V)
    return create_basis(name, description, X, basis)
end

function create_basis(name, description, X::Vector{<:Vecish{D}}, basis::Vector) where D
    @assert length(X) == length(basis)
    @debug "create basis given basis functions" name description X basis
    dbasis = calculate_interpolation_polynomial_derivatives(basis, D)
    return create_basis(name, description, Vec.(X), basis, dbasis)
end

function create_basis(name, description, X::Vector{<:Vecish{D, T}}, basis, dbasis) where {D, T}
    N = length(X)
    @debug "create basis given basis functions and derivatives" name description X basis dbasis

    Q = Expr(:block)
    for i=1:N
        push!(Q.args, :(N[$i] = $(basis[i])))
    end

    V = Expr(:block)
    for i=1:N
        push!(V.args, :(dN[$i] = Vec(float.(tuple($(dbasis[:, i]...))))))
    end

    if D == 1
        unpack = :((u,) = xi)
    elseif D == 2
        unpack = :((u, v) = xi)
    else
        unpack = :((u, v, w) = xi)
    end

    code = quote
        struct $name <: FEMBasis.AbstractBasis{$D}
        end

        Base.@pure function Base.size(::Type{$name})
            return ($D, $N)
        end

        function Base.size(::Type{$name}, j::Int)
            j == 1 && return $D
            j == 2 && return $N
        end

        Base.@pure function Base.length(::Type{$name})
            return $N
        end

        function FEMBasis.get_reference_element_coordinates(::Type{$name})
            return $X
        end

        @inline function FEMBasis.eval_basis!(::Type{$name}, N::Vector{<:Number}, xi::Vec)
            @assert length(N) == $N
            $unpack
            @inbounds $Q
            return N
        end

        @inline function FEMBasis.eval_dbasis!(::Type{$name}, dN::Vector{<:Vec{$D}}, xi::Vec)
            @assert length(dN) == $N
            $unpack
            @inbounds $V
            return dN
        end
    end
    return code
end

create_basis_and_eval(args...) = eval(create_basis(args...))

