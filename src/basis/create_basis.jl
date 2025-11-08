# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/jl/blob/master/LICENSE

__precompile__(false)

# Minimal symbolic differentiation for polynomial basis functions
# Adapted from SymDiff.jl by Jukka Aho - zero dependencies!

differentiate(::Number, ::Symbol) = 0
differentiate(f::Symbol, x::Symbol) = f == x ? 1 : 0

function differentiate(f::Expr, x::Symbol)
    @assert f.head == :call
    op = first(f.args)

    # Product rule: (fg)' = f'g + fg'
    if op == :*
        res_args = Any[:+]
        for i in 2:length(f.args)
            new_args = copy(f.args)
            new_args[i] = differentiate(f.args[i], x)
            push!(res_args, Expr(:call, new_args...))
        end
        return Expr(:call, res_args...)

        # Power rule: d/dx f^a = a * f^(a-1) * f'
    elseif op == :^
        _, f_inner, a = f.args
        df = differentiate(f_inner, x)
        return :($a * $f_inner^($a - 1) * $df)

        # Sum rule: (f + g)' = f' + g'
    elseif op == :+
        args = differentiate.(f.args[2:end], x)
        return Expr(:call, :+, args...)

        # Difference rule: (f - g)' = f' - g'
    elseif op == :-
        args = differentiate.(f.args[2:end], x)
        return Expr(:call, :-, args...)

        # Quotient rule: d/dx (f/g) = (f'g - fg')/g^2
    elseif op == :/
        _, g, h = f.args
        dg = differentiate(g, x)
        dh = differentiate(h, x)
        return :(($dg * $h - $g * $dh) / $h^2)

    else
        error("Unsupported operation: $op")
    end
end

simplify(f::Union{Number,Symbol}) = f

function simplify(ex::Expr)
    @assert ex.head == :call
    op = first(ex.args)

    # Multiplication: remove 1's, return 0 if any 0
    if op == :*
        args = simplify.(ex.args[2:end])
        0 in args && return 0
        filter!(k -> !(isa(k, Number) && k == 1), args)
        length(args) == 0 && return 1
        length(args) == 1 && return first(args)
        return Expr(:call, :*, args...)

        # Addition: remove 0's
    elseif op == :+
        args = simplify.(ex.args[2:end])
        filter!(k -> !isa(k, Number) || k != 0, args)
        length(args) == 0 && return 0
        length(args) == 1 && return first(args)
        return Expr(:call, :+, args...)

        # Subtraction: remove 0's
    elseif op == :-
        args = simplify.(ex.args[2:end])
        filter!(k -> !isa(k, Number) || k != 0, args)
        length(args) == 0 && return 0
        length(args) == 1 && return first(args)
        return Expr(:call, :-, args...)

        # Power, Division: keep as-is
    elseif op in (:^, :/)
        args = simplify.(ex.args[2:end])
        return Expr(:call, op, args...)

    else
        return ex
    end
end

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
            push!(N.args, simplify(:($ai * $bi)))  # Use our own simplify
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
            dbasis[j, i] = simplify(differentiate(N, vars[j]))  # Use our own differentiate and simplify
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

function create_basis(name, description, X::Vector{<:Vecish{D,T}}, basis, dbasis) where {D,T}
    N = length(X)
    @debug "create basis given basis functions and derivatives" name description X basis dbasis

    Q = Expr(:block)
    for i = 1:N
        push!(Q.args, :(N[$i] = $(basis[i])))
    end

    V = Expr(:block)
    for i = 1:N
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
        struct $name <: AbstractBasis{$D}
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

        function get_reference_element_coordinates(::Type{$name})
            return $X
        end

        @inline function eval_basis!(::Type{$name}, N::Vector{<:Number}, xi::Vec)
            @assert length(N) == $N
            $unpack
            @inbounds $Q
            return N
        end

        @inline function eval_dbasis!(::Type{$name}, dN::Vector{<:Vec{$D}}, xi::Vec)
            @assert length(dN) == $N
            $unpack
            @inbounds $V
            return dN
        end
    end
    return code
end

create_basis_and_eval(args...) = eval(create_basis(args...))

