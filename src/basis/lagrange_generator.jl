# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

# ==============================================================================
# LAGRANGE BASIS FUNCTION GENERATOR
# ==============================================================================
#
# This file contains the symbolic engine for generating Lagrange basis functions.
# It is NOT loaded at runtime - it's a TOOL used during development.
#
# PURPOSE:
#   Generate pre-computed basis functions and derivatives for all standard
#   Lagrange finite elements (Seg2, Tri3, Quad4, Tet10, Hex8, etc.)
#
# THEORY:
#   Lagrange basis functions satisfy the Kronecker delta property:
#   
#   N_i(x_j) = δ_ij = { 1  if i = j
#                     { 0  if i ≠ j
#
#   Given:
#   - n nodes with coordinates {x₁, x₂, ..., xₙ} in reference element
#   - Polynomial ansatz {p₁(x), p₂(x), ..., pₙ(x)} (complete to order k)
#
#   We construct: N_i(x) = Σⱼ αᵢⱼ pⱼ(x)
#
#   The Kronecker property gives: V α_i = e_i
#   
#   Where Vandermonde matrix: V_kj = pⱼ(x_k)
#
#   Solving these n systems gives all basis functions explicitly.
#   Then symbolic differentiation provides derivatives.
#
# USAGE:
#   This file is loaded by scripts/generate_lagrange_basis.jl which:
#   1. Defines all standard element types (coords + polynomial ansatz)
#   2. Calls generate_lagrange_basis() for each
#   3. Writes clean Julia code to src/basis/lagrange_generated.jl
#
# WHY GENERATE ONCE?
#   - Symbolic math is expensive (100+ ms per element type)
#   - Generated code is constant (mathematics doesn't change!)
#   - Pre-compilation is much faster
#   - Generated code is readable and debuggable
#   - Version control shows what changed
#
# SEE:
#   - docs/book/lagrange_basis_functions.md (mathematical explanation)
#   - scripts/generate_lagrange_basis.jl (generation script)
#   - src/basis/lagrange_generated.jl (output - do not edit manually!)
#
# ==============================================================================

__precompile__(false)  # This is a tool, not runtime code

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

    # Build tuple expression for eval_basis! return: (N1, N2, N3, ...)
    basis_tuple_args = [basis[i] for i = 1:N]
    basis_tuple = Expr(:tuple, basis_tuple_args...)

    # Build tuple expression for eval_dbasis! return: (dN1, dN2, dN3, ...)
    dbasis_tuple_args = [:(Vec(float.(tuple($(dbasis[:, i]...))))) for i = 1:N]
    dbasis_tuple = Expr(:tuple, dbasis_tuple_args...)

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

        # Return tuple directly - zero allocations!
        @inline function eval_basis!(::Type{$name}, ::Type{T}, xi::Vec) where T
            $unpack
            @inbounds return $basis_tuple
        end

        # Return NTuple{N,Vec{D}} directly - zero allocations!
        @inline function eval_dbasis!(::Type{$name}, xi::Vec)
            $unpack
            @inbounds return $dbasis_tuple
        end
    end
    return code
end

create_basis_and_eval(args...) = eval(create_basis(args...))

