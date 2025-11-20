# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

# ==============================================================================
# BASIS GENERATOR (symbolic) — tooling only, not loaded at runtime
# ==============================================================================
#
# Purpose
# -------
# Generate closed-form basis functions and derivatives for the catalog defined in
# `basis_descriptions.jl`, and write them to `lagrange_generated.jl`.
# (Note: Will be renamed to `basis_generated.jl` once migration is complete)
#
# Design
# ------
# - Topology and basis family/order are provided by each description.
# - Reference coordinates are taken from `reference_coordinates(topology())`.
# - A polynomial ansatz defines the Vandermonde system; interpolation polynomials
#   and their derivatives are derived symbolically.
# - Generated functions follow the new API:
#     get_basis_functions(::Type{Topo}, ::Type{Basis}, xi::Vec) → SVector{N, Float64}
#     get_basis_derivatives(::Type{Topo}, ::Type{Basis}, xi::Vec) → SVector{N, Vec{D}}
#
# Return Types
# ------------
# - Basis functions: SVector{N, Float64} (enables dot products, vector operations)
# - Derivatives: SVector{N, Vec{D, Float64}} (each gradient is a Vec from Tensors.jl)
# - Why SVector? Natural linear algebra: dot(coeffs, N), sum(values[i] * dN[i], ...)
#
# Usage
# -----
#   julia --project=. src/basis/basis_generator.jl
# ==============================================================================

__precompile__(false)

using LinearAlgebra
using Tensors

# Type alias for coordinate inputs (tuples or Vec)
const Vecish{N,T} = Union{NTuple{N,T},Vec{N,T}}

# ------------------------------------------------------------------------------
# Symbolic helpers (minimal, dependency-free)
# ------------------------------------------------------------------------------

differentiate(::Number, ::Symbol) = 0
differentiate(f::Symbol, x::Symbol) = f == x ? 1 : 0

function differentiate(f::Expr, x::Symbol)
    @assert f.head == :call
    op = first(f.args)
    if op == :*
        res_args = Any[:+]
        for i in 2:length(f.args)
            new_args = copy(f.args)
            new_args[i] = differentiate(f.args[i], x)
            push!(res_args, Expr(:call, new_args...))
        end
        return Expr(:call, res_args...)
    elseif op == :^
        _, f_inner, a = f.args
        df = differentiate(f_inner, x)
        return :($a * $f_inner^($a - 1) * $df)
    elseif op == :+
        args = differentiate.(f.args[2:end], x)
        return length(args) == 1 ? first(args) : Expr(:call, :+, args...)
    elseif op == :-
        args = differentiate.(f.args[2:end], x)
        return length(args) == 1 ? first(args) : Expr(:call, :-, args...)
    elseif op == :/
        _, g, h = f.args
        dg = differentiate(g, x)
        dh = differentiate(h, x)
        return :(($dg * $h - $g * $dh) / $h^2)
    else
        error("Unsupported operation in differentiate: $op")
    end
end

# ------------------------------------------------------------------------------
# Numerical filtering and rounding
# ------------------------------------------------------------------------------

"""
    round_coefficient(x::Float64; tol=1e-12, frac_tol=1e-7) -> Float64

Round floating point coefficient to remove numerical noise:
1. If |x| < tol, return 0.0
2. If x ≈ n/d for small integers n,d (common fractions), return exact fraction
3. Otherwise return x
"""
function round_coefficient(x::Float64; tol=1e-12, frac_tol=1e-7)
    # Filter out near-zero values
    if abs(x) < tol
        return 0.0
    end

    # Check common fractions up to denominator 20 (covers 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/18, etc.)
    for denom in 1:20
        for numer in -20*denom:20*denom
            frac = numer / denom
            if abs(x - frac) < frac_tol
                return frac
            end
        end
    end

    return x
end

"""
    filter_expr(expr) -> expr

Recursively walk expression tree and apply round_coefficient to all Float64 literals.
Also eliminate terms with zero coefficients.
"""
function filter_expr(expr)
    if isa(expr, Float64)
        return round_coefficient(expr)
    elseif isa(expr, Expr)
        filtered_args = [filter_expr(arg) for arg in expr.args]
        new_expr = Expr(expr.head, filtered_args...)

        # Eliminate zero terms
        if expr.head == :call
            op = filtered_args[1]

            # Remove 0.0 * anything
            if op == :* && length(filtered_args) >= 3
                if 0.0 in filtered_args[2:end]
                    return 0.0
                end
            end

            # Remove x + 0.0 or 0.0 + x
            if op == :+ && length(filtered_args) >= 3
                nonzero_args = filter(arg -> !(isa(arg, Number) && arg == 0.0), filtered_args[2:end])
                if isempty(nonzero_args)
                    return 0.0
                elseif length(nonzero_args) == 1
                    return nonzero_args[1]
                else
                    return Expr(:call, :+, nonzero_args...)
                end
            end
        end

        return new_expr
    else
        return expr
    end
end

# ------------------------------------------------------------------------------
# Symbolic simplification
# ------------------------------------------------------------------------------

simplify(f::Union{Number,Symbol}) = f

function simplify(ex::Expr)
    @assert ex.head == :call
    op = first(ex.args)
    if op == :*
        args = simplify.(ex.args[2:end])
        0 in args && return 0
        filter!(k -> !(isa(k, Number) && k == 1), args)

        # Constant folding: multiply all numeric values together
        # Also handle nested multiplication: 0.5 * (2u) → (0.5 * 2) * u → 1.0u
        numeric_product = 1.0
        non_numeric_args = []
        for arg in args
            if arg isa Number
                numeric_product *= arg
            elseif arg isa Expr && arg.head == :call && arg.args[1] == :* && length(arg.args) >= 3
                # Flatten nested multiplication
                for nested_arg in arg.args[2:end]
                    if nested_arg isa Number
                        numeric_product *= nested_arg
                    else
                        push!(non_numeric_args, nested_arg)
                    end
                end
            else
                push!(non_numeric_args, arg)
            end
        end

        # Build result
        if numeric_product == 0
            return 0
        elseif isempty(non_numeric_args)
            return numeric_product
        elseif numeric_product == 1
            return length(non_numeric_args) == 1 ? first(non_numeric_args) : Expr(:call, :*, non_numeric_args...)
        else
            return length(non_numeric_args) == 0 ? numeric_product :
                   length(non_numeric_args) == 1 ? Expr(:call, :*, numeric_product, first(non_numeric_args)) :
                   Expr(:call, :*, numeric_product, non_numeric_args...)
        end
    elseif op == :+
        args = simplify.(ex.args[2:end])
        filter!(k -> !(isa(k, Number) && k == 0), args)

        # Constant folding: add all numeric values together
        numeric_sum = 0.0
        non_numeric_args = []
        for arg in args
            if arg isa Number
                numeric_sum += arg
            else
                push!(non_numeric_args, arg)
            end
        end

        # Build result
        if isempty(non_numeric_args)
            return numeric_sum
        elseif numeric_sum == 0
            return length(non_numeric_args) == 0 ? 0 :
                   length(non_numeric_args) == 1 ? first(non_numeric_args) :
                   Expr(:call, :+, non_numeric_args...)
        else
            all_args = [numeric_sum; non_numeric_args]
            return length(all_args) == 1 ? first(all_args) : Expr(:call, :+, all_args...)
        end
    elseif op == :-
        args = simplify.(ex.args[2:end])
        filter!(k -> !(isa(k, Number) && k == 0), args)
        return length(args) == 0 ? 0 : length(args) == 1 ? first(args) : Expr(:call, :-, args...)
    elseif op == :^
        # Simplify power expressions: x^1 -> x, x^0 -> 1, evaluate constant^constant
        base = simplify(ex.args[2])
        exp = simplify(ex.args[3])

        # Evaluate constant arithmetic in exponent
        if exp isa Expr
            try
                exp_val = @eval $exp
                if exp_val isa Number
                    exp = exp_val
                end
            catch
            end
        end

        # Simplify based on exponent value
        exp == 0 && return 1
        exp == 1 && return base

        # If both are numbers, evaluate
        if base isa Number && exp isa Number
            return base^exp
        end

        return Expr(:call, :^, base, exp)
    else
        return Expr(:call, op, simplify.(ex.args[2:end])...)
    end
end

# Convert common decimal fractions to fraction notation with explicit multiplication
function decimal_to_fraction(s::String)
    # Map of common fractions to their decimal representations
    fractions = [
        ("0.5", "1/2"),
        ("0.3333333333333333", "1/3"),
        ("0.6666666666666666", "2/3"),
        ("0.25", "1/4"),
        ("0.75", "3/4"),
        ("0.2", "1/5"),
        ("0.4", "2/5"),
        ("0.6", "3/5"),
        ("0.8", "4/5"),
        ("0.16666666666666666", "1/6"),
        ("0.8333333333333334", "5/6"),
        ("0.125", "1/8"),
        ("0.375", "3/8"),
        ("0.625", "5/8"),
        ("0.875", "7/8"),
        ("0.1111111111111111", "1/9"),
        ("0.2222222222222222", "2/9"),
        ("0.08333333333333333", "1/12"),
        ("0.05555555555555555", "1/18"),
        ("0.2777777777777778", "5/18"),
    ]

    result = s
    for (decimal, fraction) in fractions
        # Pattern: decimal followed by variable (letter or underscore) - needs explicit *
        result = replace(result, Regex("\\b$(decimal)([a-zA-Z_])") => SubstitutionString("$(fraction) * \\1"))

        # Pattern: negative decimal followed by variable - needs explicit *
        neg_decimal = "-" * decimal
        neg_fraction = "-" * fraction
        result = replace(result, Regex("$(neg_decimal)([a-zA-Z_])") => SubstitutionString("$(neg_fraction) * \\1"))

        # Pattern: decimal not followed by variable (at end or before operator) - can stay as fraction
        result = replace(result, Regex("\\b$(decimal)(?![a-zA-Z_0-9])") => fraction)
        result = replace(result, Regex("$(neg_decimal)(?![a-zA-Z_0-9])") => neg_fraction)
    end

    return result
end

# Simplify float coefficients: 4.0 → 4, -2.0 → -2, etc.
function simplify_float_coefficients(s::String)
    # First pass: Match floats with .0 followed by space, operator, comma, or parenthesis
    # Pattern: number.0 followed by space/*/+-/^/)/,/end
    result = replace(s, r"(\d+)\.0(?=\s|\*|\+|-|\^|\)|,|$)" => s"\1")
    # Handle negative floats: -4.0 → -4
    result = replace(result, r"(-\d+)\.0(?=\s|\*|\+|-|\^|\)|,|$)" => s"\1")

    # Second pass: Match floats like "3.0u" or "4.0v" and add explicit multiplication
    # Pattern: number.0 followed directly by a variable (letter/underscore)
    result = replace(result, r"(\d+)\.0([a-zA-Z_])" => s"\1 * \2")
    # Handle negative: -3.0u → -3 * u
    result = replace(result, r"(-\d+)\.0([a-zA-Z_])" => s"\1 * \2")

    return result
end

# Prettify expression string: convert "+ -X" to "- X" for better readability
function prettify_expr(s::String)
    # Strip whitespace first
    s = strip(s)

    # Skip empty lines and block keywords
    s == "begin" && return ""
    s == "end" && return ""

    # Replace "+ -" with " - " (with proper spacing)
    s = replace(s, r"\+\s*-" => " - ")
    # Convert common decimal fractions to fraction notation
    s = decimal_to_fraction(s)
    # Simplify float coefficients (4.0 → 4)
    s = simplify_float_coefficients(s)

    # Remove coefficient 1: "1 * u" → "u", "-1 * u" → "-u"
    s = replace(s, r"\b1 \* " => "")
    s = replace(s, r"-1 \* " => "-")

    # Remove leading unary plus
    # Case 1: "+(expr)" → "expr"
    if startswith(s, "+(") && endswith(s, ")")
        s = s[3:end-1]
    end
    # Case 2: "+variable" at start → "variable"
    if match(r"^\+[a-zA-Z_]", s) !== nothing
        s = s[2:end]
    end
    # Case 3: "= +variable" → "= variable" (after equals sign)
    s = replace(s, r"=\s*\+([a-zA-Z_])" => s"= \1")
    # Case 4: "= +(expr)" → "= expr"
    s = replace(s, r"=\s*\+\(" => "= (")

    # Remove double spaces: "1  -" → "1 -"
    s = replace(s, r"  +" => " ")

    # Remove unnecessary parentheses around single expressions after =
    # Pattern: "= (expr)" where expr doesn't contain operators at top level that need parens
    # Case 1: "= (variable)" → "= variable"
    s = replace(s, r"= \(([a-zA-Z_][a-zA-Z_0-9]*)\)" => s"= \1")
    # Case 2: "= (-variable)" → "= -variable"  
    s = replace(s, r"= \((-[a-zA-Z_][a-zA-Z_0-9]*)\)" => s"= \1")
    # Case 3: "= (expr)" where expr is a simple product or sum without nested parens
    # This catches cases like "= (4 * u * v)" → "= 4 * u * v"
    s = replace(s, r"= \(([^()]+)\)$" => s"= \1")

    return s
end

# ------------------------------------------------------------------------------
# Polynomial assembly (Vandermonde approach)
# ------------------------------------------------------------------------------
# Constructs basis functions by solving Vandermonde systems.
# Works for Lagrange and Serendipity families with polynomial ansatz.
# Other basis families (hierarchical, H(curl), etc.) can be added directly.

function vandermonde_matrix(p::Expr, X::Vector{<:Vecish{D}}) where D
    @assert p.head == :call && first(p.args) == :+
    terms = p.args[2:end]
    V = zeros(length(X), length(terms))
    for (i, xi) in enumerate(X)
        if D == 1
            u = xi[1]
            for (j, term) in enumerate(terms)
                V[i, j] = @eval let u = $u
                    $term
                end
            end
        elseif D == 2
            u, v = xi[1], xi[2]
            for (j, term) in enumerate(terms)
                V[i, j] = @eval let u = $u, v = $v
                    $term
                end
            end
        else
            u, v, w = xi[1], xi[2], xi[3]
            for (j, term) in enumerate(terms)
                V[i, j] = @eval let u = $u, v = $v, w = $w
                    $term
                end
            end
        end
    end
    return V
end

function calculate_interpolation_polynomials(p::Expr, V::Matrix)
    @assert first(p.args) == :+
    args = p.args[2:end]
    n = size(V, 1)
    basis = Vector{Expr}()
    for i in 1:n
        b = zeros(n)
        b[i] = 1.0
        coeffs = V \ b
        N = Expr(:call, :+)
        for (ai, bi) in zip(coeffs, args)
            isapprox(ai, 0.0) && continue
            push!(N.args, simplify(:($ai * $bi)))
        end
        push!(basis, N)
    end
    return basis
end

function calculate_interpolation_polynomial_derivatives(basis::Vector{Expr}, D::Int)
    vars = (:u, :v, :w)
    dbasis = Matrix{Any}(undef, D, length(basis))
    for (i, N) in enumerate(basis), j in 1:D
        dbasis[j, i] = simplify(differentiate(N, vars[j]))
    end
    return dbasis
end

# ------------------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------------------

function create_basis_code(topology_type_expr, basis_type_expr::Type, description, X::Vector{<:Vecish{D}}, ansatz::Expr) where D
    V = vandermonde_matrix(ansatz, X)
    basis = calculate_interpolation_polynomials(ansatz, V)
    dbasis = calculate_interpolation_polynomial_derivatives(basis, D)
    N = length(X)

    # Apply numerical filtering to remove noise from floating point arithmetic
    basis = [filter_expr(b) for b in basis]
    dbasis = [filter_expr(db) for db in dbasis]

    unpack = D == 1 ? :((u,) = xi) : D == 2 ? :((u, v) = xi) : :((u, v, w) = xi)

    # Build function body with individual variable assignments for readability
    func_body = Expr(:block)
    push!(func_body.args, unpack)

    # Add individual basis function assignments: N1 = ..., N2 = ..., etc.
    for (i, expr) in enumerate(basis)
        push!(func_body.args, :($(Symbol("N", i)) = $expr))
    end
    push!(func_body.args, :(return SVector{$N,T}($([Symbol("N", i) for i in 1:N]...))))

    # Build derivative function body
    dfunc_body = Expr(:block)
    push!(dfunc_body.args, unpack)

    # Add individual gradient assignments: dN1 = Vec{D,T}(...), dN2 = ..., etc.
    for i = 1:N
        dvec_expr = :(Vec{$D,T}(($(dbasis[:, i]...),)))
        push!(dfunc_body.args, :($(Symbol("dN", i)) = $dvec_expr))
    end
    push!(dfunc_body.args, :(return SVector{$N,Vec{$D,T}}($([Symbol("dN", i) for i in 1:N]...))))

    # Remove line numbers from generated code
    func_body = Base.remove_linenums!(func_body)
    dfunc_body = Base.remove_linenums!(dfunc_body)

    # Splat the body contents directly (avoid nested begin/end)
    return quote
        @inline function get_basis_functions(::$topology_type_expr, ::$basis_type_expr, xi::Vec{$D,T}) where T
            $(func_body.args...)
        end

        @inline function get_basis_derivatives(::$topology_type_expr, ::$basis_type_expr, xi::Vec{$D,T}) where T
            $(dfunc_body.args...)
        end
    end
end

function generate_basis_code(desc)
    topology_type_expr = desc.topology
    basis_type_expr = desc.family
    coords = collect(reference_coordinates(topology_type_expr()))  # Already Vec objects
    D = length(coords[1])
    ansatz_expr = length(desc.ansatz) == 1 ? desc.ansatz[1] : Expr(:call, :+, desc.ansatz...)
    create_basis_code(topology_type_expr, basis_type_expr, desc.description, coords, ansatz_expr)
end

function write_generated_file(elements)
    output = IOBuffer()
    println(output, "# This file is a part of JuliaFEM.")
    println(output, "# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE")
    println(output)
    println(output, "# ============================================================================")
    println(output, "# AUTO-GENERATED BASIS FUNCTIONS (All Families)")
    println(output, "# ============================================================================")
    println(output, "# Generated by: src/basis/basis_generator.jl")
    println(output, "# WARNING: DO NOT EDIT MANUALLY")
    println(output, "#")
    println(output, "# This file contains generated basis function implementations for:")
    println(output, "#   - Lagrange bases (standard nodal interpolation)")
    println(output, "#   - Serendipity bases (reduced tensor-product for quads/hexes)")
    println(output, "#   - Future: Hierarchical, modal, and exotic basis families")
    println(output, "#")
    println(output, "# Return Types:")
    println(output, "#   get_basis_functions → SVector{N, Float64}")
    println(output, "#   get_basis_derivatives → SVector{N, Vec{D, Float64}}")
    println(output, "#")
    println(output, "# Why SVector? Enables natural vector operations:")
    println(output, "#   u_interp = dot(node_values, N)")
    println(output, "#   grad_u = sum(node_values[i] * dN[i] for i in 1:N)")

    for (i, elem) in enumerate(elements)
        println("[$i/$(length(elements))] Generating $(elem.name)...")
        code_expr = generate_basis_code(elem)

        println(output)
        println(output, "# " * "─"^78)
        println(output, "# $(elem.family) on $(elem.topology): $(elem.description)")
        println(output, "# (legacy name: $(elem.name))")
        println(output)

        # Pretty print each function manually
        for expr in code_expr.args
            # Skip line numbers at top level
            if expr isa LineNumberNode
                continue
            end

            # Extract function components
            if expr isa Expr && expr.head == :macrocall && expr.args[1] == Symbol("@inline")
                func = expr.args[3]
                func_sig = func.args[1]
                func_body = func.args[2]

                # Handle where clause properly
                if func_sig isa Expr && func_sig.head == :where
                    # func_sig is like :(f(args...) where T)
                    inner_sig = func_sig.args[1]
                    type_vars = func_sig.args[2:end]
                    println(output, "@inline function $(inner_sig.args[1])($(join(inner_sig.args[2:end], ", "))) where $(join(type_vars, ", "))")
                else
                    # No where clause
                    println(output, "@inline function $(func_sig.args[1])($(join(func_sig.args[2:end], ", ")))")
                end

                # Print function body lines (skip the :block wrapper)
                if func_body isa Expr && func_body.head == :block
                    for line in func_body.args
                        if line isa LineNumberNode
                            continue
                        elseif line isa Expr || line isa Symbol || line isa Number
                            line_str = string(line)
                            line_str = prettify_expr(line_str)
                            if !isempty(line_str)  # Skip empty lines (filtered begin/end)
                                println(output, "    ", line_str)
                            end
                        end
                    end
                end

                println(output, "end")
                println(output)  # Blank line between functions
            end
        end
    end

    open(joinpath(@__DIR__, "basis_generated.jl"), "w") do io
        write(io, String(take!(output)))
    end

    println("Generated $(length(elements)) basis families")
    outpath = joinpath(@__DIR__, "basis_generated.jl")
    println("Output: $outpath")
end

# ------------------------------------------------------------------------------
# Script entry point (STANDALONE - does not require JuliaFEM to be loaded)
# ------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using StaticArrays: SVector

    # Load topology and basis definitions WITHOUT loading JuliaFEM
    # This breaks the circular dependency!
    include("../topology/api.jl")  # Defines AbstractTopology
    include("../topology/segments.jl")
    include("../topology/triangles.jl")
    include("../topology/quadrilaterals.jl")
    include("../topology/tetrahedra.jl")
    include("../topology/hexahedra.jl")
    include("../topology/pyramids.jl")
    include("../topology/wedges.jl")

    include("api.jl")  # Defines VandermondeBasisDescription, Lagrange, Serendipity
    include("basis_descriptions.jl")  # provides BASIS_DESCRIPTIONS
    println("\n" * "="^80)
    println("BASIS GENERATOR")
    println("="^80)
    println("Loaded $(length(BASIS_DESCRIPTIONS)) basis descriptions\n")

    write_generated_file(BASIS_DESCRIPTIONS)

    println("\n" * "="^80)
    println("Generation complete!")
    println("Output: src/basis/basis_generated.jl")
    println("Return types: SVector{N, Float64} and SVector{N, Vec{D}}")
    println("="^80 * "\n")
end
