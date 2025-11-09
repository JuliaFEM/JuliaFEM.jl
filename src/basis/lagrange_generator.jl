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

# Type alias for coordinate types (works both when included and run as script)
if !@isdefined(Vecish)
    if !@isdefined(Vec)
        # Running as standalone script - need Tensors.Vec
        using Tensors
    end
    const Vecish{N,T} = Union{NTuple{N,T},Vec{N,T}}
end

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

# Vandermonde matrix construction for polynomial basis
function vandermonde_matrix(p::Expr, X::Vector{<:Vecish{D}}) where D
    vars = [:u, :v, :w]
    @assert p.head == :call
    @assert first(p.args) == :+
    terms = p.args[2:end]
    n = length(X)
    m = length(terms)
    V = zeros(n, m)
    for (i, xi) in enumerate(X)
        for (j, term) in enumerate(terms)
            # Evaluate term at xi
            if D == 1
                u = xi[1]
                V[i, j] = @eval let u = $u
                    $term
                end
            elseif D == 2
                u, v = xi[1], xi[2]
                V[i, j] = @eval let u = $u, v = $v
                    $term
                end
            else
                u, v, w = xi[1], xi[2], xi[3]
                V[i, j] = @eval let u = $u, v = $v, w = $w
                    $term
                end
            end
        end
    end
    return V
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

function create_basis(topology_type::Symbol, polynomial_degree::Int, description, X::Vector{<:Vecish{D}}, p::Expr) where D
    @debug "create basis given ansatz polynomial" topology_type polynomial_degree description X p
    V = vandermonde_matrix(p, X)
    basis = calculate_interpolation_polynomials(p, V)
    return create_basis(topology_type, polynomial_degree, description, X, basis)
end

function create_basis(topology_type::Symbol, polynomial_degree::Int, description, X::Vector{<:Vecish{D}}, basis::Vector) where D
    @assert length(X) == length(basis)
    @debug "create basis given basis functions" topology_type polynomial_degree description X basis
    dbasis = calculate_interpolation_polynomial_derivatives(basis, D)
    return create_basis(topology_type, polynomial_degree, description, Vec.(X), basis, dbasis)
end

function create_basis(topology_type::Symbol, polynomial_degree::Int, description, X::Vector{<:Vecish{D,T}}, basis, dbasis) where {D,T}
    N = length(X)
    @debug "create basis given basis functions and derivatives" topology_type polynomial_degree description X basis dbasis

    # Build tuple expression for eval_basis! return: (N1, N2, N3, ...)
    basis_tuple_args = [basis[i] for i = 1:N]
    basis_tuple = Expr(:tuple, basis_tuple_args...)

    # Build tuple expression for eval_dbasis! return: (dN1, dN2, dN3, ...)
    dbasis_tuple_args = [:(Vec(float.(tuple($(dbasis[:, i]...))))) for i = 1:N]
    dbasis_tuple = Expr(:tuple, dbasis_tuple_args...)
    
    # Build tuple expression for reference coordinates: (Vec(...), Vec(...), ...)
    coord_tuple_args = [:(Vec{$D,$T}(tuple($(X[i]...)))) for i = 1:N]
    coord_tuple = Expr(:tuple, coord_tuple_args...)

    if D == 1
        unpack = :((u,) = xi)
    elseif D == 2
        unpack = :((u, v) = xi)
    else
        unpack = :((u, v, w) = xi)
    end

    # Generate code for Lagrange{Topology, P} instead of TopologyBasis
    code = quote
        # $description
        function get_reference_element_coordinates(::Type{Lagrange{$topology_type,$polynomial_degree}})
            return $coord_tuple
        end
        
        function get_reference_element_coordinates(::Lagrange{$topology_type,$polynomial_degree})
            return $coord_tuple
        end

        # Return tuple directly - zero allocations!
        @inline function eval_basis!(::Type{Lagrange{$topology_type,$polynomial_degree}}, ::Type{T}, xi::Vec) where T
            $unpack
            @inbounds return $basis_tuple
        end
        
        @inline function eval_basis!(::Lagrange{$topology_type,$polynomial_degree}, ::Type{T}, xi::Vec) where T
            $unpack
            @inbounds return $basis_tuple
        end

        # Return NTuple{N,Vec{D}} directly - zero allocations!
        @inline function eval_dbasis!(::Type{Lagrange{$topology_type,$polynomial_degree}}, xi::Vec)
            $unpack
            @inbounds return $dbasis_tuple
        end
        
        @inline function eval_dbasis!(::Lagrange{$topology_type,$polynomial_degree}, xi::Vec)
            $unpack
            @inbounds return $dbasis_tuple
        end
    end
    return code
end

create_basis_and_eval(args...) = eval(create_basis(args...))

# Mapping from old element names to (topology_type, polynomial_degree)
# This allows the generator to use the new parametric Lagrange{T,P} system
const ELEMENT_TO_LAGRANGE = Dict{String, Tuple{Symbol, Int}}(
    "Seg2" => (:Segment, 1),
    "Seg3" => (:Segment, 2),
    "Tri3" => (:Triangle, 1),
    "Tri6" => (:Triangle, 2),
    "Quad4" => (:Quadrilateral, 1),
    "Quad8" => (:Quadrilateral, 2),  # Serendipity, special case
    "Quad9" => (:Quadrilateral, 2),
    "Tet4" => (:Tetrahedron, 1),
    "Tet10" => (:Tetrahedron, 2),
    "Hex8" => (:Hexahedron, 1),
    "Hex20" => (:Hexahedron, 2),  # Serendipity, special case
    "Hex27" => (:Hexahedron, 2),
    "Pyr5" => (:Pyramid, 1),
    "Wedge6" => (:Wedge, 1),
    "Wedge15" => (:Wedge, 2),  # Serendipity, special case
)

# ==============================================================================
# GENERATION SCRIPT - Run as: julia --project=. src/basis/lagrange_generator.jl
# ==============================================================================
#
# This script portion executes only when this file is run directly (not included).
# It generates all 15 standard Lagrange basis types and writes them to
# src/basis/lagrange_generated.jl
#
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "../.."))

    using Dates

    println("="^80)
    println("LAGRANGE BASIS FUNCTION GENERATOR")
    println("="^80)
    println()
    println("Loading symbolic generator...")
    println("✓ Generator loaded")
    println()

    # =========================================================================
    # ELEMENT CATALOG
    # =========================================================================

    """
        ElementDescription

    Defines a Lagrange finite element for basis function generation.

    # Fields
    - `name::String`: Element type name (e.g., "Seg2", "Tri3", "Quad4")
    - `description::String`: Human-readable description
    - `coordinates::Vector{Vector{Float64}}`: Node coordinates in reference element
    - `ansatz::Vector{Any}`: Polynomial terms for Vandermonde system (expressions or literals)
    """
    struct ElementDescription
        name::String
        description::String
        coordinates::Vector{Vector{Float64}}
        ansatz::Vector{Any}

        # Constructor accepting keyword arguments for readability
        function ElementDescription(; name, description, coordinates, ansatz)
            new(name, description, coordinates, ansatz)
        end
    end

    elements = ElementDescription[]

    # ─────────────────────────────────────────────────────────────────────────
    # 1D ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Seg2: 2-node linear segment
    push!(elements, ElementDescription(
        name="Seg2",
        description="2-node linear segment element",
        coordinates=[
            [-1.0],  # Node 1
            [1.0]   # Node 2
        ],
        ansatz=[
            :(1),    # 1
            :(u)     # u
        ]
    ))

    # Seg3: 3-node quadratic segment
    push!(elements, ElementDescription(
        name="Seg3",
        description="3-node quadratic segment element",
        coordinates=[
            [-1.0],  # Node 1
            [1.0],  # Node 2
            [0.0]   # Node 3 (midpoint)
        ],
        ansatz=[
            :(1),    # 1
            :(u),    # u
            :(u^2)   # u²
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2D TRIANGULAR ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Tri3: 3-node linear triangle
    push!(elements, ElementDescription(
        name="Tri3",
        description="3-node linear triangular element",
        coordinates=[
            [0.0, 0.0],  # Node 1 (origin)
            [1.0, 0.0],  # Node 2 (u-axis)
            [0.0, 1.0]   # Node 3 (v-axis)
        ],
        ansatz=[
            :(1),    # 1
            :(u),    # u
            :(v)     # v
        ]
    ))

    # Tri6: 6-node quadratic triangle
    push!(elements, ElementDescription(
        name="Tri6",
        description="6-node quadratic triangular element",
        coordinates=[
            [0.0, 0.0],  # Node 1 (vertex)
            [1.0, 0.0],  # Node 2 (vertex)
            [0.0, 1.0],  # Node 3 (vertex)
            [0.5, 0.0],  # Node 4 (edge 1-2)
            [0.5, 0.5],  # Node 5 (edge 2-3)
            [0.0, 0.5]   # Node 6 (edge 3-1)
        ],
        ansatz=[
            :(1),    # 1
            :(u), :(v),         # linear
            :(u^2), :(u * v), :(v^2)  # quadratic
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2D QUADRILATERAL ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Quad4: 4-node bilinear quadrilateral
    push!(elements, ElementDescription(
        name="Quad4",
        description="4-node bilinear quadrilateral element",
        coordinates=[
            [-1.0, -1.0],  # Node 1
            [1.0, -1.0],  # Node 2
            [1.0, 1.0],   # Node 3
            [-1.0, 1.0]    # Node 4
        ],
        ansatz=[
            :(1),        # 1
            :(u), :(v),  # linear
            :(u * v)     # bilinear
        ]
    ))

    # Quad8: 8-node serendipity quadrilateral
    push!(elements, ElementDescription(
        name="Quad8",
        description="8-node serendipity quadrilateral element",
        coordinates=[
            [-1.0, -1.0],  # Node 1 (vertex)
            [1.0, -1.0],  # Node 2 (vertex)
            [1.0, 1.0],   # Node 3 (vertex)
            [-1.0, 1.0],   # Node 4 (vertex)
            [0.0, -1.0],  # Node 5 (edge 1-2)
            [1.0, 0.0],   # Node 6 (edge 2-3)
            [0.0, 1.0],   # Node 7 (edge 3-4)
            [-1.0, 0.0]    # Node 8 (edge 4-1)
        ],
        ansatz=[
            :(1),                          # 1
            :(u), :(v),                    # linear
            :(u^2), :(u * v), :(v^2),      # quadratic
            :(u^2 * v), :(u * v^2)         # serendipity (no u²v²)
        ]
    ))

    # Quad9: 9-node biquadratic quadrilateral
    push!(elements, ElementDescription(
        name="Quad9",
        description="9-node biquadratic quadrilateral element",
        coordinates=[
            [-1.0, -1.0],  # Node 1 (vertex)
            [1.0, -1.0],  # Node 2 (vertex)
            [1.0, 1.0],   # Node 3 (vertex)
            [-1.0, 1.0],   # Node 4 (vertex)
            [0.0, -1.0],  # Node 5 (edge 1-2)
            [1.0, 0.0],   # Node 6 (edge 2-3)
            [0.0, 1.0],   # Node 7 (edge 3-4)
            [-1.0, 0.0],   # Node 8 (edge 4-1)
            [0.0, 0.0]    # Node 9 (center)
        ],
        ansatz=[
            :(1),                                # 1
            :(u), :(v),                          # linear
            :(u^2), :(u * v), :(v^2),            # quadratic
            :(u^2 * v), :(u * v^2), :(u^2 * v^2) # biquadratic
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 3D TETRAHEDRAL ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Tet4: 4-node linear tetrahedron
    push!(elements, ElementDescription(
        name="Tet4",
        description="4-node linear tetrahedral element",
        coordinates=[
            [0.0, 0.0, 0.0],  # Node 1 (origin)
            [1.0, 0.0, 0.0],  # Node 2 (u-axis)
            [0.0, 1.0, 0.0],  # Node 3 (v-axis)
            [0.0, 0.0, 1.0]   # Node 4 (w-axis)
        ],
        ansatz=[
            :(1),           # 1
            :(u), :(v), :(w)  # linear
        ]
    ))

    # Tet10: 10-node quadratic tetrahedron
    push!(elements, ElementDescription(
        name="Tet10",
        description="10-node quadratic tetrahedral element",
        coordinates=[
            [0.0, 0.0, 0.0],  # Node 1 (vertex)
            [1.0, 0.0, 0.0],  # Node 2 (vertex)
            [0.0, 1.0, 0.0],  # Node 3 (vertex)
            [0.0, 0.0, 1.0],  # Node 4 (vertex)
            [0.5, 0.0, 0.0],  # Node 5 (edge 1-2)
            [0.5, 0.5, 0.0],  # Node 6 (edge 2-3)
            [0.0, 0.5, 0.0],  # Node 7 (edge 3-1)
            [0.0, 0.0, 0.5],  # Node 8 (edge 1-4)
            [0.5, 0.0, 0.5],  # Node 9 (edge 2-4)
            [0.0, 0.5, 0.5]   # Node 10 (edge 3-4)
        ],
        ansatz=[
            :(1),                                      # 1
            :(u), :(v), :(w),                          # linear
            :(u^2), :(v^2), :(w^2),                    # pure quadratic
            :(u * v), :(u * w), :(v * w)               # bilinear
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 3D HEXAHEDRAL ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Hex8: 8-node trilinear hexahedron
    push!(elements, ElementDescription(
        name="Hex8",
        description="8-node trilinear hexahedral element",
        coordinates=[
            [-1.0, -1.0, -1.0],  # Node 1
            [1.0, -1.0, -1.0],  # Node 2
            [1.0, 1.0, -1.0],   # Node 3
            [-1.0, 1.0, -1.0],   # Node 4
            [-1.0, -1.0, 1.0],   # Node 5
            [1.0, -1.0, 1.0],   # Node 6
            [1.0, 1.0, 1.0],    # Node 7
            [-1.0, 1.0, 1.0]     # Node 8
        ],
        ansatz=[
            :(1),                        # 1
            :(u), :(v), :(w),            # linear
            :(u * v), :(u * w), :(v * w),  # bilinear
            :(u * v * w)                 # trilinear
        ]
    ))

    # Hex20: 20-node serendipity hexahedron
    push!(elements, ElementDescription(
        name="Hex20",
        description="20-node serendipity hexahedral element",
        coordinates=[
            [-1.0, -1.0, -1.0],  # Vertex 1
            [1.0, -1.0, -1.0],  # Vertex 2
            [1.0, 1.0, -1.0],   # Vertex 3
            [-1.0, 1.0, -1.0],   # Vertex 4
            [-1.0, -1.0, 1.0],   # Vertex 5
            [1.0, -1.0, 1.0],   # Vertex 6
            [1.0, 1.0, 1.0],    # Vertex 7
            [-1.0, 1.0, 1.0],    # Vertex 8
            [0.0, -1.0, -1.0],  # Edge midpoint
            [1.0, 0.0, -1.0],   # Edge midpoint
            [0.0, 1.0, -1.0],   # Edge midpoint
            [-1.0, 0.0, -1.0],   # Edge midpoint
            [-1.0, -1.0, 0.0],   # Edge midpoint
            [1.0, -1.0, 0.0],   # Edge midpoint
            [1.0, 1.0, 0.0],    # Edge midpoint
            [-1.0, 1.0, 0.0],    # Edge midpoint
            [0.0, -1.0, 1.0],   # Edge midpoint
            [1.0, 0.0, 1.0],    # Edge midpoint
            [0.0, 1.0, 1.0],    # Edge midpoint
            [-1.0, 0.0, 1.0]     # Edge midpoint
        ],
        ansatz=[
            :(1),                                                  # 1
            :(u), :(v), :(w),                                      # linear
            :(u^2), :(v^2), :(w^2),                                # quadratic
            :(u * v), :(u * w), :(v * w),                          # bilinear
            :(u^2 * v), :(u^2 * w), :(v^2 * u), :(v^2 * w), :(w^2 * u), :(w^2 * v),  # serendipity
            :(u * v * w), :(u^2 * v * w), :(u * v^2 * w), :(u * v * w^2)  # trilinear + serendipity
        ]
    ))

    # Hex27: 27-node triquadratic hexahedron
    push!(elements, ElementDescription(
        name="Hex27",
        description="27-node triquadratic hexahedral element",
        coordinates=[
            [-1.0, -1.0, -1.0],  # Vertex 1
            [1.0, -1.0, -1.0],  # Vertex 2
            [1.0, 1.0, -1.0],   # Vertex 3
            [-1.0, 1.0, -1.0],   # Vertex 4
            [-1.0, -1.0, 1.0],   # Vertex 5
            [1.0, -1.0, 1.0],   # Vertex 6
            [1.0, 1.0, 1.0],    # Vertex 7
            [-1.0, 1.0, 1.0],    # Vertex 8
            [0.0, -1.0, -1.0],  # Edge midpoint
            [1.0, 0.0, -1.0],   # Edge midpoint
            [0.0, 1.0, -1.0],   # Edge midpoint
            [-1.0, 0.0, -1.0],   # Edge midpoint
            [-1.0, -1.0, 0.0],   # Edge midpoint
            [1.0, -1.0, 0.0],   # Edge midpoint
            [1.0, 1.0, 0.0],    # Edge midpoint
            [-1.0, 1.0, 0.0],    # Edge midpoint
            [0.0, -1.0, 1.0],   # Edge midpoint
            [1.0, 0.0, 1.0],    # Edge midpoint
            [0.0, 1.0, 1.0],    # Edge midpoint
            [-1.0, 0.0, 1.0],    # Edge midpoint
            [0.0, 0.0, -1.0],   # Face center
            [0.0, 0.0, 1.0],    # Face center
            [0.0, -1.0, 0.0],   # Face center
            [1.0, 0.0, 0.0],    # Face center
            [0.0, 1.0, 0.0],    # Face center
            [-1.0, 0.0, 0.0],    # Face center
            [0.0, 0.0, 0.0]     # Volume center
        ],
        ansatz=[
            :(1),                                    # 1
            :(u), :(v), :(w),                        # linear (3)
            :(u^2), :(v^2), :(w^2),                  # quadratic (3)
            :(u * v), :(u * w), :(v * w),            # bilinear (3)
            :(u^2 * v), :(u^2 * w), :(v^2 * u), :(v^2 * w), :(w^2 * u), :(w^2 * v),  # mixed (6)
            :(u * v * w),                            # trilinear (1)
            :(u^2 * v^2), :(u^2 * w^2), :(v^2 * w^2),  # biquadratic (3)
            :(u^2 * v * w), :(u * v^2 * w), :(u * v * w^2),  # mixed (3)
            :(u^2 * v^2 * w), :(u^2 * v * w^2), :(u * v^2 * w^2),  # mixed (3)
            :(u^2 * v^2 * w^2)                       # triquadratic (1)
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 3D PYRAMID ELEMENTS
    # ─────────────────────────────────────────────────────────────────────────

    # Pyr5: 5-node linear pyramid
    push!(elements, ElementDescription(
        name="Pyr5",
        description="5-node linear pyramid element",
        coordinates=[
            [-1.0, -1.0, 0.0],  # Base node 1
            [1.0, -1.0, 0.0],  # Base node 2
            [1.0, 1.0, 0.0],   # Base node 3
            [-1.0, 1.0, 0.0],   # Base node 4
            [0.0, 0.0, 1.0]    # Apex node 5
        ],
        ansatz=[
            :(1),                        # 1
            :(u), :(v), :(w),            # linear
            :(u * v)                     # bilinear base
        ]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 3D WEDGE ELEMENTS (Triangular prisms)
    # ─────────────────────────────────────────────────────────────────────────

    # Wedge6: 6-node linear wedge
    push!(elements, ElementDescription(
        name="Wedge6",
        description="6-node linear wedge element (triangular prism)",
        coordinates=[
            [0.0, 0.0, -1.0],  # Bottom triangle node 1
            [1.0, 0.0, -1.0],  # Bottom triangle node 2
            [0.0, 1.0, -1.0],  # Bottom triangle node 3
            [0.0, 0.0, 1.0],   # Top triangle node 4
            [1.0, 0.0, 1.0],   # Top triangle node 5
            [0.0, 1.0, 1.0]    # Top triangle node 6
        ],
        ansatz=[
            :(1),                        # 1
            :(u), :(v), :(w),            # linear
            :(u * w), :(v * w)           # prism bilinear
        ]
    ))

    # Wedge15: 15-node quadratic wedge
    push!(elements, ElementDescription(
        name="Wedge15",
        description="15-node quadratic wedge element",
        coordinates=[
            [0.0, 0.0, -1.0],  # Bottom vertex 1
            [1.0, 0.0, -1.0],  # Bottom vertex 2
            [0.0, 1.0, -1.0],  # Bottom vertex 3
            [0.0, 0.0, 1.0],   # Top vertex 4
            [1.0, 0.0, 1.0],   # Top vertex 5
            [0.0, 1.0, 1.0],   # Top vertex 6
            [0.5, 0.0, -1.0],  # Bottom edge 1-2
            [0.5, 0.5, -1.0],  # Bottom edge 2-3
            [0.0, 0.5, -1.0],  # Bottom edge 3-1
            [0.0, 0.0, 0.0],   # Vertical edge 1-4
            [1.0, 0.0, 0.0],   # Vertical edge 2-5
            [0.0, 1.0, 0.0],   # Vertical edge 3-6
            [0.5, 0.0, 1.0],   # Top edge 4-5
            [0.5, 0.5, 1.0],   # Top edge 5-6
            [0.0, 0.5, 1.0]    # Top edge 6-4
        ],
        ansatz=[
            :(1),                                      # 1
            :(u), :(v), :(w),                          # linear (3)
            :(u^2), :(v^2), :(w^2),                    # quadratic (3)
            :(u * v),                                  # triangle bilinear (1)
            :(u * w), :(v * w),                        # prism bilinear (2)
            :(u^2 * w), :(v^2 * w), :(u * v * w),      # mixed (3)
            :(u * w^2), :(v * w^2)                     # mixed (2)
        ]
    ))

    println("Element catalog loaded: $(length(elements)) element types")
    println()

    # =========================================================================
    # GENERATION LOOP
    # =========================================================================

    println("Generating basis functions...")
    println("─"^80)

    output = IOBuffer()

    # File header
    println(output, "# This file is a part of JuliaFEM.")
    println(output, "# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE")
    println(output)
    println(output, "# ============================================================================")
    println(output, "# AUTO-GENERATED LAGRANGE BASIS FUNCTIONS")
    println(output, "# ============================================================================")
    println(output, "#")
    println(output, "# WARNING: DO NOT EDIT THIS FILE MANUALLY!")
    println(output, "#")
    println(output, "# This file was automatically generated by:")
    println(output, "#   julia --project=. src/basis/lagrange_generator.jl")
    println(output, "#")
    println(output, "# To regenerate (e.g., after adding new element types):")
    println(output, "#   cd /path/to/JuliaFEM.jl")
    println(output, "#   julia --project=. src/basis/lagrange_generator.jl")
    println(output, "#")
    println(output, "# Theory:")
    println(output, "#   See docs/book/lagrange_basis_functions.md")
    println(output, "#")
    println(output, "# Generator:")
    println(output, "#   src/basis/lagrange_generator.jl (symbolic engine)")
    println(output, "#")
    println(output, "# Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(output, "# ============================================================================")
    println(output)

    # No exports needed - we generate methods for the parametric Lagrange{T,P} type
    println(output, "# This file generates methods for Lagrange{T,P} where:")
    println(output, "#   T = topology type (Segment, Triangle, Quadrilateral, etc.)")
    println(output, "#   P = polynomial degree (1, 2, 3, ...)")
    println(output, "#")
    println(output, "# Example: Lagrange{Triangle, 2} is a quadratic triangular element")
    println(output)

    for (i, elem) in enumerate(elements)
        println("[$i/$(length(elements))] Generating $(elem.name)...")

        try
            # Map old element name to (topology_type, polynomial_degree)
            if !haskey(ELEMENT_TO_LAGRANGE, elem.name)
                println("  ⚠ Warning: $(elem.name) not in ELEMENT_TO_LAGRANGE mapping")
                println("  Skipping...")
                continue
            end
            
            topology_type, poly_degree = ELEMENT_TO_LAGRANGE[elem.name]
            
            # Convert coordinates to proper format (tuples, not vectors)
            coords = [tuple(coord...) for coord in elem.coordinates]

            # Build polynomial ansatz as single expression: term1 + term2 + ...
            if length(elem.ansatz) == 1
                ansatz_expr = elem.ansatz[1]
            else
                ansatz_expr = Expr(:call, :+, elem.ansatz...)
            end

            # Generate basis code using symbolic engine
            # Now generates methods for Lagrange{topology_type, poly_degree}
            basis_code_expr = create_basis(
                topology_type,
                poly_degree,
                elem.description,
                coords,
                ansatz_expr
            )

            # Convert Expr to readable Julia code string
            basis_code_str = string(basis_code_expr)

            # Pretty-print: The generated code is in a quote block, extract inner code
            basis_code_str = replace(basis_code_str, r"^(quote|begin)\s+" => "")
            basis_code_str = replace(basis_code_str, r"\s*end$" => "")

            # Clean up generator artifacts (file paths, line numbers, etc.)
            basis_code_str = replace(basis_code_str, r"#=.*?=#\n?" => "")
            basis_code_str = replace(basis_code_str, r"\n{3,}" => "\n\n")

            # Write to output with nice formatting
            println(output, "# " * "─"^78)
            println(output, "# Lagrange{$(topology_type), $(poly_degree)}: $(elem.description)")
            println(output, "# (Old name: $(elem.name))")
            println(output, "# " * "─"^78)
            println(output)
            println(output, basis_code_str)
            println(output)

        catch e
            println("  ⚠ Error generating $(elem.name):")
            println("     $e")
            if isa(e, ErrorException)
                for (exc, bt) in Base.catch_stack()
                    showerror(stdout, exc, bt)
                    println()
                end
            end
            println("  Skipping...")
        end
    end

    println("─"^80)
    println()

    # Write output file
    output_path = joinpath(@__DIR__, "lagrange_generated.jl")
    println("Writing to: $output_path")
    write(output_path, String(take!(output)))
    println("✓ Generation complete!")
    println()

    # Summary
    println("Generated $(length(elements)) element types:")
    for elem in elements
        println("  - $(elem.name): $(elem.description)")
    end
    println()

    println("Next steps:")
    println("  1. Review: src/basis/lagrange_generated.jl")
    println("  2. Test: julia --project=. -e 'using JuliaFEM'")
    println("  3. Run tests: julia --project=. -e 'using Pkg; Pkg.test()'")
    println("  4. Commit: git add src/basis/lagrange_generated.jl && git commit")
    println()
    println("="^80)
end

