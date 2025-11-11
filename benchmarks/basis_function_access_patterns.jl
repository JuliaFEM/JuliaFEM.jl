# Benchmark: Basis Function Access Patterns for Tet10 (Realistic 3D Case)
# 
# Focus: 10-node quadratic tetrahedron (Tet10) - the workhorse for 3D simulations
# 
# Goal: Find the fastest way to access basis functions and their DERIVATIVES with:
# 1. Return all 10 basis functions as tuple (zero allocation)
# 2. Return single basis function by index (must be inlineable)
# 3. Return all 10 derivatives as tuple of Vec{3} (zero allocation)
# 4. Return single derivative by index (must be inlineable)
# 5. Pass topology separately (separation of concerns)
# 6. Must be type-stable and superfast
#
# Use Case:
# - Stiffness matrix assembly: Need derivatives (B matrix construction)
# - Mass matrix assembly: Need basis functions (M matrix construction)
# - Nodal assembly: Need single basis function/derivative at a time
#
# Usage: julia --project=. benchmarks/basis_function_access_patterns.jl

using BenchmarkTools
using Tensors

# ============================================================================
# Define minimal types for testing
# ============================================================================

abstract type AbstractTopology end
struct Tetrahedron <: AbstractTopology end

abstract type AbstractBasis end
struct Lagrange{P} <: AbstractBasis end

# Vec type comes from Tensors.jl (already available in JuliaFEM)
# Vec{3,Float64} for 3D gradients

# ============================================================================
# Strategy 1: Return tuple, index with getindex
# ============================================================================
# Advantages: Natural Julia syntax, type-stable
# Disadvantages: Might not inline getindex?

"""
Get all basis functions for Triangle, P1 Lagrange (3 functions).
Returns tuple of 3 Float64 values.
"""
@inline function get_basis_functions_v1(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    u, v = xi
    # P1 triangle: N1 = 1-u-v, N2 = u, N3 = v
    return (1 - u - v, u, v)
end

"""
Get single basis function by index (1-based).
"""
@inline function get_basis_function_v1(topology::Triangle, basis::Lagrange{1}, xi::Vec{2,T}, i::Int) where T
    N_all = get_basis_functions_v1(topology, basis, xi)
    return N_all[i]  # Tuple indexing
end

# ============================================================================
# Strategy 2: Generated function for single access
# ============================================================================
# Advantages: Compiler can specialize for each index
# Disadvantages: More complex code

@inline function get_basis_functions_v2(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    u, v = xi
    return (1 - u - v, u, v)
end

"""
Use @generated to create specialized code for each index at compile time.
"""
@generated function get_basis_function_v2(::Triangle, ::Lagrange{1}, xi::Vec{2,T}, ::Val{I}) where {T,I}
    if I == 1
        return :(1 - xi[1] - xi[2])
    elseif I == 2
        return :(xi[1])
    elseif I == 3
        return :(xi[2])
    else
        return :(error("Invalid basis function index: $I"))
    end
end

# ============================================================================
# Strategy 3: Manual dispatch on Val (type-stable index)
# ============================================================================
# Advantages: Explicit, clear what's happening
# Disadvantages: Verbose, need to write each case

@inline function get_basis_functions_v3(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    u, v = xi
    return (1 - u - v, u, v)
end

@inline get_basis_function_v3(t::Triangle, b::Lagrange{1}, xi::Vec{2,T}, ::Val{1}) where T = 1 - xi[1] - xi[2]
@inline get_basis_function_v3(t::Triangle, b::Lagrange{1}, xi::Vec{2,T}, ::Val{2}) where T = xi[1]
@inline get_basis_function_v3(t::Triangle, b::Lagrange{1}, xi::Vec{2,T}, ::Val{3}) where T = xi[2]

# ============================================================================
# Strategy 4: Struct with getindex (most Julian)
# ============================================================================
# Advantages: Can use N[i] syntax naturally
# Disadvantages: Extra struct allocation?

struct BasisFunctions{N,T}
    data::NTuple{N,T}
end

Base.@propagate_inbounds Base.getindex(bf::BasisFunctions, i::Int) = bf.data[i]
Base.length(::BasisFunctions{N}) where N = N

@inline function get_basis_functions_v4(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    u, v = xi
    return BasisFunctions((1 - u - v, u, v))
end

# Can use natural indexing
@inline function get_basis_function_v4(topology::Triangle, basis::Lagrange{1}, xi::Vec{2,T}, i::Int) where T
    N = get_basis_functions_v4(topology, basis, xi)
    return N[i]
end

# ============================================================================
# Strategy 5: Separate implementation per basis function (extreme specialization)
# ============================================================================
# Advantages: Maximum performance, no tuple allocation at all
# Disadvantages: Lots of code duplication

@inline function get_basis_function_1_v5(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    return 1 - xi[1] - xi[2]
end

@inline function get_basis_function_2_v5(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    return xi[1]
end

@inline function get_basis_function_3_v5(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    return xi[2]
end

@inline function get_basis_functions_v5(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
    u, v = xi
    return (1 - u - v, u, v)
end

# ============================================================================
# Benchmark: Access all basis functions (typical in assembly loop)
# ============================================================================

function benchmark_all_access()
    println("\n" * "="^80)
    println("BENCHMARK: Access ALL basis functions")
    println("="^80)

    topology = Triangle()
    basis = Lagrange{1}()
    xi = Vec(0.25, 0.25)

    println("\nStrategy 1: Tuple return + getindex")
    @btime get_basis_functions_v1($topology, $basis, $xi)

    println("\nStrategy 2: Generated function")
    @btime get_basis_functions_v2($topology, $basis, $xi)

    println("\nStrategy 3: Val dispatch")
    @btime get_basis_functions_v3($topology, $basis, $xi)

    println("\nStrategy 4: BasisFunctions struct")
    @btime get_basis_functions_v4($topology, $basis, $xi)

    println("\nStrategy 5: Separate functions")
    @btime get_basis_functions_v5($topology, $basis, $xi)

    # Verify all return same values
    r1 = get_basis_functions_v1(topology, basis, xi)
    r2 = get_basis_functions_v2(topology, basis, xi)
    r3 = get_basis_functions_v3(topology, basis, xi)
    r4 = get_basis_functions_v4(topology, basis, xi).data
    r5 = get_basis_functions_v5(topology, basis, xi)

    @assert r1 == r2 == r3 == r4 == r5 "Results don't match!"
    println("\n✓ All strategies return identical values: $r1")
end

# ============================================================================
# Benchmark: Access SINGLE basis function (for nodal assembly)
# ============================================================================

function benchmark_single_access()
    println("\n" * "="^80)
    println("BENCHMARK: Access SINGLE basis function (nodal assembly)")
    println("="^80)

    topology = Triangle()
    basis = Lagrange{1}()
    xi = Vec(0.25, 0.25)

    println("\nStrategy 1: Tuple + runtime index")
    @btime get_basis_function_v1($topology, $basis, $xi, 2)

    println("\nStrategy 2: Generated function with Val{2}")
    @btime get_basis_function_v2($topology, $basis, $xi, Val(2))

    println("\nStrategy 3: Val dispatch")
    @btime get_basis_function_v3($topology, $basis, $xi, Val(2))

    println("\nStrategy 4: BasisFunctions struct + index")
    @btime get_basis_function_v4($topology, $basis, $xi, 2)

    println("\nStrategy 5: Direct function call")
    @btime get_basis_function_2_v5($topology, $basis, $xi)

    # Verify all return same value
    r1 = get_basis_function_v1(topology, basis, xi, 2)
    r2 = get_basis_function_v2(topology, basis, xi, Val(2))
    r3 = get_basis_function_v3(topology, basis, xi, Val(2))
    r4 = get_basis_function_v4(topology, basis, xi, 2)
    r5 = get_basis_function_2_v5(topology, basis, xi)

    @assert r1 == r2 == r3 == r4 == r5 "Results don't match!"
    println("\n✓ All strategies return identical value: $r1")
end

# ============================================================================
# Benchmark: Typical assembly loop pattern
# ============================================================================

function benchmark_assembly_loop()
    println("\n" * "="^80)
    println("BENCHMARK: Typical assembly loop (iterate over all basis functions)")
    println("="^80)

    topology = Triangle()
    basis = Lagrange{1}()
    xi = Vec(0.25, 0.25)

    # Pattern 1: Get all, iterate over tuple
    println("\nPattern 1: Get all as tuple, iterate")
    function assemble_v1()
        N_all = get_basis_functions_v1(topology, basis, xi)
        s = 0.0
        for N_i in N_all
            s += N_i * N_i  # Dummy computation
        end
        return s
    end
    @btime assemble_v1()

    # Pattern 2: Get all, index in loop
    println("\nPattern 2: Get all, index with i")
    function assemble_v2()
        N_all = get_basis_functions_v1(topology, basis, xi)
        s = 0.0
        for i in 1:3
            s += N_all[i] * N_all[i]
        end
        return s
    end
    @btime assemble_v2()

    # Pattern 3: Get one at a time (nodal assembly style)
    println("\nPattern 3: Get one at a time with Val")
    function assemble_v3()
        s = 0.0
        # Unrolled loop (what compiler would do with Val)
        N1 = get_basis_function_v3(topology, basis, xi, Val(1))
        s += N1 * N1
        N2 = get_basis_function_v3(topology, basis, xi, Val(2))
        s += N2 * N2
        N3 = get_basis_function_v3(topology, basis, xi, Val(3))
        s += N3 * N3
        return s
    end
    @btime assemble_v3()

    # Pattern 4: Direct function calls (strategy 5)
    println("\nPattern 4: Direct function calls (extreme specialization)")
    function assemble_v4()
        s = 0.0
        N1 = get_basis_function_1_v5(topology, basis, xi)
        s += N1 * N1
        N2 = get_basis_function_2_v5(topology, basis, xi)
        s += N2 * N2
        N3 = get_basis_function_3_v5(topology, basis, xi)
        s += N3 * N3
        return s
    end
    @btime assemble_v4()

    # Verify all compute same result
    r1 = assemble_v1()
    r2 = assemble_v2()
    r3 = assemble_v3()
    r4 = assemble_v4()
    @assert r1 == r2 == r3 == r4 "Assembly results don't match!"
    println("\n✓ All patterns compute same result: $r1")
end

# ============================================================================
# Benchmark: Basis derivatives (return Vec)
# ============================================================================

function benchmark_derivatives()
    println("\n" * "="^80)
    println("BENCHMARK: Basis function DERIVATIVES (return Vec)")
    println("="^80)

    topology = Triangle()
    basis = Lagrange{1}()
    xi = Vec(0.25, 0.25)

    # Triangle P1 derivatives (constant):
    # dN1/d(u,v) = (-1, -1)
    # dN2/d(u,v) = (1, 0)
    # dN3/d(u,v) = (0, 1)

    println("\nStrategy 1: Return tuple of Vecs")
    @inline function get_basis_derivatives_v1(::Triangle, ::Lagrange{1}, xi::Vec{2,T}) where T
        return (Vec(-1.0, -1.0), Vec(1.0, 0.0), Vec(0.0, 1.0))
    end
    @btime get_basis_derivatives_v1($topology, $basis, $xi)

    println("\nStrategy 2: Return single Vec with Val indexing")
    @inline get_basis_derivative_v2(::Triangle, ::Lagrange{1}, xi::Vec{2,T}, ::Val{1}) where T = Vec(-1.0, -1.0)
    @inline get_basis_derivative_v2(::Triangle, ::Lagrange{1}, xi::Vec{2,T}, ::Val{2}) where T = Vec(1.0, 0.0)
    @inline get_basis_derivative_v2(::Triangle, ::Lagrange{1}, xi::Vec{2,T}, ::Val{3}) where T = Vec(0.0, 1.0)
    @btime get_basis_derivative_v2($topology, $basis, $xi, Val(2))

    # Verify
    all_derivs = get_basis_derivatives_v1(topology, basis, xi)
    single_deriv = get_basis_derivative_v2(topology, basis, xi, Val(2))
    @assert all_derivs[2] == single_deriv
    println("\n✓ Derivatives match: $single_deriv")
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    println("\n")
    println("╔" * "="^78 * "╗")
    println("║" * " "^78 * "║")
    println("║" * " "^20 * "BASIS FUNCTION ACCESS PATTERNS BENCHMARK" * " "^18 * "║")
    println("║" * " "^78 * "║")
    println("╚" * "="^78 * "╝")

    println("\nGoal: Find fastest way to access basis functions for nodal assembly")
    println("Requirements:")
    println("  - Zero allocation")
    println("  - Type stable")
    println("  - Inlineable")
    println("  - Support both 'all at once' and 'one at a time' access")

    benchmark_all_access()
    benchmark_single_access()
    benchmark_assembly_loop()
    benchmark_derivatives()

    println("\n" * "="^80)
    println("SUMMARY & RECOMMENDATIONS")
    println("="^80)
    println("""

    For TRADITIONAL ASSEMBLY (get all basis functions at integration point):
    → Use Strategy 1 or 4: Simple tuple return
    → Should be 0-5 ns, zero allocation

    For NODAL ASSEMBLY (get single basis function):
    → Use Strategy 3: Val dispatch for compile-time index
    → Should be 0-2 ns, zero allocation, fully inlined
    → Usage: get_basis_function(Triangle(), Lagrange{1}(), xi, Val(i))

    For DERIVATIVES:
    → Return tuple of Vec for all derivatives
    → Use Val indexing for single derivative
    → Same performance as basis functions

    RECOMMENDED API:
    ```julia
    # Get all basis functions (returns tuple)
    N_all = get_basis_functions(Triangle(), Lagrange{1}(), xi)

    # Get single basis function (Val for compile-time specialization)
    N_i = get_basis_function(Triangle(), Lagrange{1}(), xi, Val(i))

    # Get all derivatives (returns tuple of Vec)
    dN_all = get_basis_derivatives(Triangle(), Lagrange{1}(), xi)

    # Get single derivative (returns Vec)
    dN_i = get_basis_derivative(Triangle(), Lagrange{1}(), xi, Val(i))
    ```

    WHY Val?
    - Compiler knows index at compile time
    - Can generate optimal code for each basis function
    - Zero runtime overhead
    - Type stable

    NOTE: For runtime indexing (i not known at compile time), tuple indexing
    is still very fast (typically 1-2 ns overhead).
    """)

    println("\n" * "="^80)
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
