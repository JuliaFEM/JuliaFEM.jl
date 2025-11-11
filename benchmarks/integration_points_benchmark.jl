# Benchmark: Integration Point Access Patterns
# ============================================
# 
# This benchmark compares different approaches to storing and accessing
# integration points in finite element assembly loops.
#
# Key Question: What's the fastest way to get integration points?
#
# Approaches tested:
# 1. OLD: Runtime dispatch + mutable struct with Dict (type-unstable)
# 2. NEW Option A: Compile-time function (like eval_basis!)
# 3. NEW Option B: Store in element as NTuple
# 4. NEW Option C: Hybrid (compile-time generation + caching)

using BenchmarkTools
using Tensors
using StaticArrays

# ============================================================================
# OLD APPROACH: Runtime dispatch with mutable struct
# ============================================================================

struct OldIP
    id::UInt
    weight::Float64
    coords::Tuple{Vararg{Float64}}
    fields::Dict{String,Any}  # Type-unstable!
end

function get_integration_points_old(::Type{Val{:Tri3}})
    # Simulate runtime dispatch to get IPs
    return [
        OldIP(UInt(1), 0.5, (1 / 3, 1 / 3), Dict{String,Any}()),
    ]
end

function assembly_loop_old()
    sum_val = 0.0
    for _ in 1:1000  # Simulate 1000 elements
        ips = get_integration_points_old(Val{:Tri3})
        for ip in ips
            w = ip.weight
            xi, eta = ip.coords
            # Simulate some computation
            sum_val += w * (xi + eta)
        end
    end
    return sum_val
end

# ============================================================================
# NEW OPTION A: Compile-time function (zero allocation)
# ============================================================================

"""
Return integration points as tuple at compile time.
Similar to eval_basis! - zero allocation, fully inlined.
"""
@inline function get_integration_points!(::Type{Val{:Tri3_Gauss1}})
    # Return as tuple of (weight, coords) pairs
    return (
        (0.5, (1 / 3, 1 / 3)),
    )
end

@inline function get_integration_points!(::Type{Val{:Tet4_Gauss1}})
    return (
        (1 / 24, (0.25, 0.25, 0.25)),
    )
end

function assembly_loop_option_a()
    sum_val = 0.0
    for _ in 1:1000
        ips = get_integration_points!(Val{:Tri3_Gauss1})
        for (w, (xi, eta)) in ips
            sum_val += w * (xi + eta)
        end
    end
    return sum_val
end

# ============================================================================
# NEW OPTION B: Store in element as NTuple (what we have now)
# ============================================================================

struct IntegrationPoint{D}
    ξ::NTuple{D,Float64}
    weight::Float64
end

struct MockElement{NIP}
    ips::NTuple{NIP,IntegrationPoint{2}}
end

function create_element_b()
    ips = (
        IntegrationPoint((1 / 3, 1 / 3), 0.5),
    )
    return MockElement(ips)
end

function assembly_loop_option_b()
    elements = [create_element_b() for _ in 1:1000]
    sum_val = 0.0
    for element in elements
        for ip in element.ips
            w = ip.weight
            xi, eta = ip.ξ
            sum_val += w * (xi + eta)
        end
    end
    return sum_val
end

# ============================================================================
# NEW OPTION C: Compile-time with Tensors.jl Vec (recommended for FEM)
# ============================================================================

"""
Return integration points with Vec{D} coordinates (Tensors.jl).
This matches the golden standard from nodal assembly demos.
"""
@inline function get_integration_points_vec!(::Type{Val{:Tri3_Gauss1}})
    return (
        (0.5, Vec{2}((1 / 3, 1 / 3))),
    )
end

@inline function get_integration_points_vec!(::Type{Val{:Tet4_Gauss1}})
    return (
        (1 / 24, Vec{3}((0.25, 0.25, 0.25))),
    )
end

function assembly_loop_option_c()
    sum_val = 0.0
    for _ in 1:1000
        ips = get_integration_points_vec!(Val{:Tri3_Gauss1})
        for (w, xi) in ips
            # Vec arithmetic is optimized by Tensors.jl
            sum_val += w * sum(xi)
        end
    end
    return sum_val
end

# ============================================================================
# NEW OPTION D: Pre-generated global constants (ultimate zero-cost)
# ============================================================================

const TRI3_GAUSS1_IPS = (
    (0.5, Vec{2}((1 / 3, 1 / 3))),
)

const TET4_GAUSS1_IPS = (
    (1 / 24, Vec{3}((0.25, 0.25, 0.25))),
)

function assembly_loop_option_d()
    sum_val = 0.0
    for _ in 1:1000
        for (w, xi) in TRI3_GAUSS1_IPS
            sum_val += w * sum(xi)
        end
    end
    return sum_val
end

# ============================================================================
# OPTION E: Hybrid - Function returns pre-computed constant
# ============================================================================

@inline get_ips_tri3_gauss1() = TRI3_GAUSS1_IPS
@inline get_ips_tet4_gauss1() = TET4_GAUSS1_IPS

function assembly_loop_option_e()
    sum_val = 0.0
    for _ in 1:1000
        for (w, xi) in get_ips_tri3_gauss1()
            sum_val += w * sum(xi)
        end
    end
    return sum_val
end

# ============================================================================
# Run Benchmarks
# ============================================================================

println("="^80)
println("Integration Point Access Pattern Benchmark")
println("="^80)
println()

println("OLD APPROACH: Runtime dispatch + mutable struct with Dict")
println("-"^80)
@btime assembly_loop_old()
println()

println("OPTION A: Compile-time function returning tuples")
println("-"^80)
@btime assembly_loop_option_a()
println()

println("OPTION B: Store in element as NTuple (current approach)")
println("-"^80)
@btime assembly_loop_option_b()
println()

println("OPTION C: Compile-time function with Vec{D} (Tensors.jl)")
println("-"^80)
@btime assembly_loop_option_c()
println()

println("OPTION D: Pre-generated global constants")
println("-"^80)
@btime assembly_loop_option_d()
println()

println("OPTION E: Function returning pre-computed constant")
println("-"^80)
@btime assembly_loop_option_e()
println()

# ============================================================================
# Realistic FEM Assembly Benchmark
# ============================================================================

println()
println("="^80)
println("REALISTIC FEM ASSEMBLY COMPARISON")
println("="^80)
println()

# Simulate realistic element stiffness computation
function compute_element_stiffness_old(element_type::Type{Val{:Tri3}})
    K_elem = zeros(6, 6)
    ips = get_integration_points_old(element_type)
    for ip in ips
        w = ip.weight
        xi, eta = ip.coords
        # Simulate shape function evaluation and stiffness computation
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        # Accumulate (simplified)
        K_elem[1, 1] += w * (N1^2)
    end
    return K_elem[1, 1]
end

function compute_element_stiffness_new()
    K_elem = 0.0
    for (w, xi) in get_integration_points_vec!(Val{:Tri3_Gauss1})
        # Vec arithmetic
        xi_val = xi[1]
        eta_val = xi[2]
        N1 = 1 - xi_val - eta_val
        K_elem += w * (N1^2)
    end
    return K_elem
end

println("OLD: Realistic element stiffness assembly")
@btime begin
    sum_val = 0.0
    for _ in 1:1000
        sum_val += compute_element_stiffness_old(Val{:Tri3})
    end
    sum_val
end

println()
println("NEW: Realistic element stiffness assembly")
@btime begin
    sum_val = 0.0
    for _ in 1:1000
        sum_val += compute_element_stiffness_new()
    end
    sum_val
end

println()
println("="^80)
println("SUMMARY")
println("="^80)
println()
println("Expected ranking (fastest to slowest):")
println("1. Option D/E: Pre-computed constants (ultimate zero-cost)")
println("2. Option C: Compile-time with Vec{D} (recommended for FEM)")
println("3. Option A: Compile-time with plain tuples")
println("4. Option B: Stored in element NTuple (slight overhead)")
println("5. OLD: Runtime dispatch + Dict (type-unstable)")
println()
println("RECOMMENDATION:")
println("  Use Option C or D/E for integration points:")
println("  - Compile-time generation like eval_basis!")
println("  - Return as Tuple of (weight, Vec{D}) pairs")
println("  - Zero allocation, fully inlined")
println("  - Matches golden standard architecture")
