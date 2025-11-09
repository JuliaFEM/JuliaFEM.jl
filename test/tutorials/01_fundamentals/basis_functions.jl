# # Numerical Integration and Jacobian
#
# **Purpose:** Understand how FEM uses numerical integration with Tensors.jl
#
# This tutorial explores numerical integration in finite element analysis,
# which is fundamental to computing element matrices and vectors.
#
# ## Why This Matters
#
# In FEM, we compute element matrices by integrating:
# ```math
# K = \int_{\Omega} B^T D B \, dΩ
# ```
#
# Numerically:
# ```math
# K ≈ \sum_{ip} w_{ip} B^T D B |J|_{ip}
# ```
#
# Where:
# - ip = integration points (Gauss quadrature points)
# - w = quadrature weights
# - |J| = Jacobian determinant (coordinate transformation scaling)

using JuliaFEM
using Test

# ## Step 1: Integration Points (Gauss Quadrature)
#
# JuliaFEM uses Gauss quadrature for numerical integration.
# For Quad4, we use 2×2 Gauss quadrature (4 points).

# Create a unit square element
nodes = Dict(
    1 => [0.0, 0.0],
    2 => [1.0, 0.0],
    3 => [1.0, 1.0],
    4 => [0.0, 1.0]
)

element = Element(Quad4, [1, 2, 3, 4])
update!(element, "geometry", nodes)

@testset "Integration Points: Structure" begin
    ips = get_integration_points(element)

    @test length(ips) == 4  # 2×2 Gauss quadrature for Quad4

    # Each integration point has coords and weight
    @test hasfield(typeof(ips[1]), :weight)
    @test hasfield(typeof(ips[1]), :coords)

    # Coordinates are in parametric space [-1, 1]²
    for ip in ips
        ξ, η = ip.coords
        @test -1 <= ξ <= 1
        @test -1 <= η <= 1
    end
end

@testset "Integration Points: Weights" begin
    ips = get_integration_points(element)

    # For 2D Gauss quadrature in [-1,1]², weights sum to 4
    total_weight = sum(ip.weight for ip in ips)
    @test total_weight ≈ 4.0

    # For 2×2 Gauss, all weights are equal (symmetry)
    weights = [ip.weight for ip in ips]
    @test all(w ≈ weights[1] for w in weights)
    @test weights[1] ≈ 1.0  # Each weight = 1 for 2×2 Gauss
end

# ## Step 2: Jacobian Evaluation (Now Working with Tensors.jl!)
#
# The Jacobian transforms derivatives from parametric to physical coordinates.
# With our Tensors.jl fixes, this now works correctly.

@testset "Jacobian: Determinant" begin
    ips = get_integration_points(element)

    for ip in ips
        # Jacobian determinant must be positive (non-inverted element)
        detJ = element(ip, 0.0, Val{:detJ})
        @test detJ > 0

        # For unit square, Jacobian is constant
        # At any point, |J| should be 0.25 (scale factor from [-1,1]² to [0,1]²)
        @test detJ ≈ 0.25
    end
end

@testset "Jacobian: Matrix" begin
    ips = get_integration_points(element)

    for ip in ips
        # Get full Jacobian matrix
        J = element(ip, 0.0, Val{:Jacobian})

        # Should be 2×2 for 2D element
        @test size(J) == (2, 2)

        # For unit square aligned with axes, should be diagonal
        @test J[1, 1] ≈ 0.5  # ∂x/∂ξ
        @test J[2, 2] ≈ 0.5  # ∂y/∂η
        @test abs(J[1, 2]) < 1e-10  # ∂y/∂ξ ≈ 0
        @test abs(J[2, 1]) < 1e-10  # ∂x/∂η ≈ 0
    end
end

# ## Step 3: Numerical Integration
#
# Now that Jacobian works, we can perform numerical integration!

@testset "Integration: Constant Function" begin
    # Integrate f(x,y) = 1 over unit square → area = 1.0
    ips = get_integration_points(element)

    integral = 0.0
    for ip in ips
        detJ = element(ip, 0.0, Val{:detJ})
        # Integrate constant function f=1
        integral += ip.weight * 1.0 * detJ
    end

    @test integral ≈ 1.0 atol = 1e-10  # Area of unit square
end

@testset "Integration: Linear Function x" begin
    # Integrate f(x,y) = x over unit square
    # Analytical: ∫₀¹ ∫₀¹ x dy dx = 1/2
    ips = get_integration_points(element)

    integral = 0.0
    for ip in ips
        # Get physical coordinates at this integration point
        # Use basis functions to interpolate
        N = element(ip, 0.0)
        x_ip = sum(N[i] * nodes[i][1] for i in 1:4)

        detJ = element(ip, 0.0, Val{:detJ})
        integral += ip.weight * x_ip * detJ
    end

    @test integral ≈ 0.5 atol = 1e-10
end

@testset "Integration: Quadratic Function x²" begin
    # Integrate f(x,y) = x² over unit square  
    # Analytical: ∫₀¹ ∫₀¹ x² dy dx = 1/3
    ips = get_integration_points(element)

    integral = 0.0
    for ip in ips
        N = element(ip, 0.0)
        x_ip = sum(N[i] * nodes[i][1] for i in 1:4)
        detJ = element(ip, 0.0, Val{:detJ})

        integral += ip.weight * x_ip^2 * detJ
    end

    @test integral ≈ 1 / 3 atol = 1e-10
end

# ## Step 4: Different Element Types

@testset "Integration: Seg2 (1D)" begin
    # 1D line element
    nodes_1d = Dict(1 => [0.0], 2 => [2.0])
    element_1d = Element(Seg2, [1, 2])
    update!(element_1d, "geometry", nodes_1d)

    ips = get_integration_points(element_1d)
    @test length(ips) == 2  # 2-point Gauss in 1D

    # Integrate over length
    length_integral = sum(ip.weight * element_1d(ip, 0.0, Val{:detJ}) for ip in ips)
    @test length_integral ≈ 2.0  # Length of element
end

@testset "Integration: Tri3 (Triangle)" begin
    # Triangular element
    nodes_tri = Dict(
        1 => [0.0, 0.0],
        2 => [1.0, 0.0],
        3 => [0.0, 1.0]
    )
    element_tri = Element(Tri3, [1, 2, 3])
    update!(element_tri, "geometry", nodes_tri)

    ips = get_integration_points(element_tri)
    @test length(ips) >= 1  # At least one integration point

    # Integrate constant → area of triangle = 0.5
    area = sum(ip.weight * element_tri(ip, 0.0, Val{:detJ}) for ip in ips)
    @test area ≈ 0.5 atol = 1e-10
end

# ## Discussion
#
# With Tensors.jl properly integrated throughout, we can now:
#
# 1. **Evaluate Jacobian:** Transform between parametric and physical coordinates
# 2. **Perform Integration:** Numerical quadrature works correctly
# 3. **Use Multiple Element Types:** Seg2, Tri3, Quad4 all work
#
# ## Key Architectural Decision
#
# **Using Tensors.jl everywhere** provides:
# - Zero-cost abstractions
# - Type stability
# - Consistent API across all geometric calculations
# - Material science compatibility
#
# ## What's Next?
#
# - Assembly: Build global matrices using these integrations
# - Solvers: Solve FEM problems end-to-end
# - Advanced elements: Higher-order elements, 3D
#
# ## References
#
# - Tensors.jl documentation: https://github.com/Ferrite-FEM/Tensors.jl
# - Hughes, T.J.R., "The Finite Element Method", Dover (Chapter 3)

println()
println("="^70)
println("Numerical Integration Tutorial Complete!")
println("="^70)
println("✓ Integration points and Gauss quadrature working")
println("✓ Jacobian evaluation fixed with Tensors.jl")
println("✓ Numerical integration validated (constant, linear, quadratic)")
println("✓ Multiple element types tested (Quad4, Seg2, Tri3)")
println()
println("Tensors.jl is now consistently used throughout JuliaFEM!")
println("="^70)
