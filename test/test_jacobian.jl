using Test
using JuliaFEM
using Tensors
using LinearAlgebra

@testset "Jacobian Computation" begin

    @testset "2D Triangle - Identity Element" begin
        # Reference triangle mapped to itself (identity transformation)
        X = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        )

        # Evaluate at center
        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)

        # Compute Jacobian
        J = compute_jacobian(X, dN_dξ)

        # For identity mapping, J should be identity matrix
        @test J ≈ Tensor{2,2}((1.0, 0.0, 0.0, 1.0))
        @test det(J) ≈ 1.0
    end

    @testset "2D Triangle - Scaled Element" begin
        # Triangle scaled by 2 in x and 1.5 in y
        X = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((2.0, 0.0)),
            Vec{2}((0.0, 1.5))
        )

        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)

        J = compute_jacobian(X, dN_dξ)

        # Jacobian should reflect scaling
        @test J[1, 1] ≈ 2.0   # ∂x/∂ξ
        @test J[1, 2] ≈ 0.0   # ∂x/∂η
        @test J[2, 1] ≈ 0.0   # ∂y/∂ξ
        @test J[2, 2] ≈ 1.5   # ∂y/∂η
        @test det(J) ≈ 3.0   # Area scaling = 2 × 1.5
    end

    @testset "2D Triangle - Rotated Element" begin
        # 90° counter-clockwise rotation
        θ = π / 2
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]

        # Original nodes
        X_orig = [0.0 1.0 0.0; 0.0 0.0 1.0]

        # Rotate
        X_rot = R * X_orig
        X = (
            Vec{2}((X_rot[1, 1], X_rot[2, 1])),
            Vec{2}((X_rot[1, 2], X_rot[2, 2])),
            Vec{2}((X_rot[1, 3], X_rot[2, 3]))
        )

        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)

        J = compute_jacobian(X, dN_dξ)

        # Jacobian should contain rotation
        @test det(J) ≈ 1.0  # Area preserved under rotation
        @test norm(J) > 0   # Well-conditioned
    end

    @testset "3D Tetrahedron - Identity Element" begin
        # Reference tetrahedron mapped to itself
        X = (
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        )

        xi = Vec{3}((0.25, 0.25, 0.25))
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,1}(), xi)

        J = compute_jacobian(X, dN_dξ)

        # Identity mapping
        @test J ≈ Tensor{2,3}((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        @test det(J) ≈ 1.0
    end

    @testset "3D Tetrahedron - Scaled Element" begin
        # Tetrahedron scaled differently in each direction
        X = (
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((2.0, 0.0, 0.0)),
            Vec{3}((0.0, 3.0, 0.0)),
            Vec{3}((0.0, 0.0, 4.0))
        )

        xi = Vec{3}((0.25, 0.25, 0.25))
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,1}(), xi)

        J = compute_jacobian(X, dN_dξ)

        # Diagonal Jacobian (aligned with axes)
        @test J[1, 1] ≈ 2.0
        @test J[2, 2] ≈ 3.0
        @test J[3, 3] ≈ 4.0
        @test det(J) ≈ 24.0  # Volume scaling = 2 × 3 × 4
    end

    @testset "Physical Derivatives - 2D Triangle" begin
        X = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((2.0, 0.0)),
            Vec{2}((0.0, 1.5))
        )

        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)

        J = compute_jacobian(X, dN_dξ)
        dN_dx = physical_derivatives(J, dN_dξ)

        # Verify constant strain condition: ∑ᵢ dNᵢ/dx = 0
        sum_dN_dx = sum(dN_dx)
        @test norm(sum_dN_dx) < 1e-10

        # Verify partition of unity holds
        # (Not directly, but derivatives should be consistent)
        @test length(dN_dx) == 3
    end

    @testset "Physical Derivatives - 3D Tetrahedron" begin
        X = (
            Vec{3}((0.0, 0.0, 0.0)),
            Vec{3}((1.0, 0.0, 0.0)),
            Vec{3}((0.0, 1.0, 0.0)),
            Vec{3}((0.0, 0.0, 1.0))
        )

        xi = Vec{3}((0.25, 0.25, 0.25))
        dN_dξ = get_basis_derivatives(Tetrahedron(), Lagrange{Tetrahedron,1}(), xi)

        J = compute_jacobian(X, dN_dξ)
        dN_dx = physical_derivatives(J, dN_dξ)

        # Constant strain condition
        sum_dN_dx = sum(dN_dx)
        @test norm(sum_dN_dx) < 1e-10

        # Check each derivative is a 3D vector
        for dN in dN_dx
            @test length(dN) == 3
        end
    end

    @testset "Jacobian Determinant - Element Quality" begin
        # Well-shaped triangle
        X_good = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        )

        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)
        J_good = compute_jacobian(X_good, dN_dξ)

        @test det(J_good) > 0  # Positive (properly oriented)
        @test abs(det(J_good)) > 0.1  # Well-conditioned

        # Degenerate triangle (collapsed to line)
        X_bad = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((2.0, 0.0))  # Collinear!
        )

        J_bad = compute_jacobian(X_bad, dN_dξ)
        @test abs(det(J_bad)) < 1e-10  # Nearly zero (degenerate)
    end

    @testset "Type Stability and Zero Allocation" begin
        X = (
            Vec{2}((0.0, 0.0)),
            Vec{2}((1.0, 0.0)),
            Vec{2}((0.0, 1.0))
        )

        xi = Vec{2}((1 / 3, 1 / 3))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)

        # Type stability
        J = @inferred compute_jacobian(X, dN_dξ)
        @test J isa Tensor{2,2}

        dN_dx = @inferred physical_derivatives(J, dN_dξ)
        @test dN_dx isa Tuple

        # Zero allocation (run twice to avoid compilation)
        compute_jacobian(X, dN_dξ)
        allocs = @allocated compute_jacobian(X, dN_dξ)
        @test allocs == 0

        physical_derivatives(J, dN_dξ)
        allocs = @allocated physical_derivatives(J, dN_dξ)
        @test allocs == 0
    end

    @testset "Consistency with Manual Calculation" begin
        # Triangle with known Jacobian
        X = (
            Vec{2}((1.0, 2.0)),
            Vec{2}((4.0, 3.0)),
            Vec{2}((2.0, 6.0))
        )

        xi = Vec{2}((0.5, 0.25))
        dN_dξ = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)
        # dN_dξ = (Vec(-1, -1), Vec(1, 0), Vec(0, 1))

        J = compute_jacobian(X, dN_dξ)

        # Manual calculation:
        # J = X2 - X1 in first column, X3 - X1 in second column
        # J = [4-1  2-1] = [3  1]
        #     [3-2  6-2]   [1  4]

        @test J[1, 1] ≈ 3.0
        @test J[1, 2] ≈ 1.0
        @test J[2, 1] ≈ 1.0
        @test J[2, 2] ≈ 4.0
        @test det(J) ≈ 11.0  # 3*4 - 1*1 = 11
    end
end

@testset "Jacobian - AbstractVector Interface" begin
    # Test that Vector interface also works (less efficient)
    X_vec = [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((0.0, 1.0))]

    xi = Vec{2}((1 / 3, 1 / 3))
    dN_dξ_tuple = get_basis_derivatives(Triangle(), Lagrange{Triangle,1}(), xi)
    dN_dξ_vec = collect(dN_dξ_tuple)

    J_tuple = compute_jacobian(tuple(X_vec...), dN_dξ_tuple)
    J_vec = compute_jacobian(X_vec, dN_dξ_vec)

    @test J_tuple ≈ J_vec

    # Physical derivatives
    dN_dx_tuple = physical_derivatives(J_tuple, dN_dξ_tuple)
    dN_dx_vec = physical_derivatives(J_vec, dN_dξ_vec)

    @test all(dN_dx_tuple[i] ≈ dN_dx_vec[i] for i in 1:3)
end

println("✅ All Jacobian tests passed!")
