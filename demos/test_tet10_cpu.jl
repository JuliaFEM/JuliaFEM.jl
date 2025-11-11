"""
Test Tet10 Assembly on CPU First
=================================

Validate the JuliaFEM pattern (loop through shape functions, no B-matrix)
before moving to GPU.
"""

using LinearAlgebra
using Tensors

# Material
struct LinearElastic
    E::Float64
    ν::Float64
end

λ(mat::LinearElastic) = mat.E * mat.ν / ((1 + mat.ν) * (1 - 2mat.ν))
μ(mat::LinearElastic) = mat.E / (2(1 + mat.ν))

function compute_stress_3d(material::LinearElastic, eps::SymmetricTensor{2,3,T}) where T
    lambda_val = T(λ(material))
    mu_val = T(μ(material))
    I = one(eps)
    sigma = lambda_val * tr(eps) * I + 2 * mu_val * eps
    return sigma
end

# Tet10 Gauss quadrature
const GAUSS_TET4 = [
    (Vec{3}((0.5854101966249685, 0.1381966011250105, 0.1381966011250105)), 0.25),
    (Vec{3}((0.1381966011250105, 0.5854101966249685, 0.1381966011250105)), 0.25),
    (Vec{3}((0.1381966011250105, 0.1381966011250105, 0.5854101966249685)), 0.25),
    (Vec{3}((0.1381966011250105, 0.1381966011250105, 0.1381966011250105)), 0.25)
]

function tet10_shape_derivatives(xi, eta, zeta)::NTuple{10,Vec{3,Float64}}
    lambda = 1 - xi - eta - zeta

    # Vertex nodes
    dN1 = Vec{3}((4 * lambda - 1, 4 * lambda - 1, 4 * lambda - 1))
    dN2 = Vec{3}((4 * xi - 1, 0.0, 0.0))
    dN3 = Vec{3}((0.0, 4 * eta - 1, 0.0))
    dN4 = Vec{3}((0.0, 0.0, 4 * zeta - 1))

    # Edge midpoints
    dN5 = Vec{3}((4 * (1 - 2 * xi - eta - zeta), -4 * xi, -4 * xi))
    dN6 = Vec{3}((4 * eta, 4 * xi, 0.0))
    dN7 = Vec{3}((-4 * eta, 4 * (1 - xi - 2 * eta - zeta), -4 * eta))
    dN8 = Vec{3}((-4 * zeta, -4 * zeta, 4 * (1 - xi - eta - 2 * zeta)))
    dN9 = Vec{3}((4 * zeta, 0.0, 4 * xi))
    dN10 = Vec{3}((0.0, 4 * zeta, 4 * eta))

    return (dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10)
end

function compute_jacobian_tet10(dN_dxi, X)
    # J = Σ dN_i ⊗ X_i (tensor products!)
    return sum(dN_dxi[i] ⊗ X[i] for i in 1:10)
end

# Compute strain using tensor products
function compute_strain_from_displacements(dN_dx, u)
    # ∇u = Σ dN_i ⊗ u_i
    gradu = sum(dN_dx[i] ⊗ u[i] for i in 1:10)
    # ε = sym(∇u)
    return symmetric(gradu)
end

# Compute forces using tensors
function compute_nodal_forces_from_stress(dN_dx, sigma)
    # f_i = dN_i · σ
    return ntuple(i -> dN_dx[i] ⋅ sigma, Val(10))
end

# Test
function main()
    println("\n" * "="^70)
    println("Tet10 Assembly Test (CPU)")
    println("="^70)

    # Single Tet10 element
    X = (
        Vec{3}((0.0, 0.0, 0.0)),  # 1
        Vec{3}((1.0, 0.0, 0.0)),  # 2
        Vec{3}((0.0, 1.0, 0.0)),  # 3
        Vec{3}((0.0, 0.0, 1.0)),  # 4
        Vec{3}((0.5, 0.0, 0.0)),  # 5
        Vec{3}((0.5, 0.5, 0.0)),  # 6
        Vec{3}((0.0, 0.5, 0.0)),  # 7
        Vec{3}((0.0, 0.0, 0.5)),  # 8
        Vec{3}((0.5, 0.0, 0.5)),  # 9
        Vec{3}((0.0, 0.5, 0.5))   # 10
    )

    # Displacements: Small perturbation
    u = (
        Vec{3}((0.0, 0.0, 0.0)),      # Node 1
        Vec{3}((0.001, 0.0, 0.0)),    # Node 2 (1mm in x)
        Vec{3}((0.0, 0.0, 0.0)),      # Node 3
        Vec{3}((0.0, 0.0, 0.0)),      # Node 4
        Vec{3}((0.0005, 0.0, 0.0)),   # Node 5
        Vec{3}((0.0005, 0.0, 0.0)),   # Node 6
        Vec{3}((0.0, 0.0, 0.0)),      # Node 7
        Vec{3}((0.0, 0.0, 0.0)),      # Node 8
        Vec{3}((0.0005, 0.0, 0.0)),   # Node 9
        Vec{3}((0.0, 0.0, 0.0))       # Node 10
    )

    material = LinearElastic(200e9, 0.3)

    println("✅ Material: E=$(material.E/1e9) GPa, ν=$(material.ν)")
    println("✅ Pattern: Tensors.jl tensor products (⊗, ⋅, sym)")

    # Assemble element residual
    r_elem = [zero(Vec{3}) for _ in 1:10]

    for (xivec, w) in GAUSS_TET4
        xi, eta, zeta = xivec[1], xivec[2], xivec[3]

        # Shape function derivatives
        dN_dxi = tet10_shape_derivatives(xi, eta, zeta)

        # Jacobian
        J = compute_jacobian_tet10(dN_dxi, X)
        detJ = det(J)
        invJ = inv(J)

        # Physical derivatives: dN/dx = invJ · dN/dxi (tensor contraction)
        dN_dx = ntuple(i -> invJ ⋅ Vec{3}(dN_dxi[i]), Val(10))

        # Compute strain: ε = sym(∇u) where ∇u = Σ dN_i ⊗ u_i
        eps = compute_strain_from_displacements(dN_dx, u)

        # Compute stress
        sigma = compute_stress_3d(material, eps)

        # Compute forces: f_i = dN_i · σ
        f_contrib = compute_nodal_forces_from_stress(dN_dx, sigma)

        # Accumulate
        for i in 1:10
            r_elem[i] += f_contrib[i] * (w * detJ)
        end
    end

    r_total = vcat([r_elem[i][j] for i in 1:10 for j in 1:3]...)

    println("\n📊 Results:")
    println("  ||r||: $(norm(r_total))")
    println("  r[1:6] (node 1-2, x,y,z): $(r_total[1:6])")

    println("\n✅ CPU TEST COMPLETE!")
    println("="^70 * "\n")
end

main()
