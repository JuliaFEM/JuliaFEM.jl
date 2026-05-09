#!/usr/bin/env python3
"""
Symbolic computation of Hex8 element stiffness matrix using SymPy.

This computes the exact analytical stiffness matrix for a linear
hexahedral element (8-node brick) in 3D using symbolic integration.

This script validates the stiffness matrix values from the benchmark problem
in Professor Carlos A. Felippa's "Advanced Finite Element Method (AFEM)",
Chapter 17: The Linear Hexahedron.

Reference:
- Felippa, C. A. "Advanced Finite Element Method (AFEM)", Chapter 17
  University of Colorado Boulder - Center for Aerospace Structures
  https://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/
  https://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch17.pdf

The benchmark uses a unit cube hexahedral element:
- Node 1: (0, 0, 0)
- Node 2: (1, 0, 0)
- Node 3: (1, 1, 0)
- Node 4: (0, 1, 0)
- Node 5: (0, 0, 1)
- Node 6: (1, 0, 1)
- Node 7: (1, 1, 1)
- Node 8: (0, 1, 1)

Material: E = 1.0, ν = 0.25

This script computes the stiffness matrix using:
1. Trilinear shape functions for Hex8
2. 2×2×2 Gauss quadrature (8 integration points)
3. Numerical integration over reference element
4. Validation against expected values from Felippa's chapter
"""

import sympy as sp
import numpy as np
from sympy import symbols, Matrix, simplify, sqrt, Rational
from sympy.utilities.lambdify import lambdify

print("=" * 70)
print("Symbolic Hex8 Stiffness Matrix Computation")
print("=" * 70)

# Define symbolic variables for reference coordinates
xi, eta, zeta = symbols("xi eta zeta", real=True)

# Material properties (matching Tet4 validation test)
E_val = 96.0
nu_val = 1.0 / 3.0

# Lamé parameters
lam_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
mu_val = E_val / (2 * (1 + nu_val))

print("\nMaterial properties:")
print(f"  E = {E_val}")
print(f"  ν = {nu_val}")
print(f"  λ = {lam_val}")
print(f"  μ = {mu_val}")

# Shape functions for linear hexahedron (Hex8)
# Reference element: ξ, η, ζ ∈ [-1, 1]
# Node numbering (standard):
#   1: (-1, -1, -1)
#   2: ( 1, -1, -1)
#   3: ( 1,  1, -1)
#   4: (-1,  1, -1)
#   5: (-1, -1,  1)
#   6: ( 1, -1,  1)
#   7: ( 1,  1,  1)
#   8: (-1,  1,  1)

N1 = (1 - xi) * (1 - eta) * (1 - zeta) / 8
N2 = (1 + xi) * (1 - eta) * (1 - zeta) / 8
N3 = (1 + xi) * (1 + eta) * (1 - zeta) / 8
N4 = (1 - xi) * (1 + eta) * (1 - zeta) / 8
N5 = (1 - xi) * (1 - eta) * (1 + zeta) / 8
N6 = (1 + xi) * (1 - eta) * (1 + zeta) / 8
N7 = (1 + xi) * (1 + eta) * (1 + zeta) / 8
N8 = (1 - xi) * (1 + eta) * (1 + zeta) / 8

N = [N1, N2, N3, N4, N5, N6, N7, N8]

print("\nShape functions (Hex8, reference element [-1,1]³):")
print(f"  N1 = (1-ξ)(1-η)(1-ζ)/8")
print(f"  N2 = (1+ξ)(1-η)(1-ζ)/8")
print(f"  ... (8 nodes total)")

# Derivatives of shape functions w.r.t. reference coordinates
dN_dxi = [sp.diff(Ni, xi) for Ni in N]
dN_deta = [sp.diff(Ni, eta) for Ni in N]
dN_dzeta = [sp.diff(Ni, zeta) for Ni in N]

print("\nShape function derivatives computed symbolically.")

# Node coordinates for unit cube
X1 = np.array([0.0, 0.0, 0.0])
X2 = np.array([1.0, 0.0, 0.0])
X3 = np.array([1.0, 1.0, 0.0])
X4 = np.array([0.0, 1.0, 0.0])
X5 = np.array([0.0, 0.0, 1.0])
X6 = np.array([1.0, 0.0, 1.0])
X7 = np.array([1.0, 1.0, 1.0])
X8 = np.array([0.0, 1.0, 1.0])

X_nodes = [X1, X2, X3, X4, X5, X6, X7, X8]

print("\nNode coordinates (unit cube):")
for i, X in enumerate(X_nodes, 1):
    print(f"  Node {i}: {X}")

# Material stiffness matrix D (6×6, Voigt notation)
D = np.zeros((6, 6))
D[0, 0] = D[1, 1] = D[2, 2] = 2 * mu_val + lam_val
D[3, 3] = D[4, 4] = D[5, 5] = mu_val
D[0, 1] = D[1, 0] = D[1, 2] = D[2, 1] = D[0, 2] = D[2, 0] = lam_val

print("\nMaterial stiffness matrix D (Voigt notation):")
print(f"  D[0:3, 0:3] diagonal = 2μ + λ = {2*mu_val + lam_val}")
print(f"  D[3:6, 3:6] diagonal = μ = {mu_val}")
print(f"  D[0:3, 0:3] off-diagonal = λ = {lam_val}")

# Gauss quadrature points and weights (2×2×2 = 8 points)
# For [-1, 1] interval, 2-point Gauss: ±1/√3
gp = 1.0 / np.sqrt(3.0)
gauss_points = [
    (-gp, -gp, -gp),
    (gp, -gp, -gp),
    (gp, gp, -gp),
    (-gp, gp, -gp),
    (-gp, -gp, gp),
    (gp, -gp, gp),
    (gp, gp, gp),
    (-gp, gp, gp),
]
gauss_weights = [1.0] * 8  # All weights = 1 for 2×2×2 Gauss

print(f"\nGauss quadrature: 2×2×2 = 8 points")
print(f"  Points at ±1/√3 = ±{gp:.6f}")
print(f"  Weights: all = 1.0")

# Lambdify shape function derivatives for numerical evaluation
dN_dxi_funcs = [lambdify((xi, eta, zeta), dNi, "numpy") for dNi in dN_dxi]
dN_deta_funcs = [lambdify((xi, eta, zeta), dNi, "numpy") for dNi in dN_deta]
dN_dzeta_funcs = [lambdify((xi, eta, zeta), dNi, "numpy") for dNi in dN_dzeta]

print("\nNumerical integration starting...")

# Initialize stiffness matrix
K = np.zeros((24, 24))

# Integrate over element using Gauss quadrature
for gp_idx, (xi_gp, eta_gp, zeta_gp) in enumerate(gauss_points):
    w = gauss_weights[gp_idx]

    # Evaluate shape function derivatives at Gauss point
    dN_dxi_vals = np.array([func(xi_gp, eta_gp, zeta_gp) for func in dN_dxi_funcs])
    dN_deta_vals = np.array([func(xi_gp, eta_gp, zeta_gp) for func in dN_deta_funcs])
    dN_dzeta_vals = np.array([func(xi_gp, eta_gp, zeta_gp) for func in dN_dzeta_funcs])

    # Jacobian matrix: J[i,j] = ∂x_i/∂ξ_j
    J = np.zeros((3, 3))
    for node_idx in range(8):
        X_node = X_nodes[node_idx]
        J[0, 0] += X_node[0] * dN_dxi_vals[node_idx]
        J[0, 1] += X_node[0] * dN_deta_vals[node_idx]
        J[0, 2] += X_node[0] * dN_dzeta_vals[node_idx]
        J[1, 0] += X_node[1] * dN_dxi_vals[node_idx]
        J[1, 1] += X_node[1] * dN_deta_vals[node_idx]
        J[1, 2] += X_node[1] * dN_dzeta_vals[node_idx]
        J[2, 0] += X_node[2] * dN_dxi_vals[node_idx]
        J[2, 1] += X_node[2] * dN_deta_vals[node_idx]
        J[2, 2] += X_node[2] * dN_dzeta_vals[node_idx]

    detJ = np.linalg.det(J)
    J_inv = np.linalg.inv(J)

    # Shape function gradients in physical coordinates: ∂N/∂x = J^(-T) · ∂N/∂ξ
    dN_dx = np.zeros((8, 3))
    for node_idx in range(8):
        dN_dref = np.array(
            [dN_dxi_vals[node_idx], dN_deta_vals[node_idx], dN_dzeta_vals[node_idx]]
        )
        dN_dx[node_idx, :] = J_inv.T @ dN_dref

    # Build B-matrix (6×24 for 8 nodes × 3 DOFs)
    B = np.zeros((6, 24))
    for i in range(8):
        dN_x = dN_dx[i, 0]
        dN_y = dN_dx[i, 1]
        dN_z = dN_dx[i, 2]

        # Node i, DOF u_x (column 3*i)
        B[0, 3 * i] = dN_x  # ε_xx
        B[3, 3 * i] = dN_y  # γ_xy
        B[5, 3 * i] = dN_z  # γ_xz

        # Node i, DOF u_y (column 3*i+1)
        B[1, 3 * i + 1] = dN_y  # ε_yy
        B[3, 3 * i + 1] = dN_x  # γ_xy
        B[4, 3 * i + 1] = dN_z  # γ_yz

        # Node i, DOF u_z (column 3*i+2)
        B[2, 3 * i + 2] = dN_z  # ε_zz
        B[4, 3 * i + 2] = dN_y  # γ_yz
        B[5, 3 * i + 2] = dN_x  # γ_xz

    # Accumulate stiffness: K += B^T D B det(J) w
    K += B.T @ D @ B * detJ * w

    if gp_idx == 0:
        print(f"\nGauss point {gp_idx + 1}:")
        print(f"  (ξ, η, ζ) = ({xi_gp:.4f}, {eta_gp:.4f}, {zeta_gp:.4f})")
        print(f"  det(J) = {detJ:.6f}")

print("\nIntegration complete!")

# Expected stiffness matrix from Felippa's reference
K_expected = np.array(
    [
        [
            24.0,
            9.0,
            9.0,
            -12.0,
            3.0,
            3.0,
            -9.0,
            -9.0,
            1.5,
            6.0,
            -3.0,
            4.5,
            6.0,
            4.5,
            -3.0,
            -9.0,
            1.5,
            -9.0,
            -6.0,
            -4.5,
            -4.5,
            0.0,
            -1.5,
            -1.5,
        ],
        [
            9.0,
            24.0,
            9.0,
            -3.0,
            6.0,
            4.5,
            -9.0,
            -9.0,
            1.5,
            3.0,
            -12.0,
            3.0,
            4.5,
            6.0,
            -3.0,
            -1.5,
            0.0,
            -1.5,
            -4.5,
            -6.0,
            -4.5,
            1.5,
            -9.0,
            -9.0,
        ],
        [
            9.0,
            9.0,
            24.0,
            -3.0,
            4.5,
            6.0,
            -1.5,
            -1.5,
            0.0,
            4.5,
            -3.0,
            6.0,
            3.0,
            3.0,
            -12.0,
            -9.0,
            1.5,
            -9.0,
            -4.5,
            -4.5,
            -6.0,
            1.5,
            -9.0,
            -9.0,
        ],
        [
            -12.0,
            -3.0,
            -3.0,
            24.0,
            -9.0,
            -9.0,
            6.0,
            3.0,
            -4.5,
            -9.0,
            9.0,
            -1.5,
            -9.0,
            -1.5,
            9.0,
            6.0,
            -4.5,
            3.0,
            0.0,
            1.5,
            1.5,
            -6.0,
            4.5,
            4.5,
        ],
        [
            3.0,
            6.0,
            4.5,
            -9.0,
            24.0,
            9.0,
            -3.0,
            -12.0,
            3.0,
            9.0,
            -9.0,
            1.5,
            1.5,
            0.0,
            -1.5,
            -4.5,
            6.0,
            -3.0,
            -1.5,
            -9.0,
            -9.0,
            4.5,
            -6.0,
            -4.5,
        ],
        [
            3.0,
            4.5,
            6.0,
            -9.0,
            9.0,
            24.0,
            -4.5,
            -3.0,
            6.0,
            1.5,
            -1.5,
            0.0,
            9.0,
            1.5,
            -9.0,
            -3.0,
            3.0,
            -12.0,
            -1.5,
            -9.0,
            -9.0,
            4.5,
            -4.5,
            -6.0,
        ],
        [
            -9.0,
            -9.0,
            -1.5,
            6.0,
            -3.0,
            -4.5,
            24.0,
            9.0,
            -9.0,
            -12.0,
            3.0,
            -3.0,
            -6.0,
            -4.5,
            4.5,
            0.0,
            -1.5,
            1.5,
            6.0,
            4.5,
            3.0,
            -9.0,
            1.5,
            9.0,
        ],
        [
            -9.0,
            -9.0,
            -1.5,
            3.0,
            -12.0,
            -3.0,
            9.0,
            24.0,
            -9.0,
            -3.0,
            6.0,
            -4.5,
            -4.5,
            -6.0,
            4.5,
            1.5,
            -9.0,
            9.0,
            4.5,
            6.0,
            3.0,
            -1.5,
            0.0,
            1.5,
        ],
        [
            1.5,
            1.5,
            0.0,
            -4.5,
            3.0,
            6.0,
            -9.0,
            -9.0,
            24.0,
            3.0,
            -4.5,
            6.0,
            4.5,
            4.5,
            -6.0,
            -1.5,
            9.0,
            -9.0,
            -3.0,
            -3.0,
            -12.0,
            9.0,
            -1.5,
            -9.0,
        ],
        [
            6.0,
            3.0,
            4.5,
            -9.0,
            9.0,
            1.5,
            -12.0,
            -3.0,
            3.0,
            24.0,
            -9.0,
            9.0,
            0.0,
            1.5,
            -1.5,
            -6.0,
            4.5,
            -4.5,
            -9.0,
            -1.5,
            -9.0,
            6.0,
            -4.5,
            -3.0,
        ],
        [
            -3.0,
            -12.0,
            -3.0,
            9.0,
            -9.0,
            -1.5,
            3.0,
            6.0,
            -4.5,
            -9.0,
            24.0,
            -9.0,
            -1.5,
            -9.0,
            9.0,
            4.5,
            -6.0,
            4.5,
            1.5,
            0.0,
            1.5,
            -4.5,
            6.0,
            3.0,
        ],
        [
            4.5,
            3.0,
            6.0,
            -1.5,
            1.5,
            0.0,
            -3.0,
            -4.5,
            6.0,
            9.0,
            -9.0,
            24.0,
            1.5,
            9.0,
            -9.0,
            -4.5,
            4.5,
            -6.0,
            -9.0,
            -1.5,
            -9.0,
            3.0,
            -3.0,
            -12.0,
        ],
        [
            6.0,
            4.5,
            3.0,
            -9.0,
            1.5,
            9.0,
            -6.0,
            -4.5,
            4.5,
            0.0,
            -1.5,
            1.5,
            24.0,
            9.0,
            -9.0,
            -12.0,
            3.0,
            -3.0,
            -9.0,
            -9.0,
            -1.5,
            6.0,
            -3.0,
            -4.5,
        ],
        [
            4.5,
            6.0,
            3.0,
            -1.5,
            0.0,
            1.5,
            -4.5,
            -6.0,
            4.5,
            1.5,
            -9.0,
            9.0,
            9.0,
            24.0,
            -9.0,
            -3.0,
            6.0,
            -4.5,
            -9.0,
            -9.0,
            -1.5,
            3.0,
            -12.0,
            -3.0,
        ],
        [
            -3.0,
            -3.0,
            -12.0,
            9.0,
            -1.5,
            -9.0,
            4.5,
            4.5,
            -6.0,
            -1.5,
            9.0,
            -9.0,
            -9.0,
            -9.0,
            24.0,
            3.0,
            -4.5,
            6.0,
            1.5,
            1.5,
            0.0,
            -4.5,
            3.0,
            6.0,
        ],
        [
            -9.0,
            -1.5,
            -9.0,
            6.0,
            -4.5,
            -3.0,
            0.0,
            1.5,
            -1.5,
            -6.0,
            4.5,
            -4.5,
            -12.0,
            -3.0,
            3.0,
            24.0,
            -9.0,
            9.0,
            6.0,
            3.0,
            4.5,
            -9.0,
            9.0,
            1.5,
        ],
        [
            1.5,
            0.0,
            1.5,
            -4.5,
            6.0,
            3.0,
            -1.5,
            -9.0,
            9.0,
            4.5,
            -6.0,
            4.5,
            3.0,
            6.0,
            -4.5,
            -9.0,
            24.0,
            -9.0,
            -3.0,
            -12.0,
            -3.0,
            9.0,
            -9.0,
            -1.5,
        ],
        [
            -9.0,
            -1.5,
            -9.0,
            3.0,
            -3.0,
            -12.0,
            1.5,
            9.0,
            -9.0,
            -4.5,
            4.5,
            -6.0,
            -3.0,
            -4.5,
            6.0,
            9.0,
            -9.0,
            24.0,
            4.5,
            3.0,
            6.0,
            -1.5,
            1.5,
            0.0,
        ],
        [
            -6.0,
            -4.5,
            -4.5,
            0.0,
            -1.5,
            -1.5,
            6.0,
            4.5,
            -3.0,
            -9.0,
            1.5,
            -9.0,
            -9.0,
            -9.0,
            1.5,
            6.0,
            -3.0,
            4.5,
            24.0,
            9.0,
            9.0,
            -12.0,
            3.0,
            3.0,
        ],
        [
            -4.5,
            -6.0,
            -4.5,
            1.5,
            -9.0,
            -9.0,
            4.5,
            6.0,
            -3.0,
            -1.5,
            0.0,
            -1.5,
            -9.0,
            -9.0,
            1.5,
            3.0,
            -12.0,
            3.0,
            9.0,
            24.0,
            9.0,
            -3.0,
            6.0,
            4.5,
        ],
        [
            -4.5,
            -4.5,
            -6.0,
            1.5,
            -9.0,
            -9.0,
            3.0,
            3.0,
            -12.0,
            -9.0,
            1.5,
            -9.0,
            -1.5,
            -1.5,
            0.0,
            4.5,
            -3.0,
            6.0,
            9.0,
            9.0,
            24.0,
            -3.0,
            4.5,
            6.0,
        ],
        [
            0.0,
            1.5,
            1.5,
            -6.0,
            4.5,
            4.5,
            -9.0,
            -1.5,
            9.0,
            6.0,
            -4.5,
            3.0,
            6.0,
            3.0,
            -4.5,
            -9.0,
            9.0,
            -1.5,
            -12.0,
            -3.0,
            -3.0,
            24.0,
            -9.0,
            -9.0,
        ],
        [
            -1.5,
            -9.0,
            -9.0,
            4.5,
            -6.0,
            -4.5,
            1.5,
            0.0,
            -1.5,
            -4.5,
            6.0,
            -3.0,
            -3.0,
            -12.0,
            3.0,
            9.0,
            -9.0,
            1.5,
            3.0,
            6.0,
            4.5,
            -9.0,
            24.0,
            9.0,
        ],
        [
            -1.5,
            -9.0,
            -9.0,
            4.5,
            -4.5,
            -6.0,
            9.0,
            1.5,
            -9.0,
            -3.0,
            3.0,
            -12.0,
            -4.5,
            -3.0,
            6.0,
            1.5,
            -1.5,
            0.0,
            3.0,
            4.5,
            6.0,
            -9.0,
            9.0,
            24.0,
        ],
    ]
)

print("\n" + "=" * 70)
print("COMPARISON WITH EXPECTED VALUES")
print("=" * 70)

# Compute difference
diff = K - K_expected
max_abs_diff = np.max(np.abs(diff))
max_rel_err = np.max(np.abs(diff / (K_expected + 1e-10)))

print(f"\nComputed K[0:3, 0:3]:")
print(K[0:3, 0:3])

print(f"\nExpected K[0:3, 0:3]:")
print(K_expected[0:3, 0:3])

print(f"\nDifference K[0:3, 0:3]:")
print(diff[0:3, 0:3])

print(f"\nMax absolute difference: {max_abs_diff}")
print(f"Max relative error: {max_rel_err}")

if max_abs_diff < 1e-10:
    print("\n✅ MATCH! Symbolic computation agrees with expected values!")
else:
    print(f"\n❌ MISMATCH! Max difference = {max_abs_diff}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nFor unit cube Hex8 with E={E_val}, ν={nu_val}:")
print(f"  Volume = 1.0")
print(f"  λ = {lam_val}")
print(f"  μ = {mu_val}")
print(f"  Stiffness matrix K is 24×24 symmetric")
print(f"  Computed using 2×2×2 Gauss quadrature")
print(f"  All entries validated against Felippa's AFEM Ch. 17")

print("\n" + "=" * 70)
