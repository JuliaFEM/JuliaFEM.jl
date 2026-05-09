#!/usr/bin/env python3
"""
Symbolic computation of Tet4 element stiffness matrix using SymPy.

This computes the exact analytical stiffness matrix for a linear
tetrahedron element in 3D using symbolic integration.

This script validates the stiffness matrix values from the benchmark problem
in Professor Carlos A. Felippa's "Advanced Finite Element Method (AFEM)",
Chapter 15: The Linear Tetrahedron.

Reference:
- Felippa, C. A. "Advanced Finite Element Method (AFEM)", Chapter 15
  University of Colorado Boulder - Center for Aerospace Structures
  https://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/
  https://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch15.pdf

The benchmark uses an arbitrary tetrahedral element (not reference element)
with nodes at:
- Node 1: (2, 3, 4)
- Node 2: (6, 3, 2)
- Node 3: (2, 5, 1)
- Node 4: (4, 3, 6)

Material: E = 96, ν = 1/3

This script:
1. Shows the reference element computation (for understanding)
2. Computes the stiffness matrix for the actual benchmark geometry
3. Validates against expected values from Felippa's chapter

The original chapter includes Mathematica verification modules for exact
computation. This Python script provides an independent verification using
SymPy symbolic integration.
"""

import sympy as sp
import numpy as np
from sympy import symbols, Matrix, simplify, integrate, lambdify

print("=" * 70)
print("Symbolic Tet4 Stiffness Matrix Computation")
print("=" * 70)

print("\n" + "=" * 70)
print("PART 1: Reference Element (for understanding)")
print("=" * 70)

# Define symbolic variables for reference coordinates
xi, eta, zeta = symbols("xi eta zeta", real=True, nonnegative=True)

# Material properties (symbolic)
E, nu = symbols("E nu", real=True, positive=True)

# Lamé parameters
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

print("\nMaterial properties:")
print(f"  E = Young's modulus")
print(f"  ν = Poisson's ratio")
print(f"  λ = {lam}")
print(f"  μ = {mu}")

# Shape functions for linear tetrahedron (reference element)
N1 = 1 - xi - eta - zeta
N2 = xi
N3 = eta
N4 = zeta

N = [N1, N2, N3, N4]

print("\nShape functions (reference element):")
for i, Ni in enumerate(N, 1):
    print(f"  N{i} = {Ni}")

print("\n" + "=" * 70)
print("PART 2: Actual Geometry from OLD API Test")
print("=" * 70)

# Actual node coordinates from OLD API test
X1 = sp.Matrix([2.0, 3.0, 4.0])
X2 = sp.Matrix([6.0, 3.0, 2.0])
X3 = sp.Matrix([2.0, 5.0, 1.0])
X4 = sp.Matrix([4.0, 3.0, 6.0])

print("\nNode coordinates:")
print(f"  Node 1: {X1.T}")
print(f"  Node 2: {X2.T}")
print(f"  Node 3: {X3.T}")
print(f"  Node 4: {X4.T}")

# Physical coordinates as function of reference coordinates
x = N1 * X1[0] + N2 * X2[0] + N3 * X3[0] + N4 * X4[0]
y = N1 * X1[1] + N2 * X2[1] + N3 * X3[1] + N4 * X4[1]
z = N1 * X1[2] + N2 * X2[2] + N3 * X3[2] + N4 * X4[2]

print("\nPhysical coordinates (x, y, z) as functions of (ξ, η, ζ):")
print(f"  x = {x}")
print(f"  y = {y}")
print(f"  z = {z}")

# Jacobian matrix: J[i,j] = ∂x_i/∂ξ_j
J = sp.Matrix(
    [
        [sp.diff(x, xi), sp.diff(x, eta), sp.diff(x, zeta)],
        [sp.diff(y, xi), sp.diff(y, eta), sp.diff(y, zeta)],
        [sp.diff(z, xi), sp.diff(z, eta), sp.diff(z, zeta)],
    ]
)

print("\nJacobian matrix J = ∂(x,y,z)/∂(ξ,η,ζ):")
print(J)

detJ = J.det()
print(f"\ndet(J) = {detJ}")
print(f"Element volume = det(J) / 6 = {detJ / 6}")

# Inverse Jacobian
J_inv = J.inv()
print("\nInverse Jacobian J^(-1):")
print(sp.simplify(J_inv))

# Shape function derivatives in physical coordinates
# ∂N/∂x = J^(-T) · ∂N/∂ξ
dN_dxi_ref = sp.Matrix([-1, 1, 0, 0])
dN_deta_ref = sp.Matrix([-1, 0, 1, 0])
dN_dzeta_ref = sp.Matrix([-1, 0, 0, 1])

# For each node
dN_dx_list = []
dN_dy_list = []
dN_dz_list = []

for i in range(4):
    dN_dref = sp.Matrix([dN_dxi_ref[i], dN_deta_ref[i], dN_dzeta_ref[i]])
    dN_dphys = J_inv.T * dN_dref
    dN_dx_list.append(dN_dphys[0])
    dN_dy_list.append(dN_dphys[1])
    dN_dz_list.append(dN_dphys[2])

print("\nShape function derivatives in physical coordinates:")
print(f"  ∂N/∂x = {dN_dx_list}")
print(f"  ∂N/∂y = {dN_dy_list}")
print(f"  ∂N/∂z = {dN_dz_list}")

# Build B-matrix
B = sp.zeros(6, 12)

for i in range(4):
    dN_dx = dN_dx_list[i]
    dN_dy = dN_dy_list[i]
    dN_dz = dN_dz_list[i]

    # Node i, DOF u_x (column 3*i)
    B[0, 3 * i] = dN_dx  # ε_xx
    B[3, 3 * i] = dN_dy  # γ_xy
    B[5, 3 * i] = dN_dz  # γ_xz

    # Node i, DOF u_y (column 3*i+1)
    B[1, 3 * i + 1] = dN_dy  # ε_yy
    B[3, 3 * i + 1] = dN_dx  # γ_xy
    B[4, 3 * i + 1] = dN_dz  # γ_yz

    # Node i, DOF u_z (column 3*i+2)
    B[2, 3 * i + 2] = dN_dz  # ε_zz
    B[4, 3 * i + 2] = dN_dy  # γ_yz
    B[5, 3 * i + 2] = dN_dx  # γ_xz

print("\nB-matrix (6×12) constructed for actual geometry")

# Material stiffness matrix D (6×6)
D = sp.zeros(6, 6)
D[0, 0] = D[1, 1] = D[2, 2] = 2 * mu + lam
D[3, 3] = D[4, 4] = D[5, 5] = mu
D[0, 1] = D[1, 0] = D[1, 2] = D[2, 1] = D[0, 2] = D[2, 0] = lam

# Stiffness matrix: K = ∫_Ω B^T D B dV
# For the actual element, we need to account for the Jacobian
# K = ∫ B^T D B × det(J) dV_ref
# Since B and det(J) are CONSTANT for Tet4:
# K = B^T D B × det(J) × (1/6)
# where 1/6 is the volume of the reference element

print("\nComputing stiffness matrix K = ∫ B^T D B dV...")
print("  For Tet4: B and Jacobian are CONSTANT")
print(f"  K = B^T D B × det(J) × (1/6)")
print(f"  det(J) = {detJ}")
print(f"  Element volume = det(J)/6 = {detJ/6}")

K_symbolic = B.T * D * B
K = K_symbolic * detJ * sp.Rational(1, 6)

print("\nStiffness matrix computed!")
print(f"  K is {K.shape[0]}×{K.shape[1]} symbolic matrix")

# Simplify (this should be fast since entries are already simple)
K = sp.simplify(K)

print("\nSimplifying expressions...")

# Substitute numerical values for verification
# Use the same values as OLD API test: E = 96, ν = 1/3
E_val = 96.0
nu_val = 1.0 / 3.0

print("\nSubstituting numerical values:")
print(f"  E = {E_val}")
print(f"  ν = {nu_val}")

# Compute Lamé parameters
lam_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
mu_val = E_val / (2 * (1 + nu_val))

print(f"  λ = {lam_val}")
print(f"  μ = {mu_val}")

# Substitute into K
K_numerical = K.subs([(E, E_val), (nu, nu_val)])

# Convert to float matrix
K_float = np.array(K_numerical).astype(np.float64)

print("\n" + "=" * 70)
print("NUMERICAL STIFFNESS MATRIX (E=96, ν=1/3)")
print("=" * 70)
print("\nK =")
print(K_float)

print("\n" + "=" * 70)
print("COMPARISON WITH OLD API TEST VALUES")
print("=" * 70)

# Expected values from OLD API test
K_expected = np.array(
    [
        [149, 108, 24, -1, 6, 12, -54, -48, 0, -94, -66, -36],
        [108, 344, 54, -24, 104, 42, -24, -216, -12, -60, -232, -84],
        [24, 54, 113, 0, 30, 35, 0, -24, -54, -24, -60, -94],
        [-1, -24, 0, 29, -18, -12, -18, 24, 0, -10, 18, 12],
        [6, 104, 30, -18, 44, 18, 12, -72, -12, 0, -76, -36],
        [12, 42, 35, -12, 18, 29, 0, -24, -18, 0, -36, -46],
        [-54, -24, 0, -18, 12, 0, 36, 0, 0, 36, 12, 0],
        [-48, -216, -24, 24, -72, -24, 0, 144, 0, 24, 144, 48],
        [0, -12, -54, 0, -12, -18, 0, 0, 36, 0, 24, 36],
        [-94, -60, -24, -10, 0, 0, 36, 24, 0, 68, 36, 24],
        [-66, -232, -60, 18, -76, -36, 12, 144, 24, 36, 164, 72],
        [-36, -84, -94, 12, -36, -46, 0, 48, 36, 24, 72, 104],
    ]
)

print("\nExpected K (from OLD API test) =")
print(K_expected)

print("\nDifference (Symbolic - Expected) =")
diff = K_float - K_expected
print(diff)

print("\nMax absolute difference:", np.max(np.abs(diff)))
print("Max relative error:", np.max(np.abs(diff / (K_expected + 1e-10))))

if np.allclose(K_float, K_expected, rtol=1e-6):
    print("\n✅ MATCH! Symbolic computation agrees with OLD API test values!")
else:
    print("\n❌ MISMATCH! Symbolic computation differs from OLD API test values!")
    print("   This means either:")
    print("   1. The OLD API test uses different geometry")
    print("   2. The OLD API has a bug")
    print("   3. This symbolic computation has an error")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nFor reference Tet4 element with E=96, ν=1/3:")
print(f"  Volume = 1/6")
print(f"  λ = {lam_val}")
print(f"  μ = {mu_val}")
print(f"  Stiffness matrix K is 12×12 symmetric")
print(f"  All entries are rational multiples of E")
print(f"  K_symbolic available for arbitrary E, ν")

print("\n" + "=" * 70)
