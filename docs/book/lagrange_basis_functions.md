---
title: "Lagrange Basis Functions"
subtitle: "Mathematical foundations of finite element interpolation"
description: "Complete derivation of Lagrange basis functions using Vandermonde matrix method"
date: 2025-11-09
author: "Jukka Aho"
categories: ["theory", "mathematics", "fem"]
keywords: ["lagrange basis", "shape functions", "interpolation", "vandermonde matrix", "fem theory"]
audience: "researchers and advanced users"
level: "expert"
type: "theory"
series: "The JuliaFEM Book"
chapter: "Part I: Foundations"
math: true
prerequisites: ["linear algebra", "numerical analysis", "fem basics"]
---

# Lagrange Basis Functions in JuliaFEM

**Date:** November 9, 2025  
**Author:** JuliaFEM Development Team

## Introduction

Lagrange basis functions are the foundation of the Finite Element Method. They provide a systematic way to construct polynomial interpolation functions that satisfy the **Kronecker delta property**: the basis function associated with node $i$ equals 1 at that node and 0 at all other nodes.

$$N_i(\mathbf{x}_j) = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

This property makes it trivial to interpolate field values: $u(\mathbf{x}) = \sum_i u_i N_i(\mathbf{x})$ where $u_i$ are nodal values.

## Mathematical Foundation

### Vandermonde Matrix Method

Given:

- $n$ nodes with coordinates $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ in reference element
- A polynomial basis (ansatz) $\{p_1(\mathbf{x}), p_2(\mathbf{x}), \ldots, p_n(\mathbf{x})\}$

We seek coefficients $\alpha_{ij}$ such that:

$$N_i(\mathbf{x}) = \sum_{j=1}^{n} \alpha_{ij} p_j(\mathbf{x})$$

The Kronecker delta property gives us:

$$N_i(\mathbf{x}_k) = \sum_{j=1}^{n} \alpha_{ij} p_j(\mathbf{x}_k) = \delta_{ik}$$

This is a linear system: $\mathbf{V} \boldsymbol{\alpha}_i = \mathbf{e}_i$

Where the **Vandermonde matrix** is:

$$V_{kj} = p_j(\mathbf{x}_k)$$

And $\mathbf{e}_i$ is the $i$-th unit vector.

### Example: 1D Linear Element (Seg2)

**Ansatz:** $p(\xi) = 1 + \xi$ (complete linear polynomial)

**Nodes:** $\xi_1 = 0$, $\xi_2 = 1$

**Vandermonde matrix:**

$$\mathbf{V} = \begin{bmatrix}
p_1(\xi_1) & p_2(\xi_1) \\
p_1(\xi_2) & p_2(\xi_2)
\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}$$

**Solve for $N_1$:** $\mathbf{V} \boldsymbol{\alpha}_1 = [1, 0]^T$

$$\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} \alpha_{11} \\ \alpha_{12} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

Solution: $\alpha_{11} = 1$, $\alpha_{12} = -1$

Therefore: $N_1(\xi) = 1 \cdot 1 + (-1) \cdot \xi = 1 - \xi$ ✓

**Solve for $N_2$:** $\mathbf{V} \boldsymbol{\alpha}_2 = [0, 1]^T$

Solution: $\alpha_{21} = 0$, $\alpha_{22} = 1$

Therefore: $N_2(\xi) = 0 \cdot 1 + 1 \cdot \xi = \xi$ ✓

**Verification:**
- $N_1(0) = 1$, $N_1(1) = 0$ ✓
- $N_2(0) = 0$, $N_2(1) = 1$ ✓
- $N_1(\xi) + N_2(\xi) = 1$ (partition of unity) ✓

## Polynomial Completeness

The ansatz polynomial must be **complete** to the desired order:

| Order | 1D | 2D | 3D | Nodes Required |
|-------|----|----|-----|----------------|
| Linear | $1 + \xi$ | $1 + \xi + \eta$ | $1 + \xi + \eta + \zeta$ | $d+1$ |
| Quadratic | $1 + \xi + \xi^2$ | $1 + \xi + \eta + \xi^2 + \xi\eta + \eta^2$ | ... | $(d+1)(d+2)/2$ |

**Example for 2D Triangle (Tri3):**

Ansatz: $p(\xi, \eta) = 1 + \xi + \eta$ (complete linear in 2D)

This is the **minimal** complete polynomial for 3 nodes.

## Implementation in JuliaFEM

### Automatic Generation Process

```julia
# 1. Define element geometry
coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]  # Tri3 nodes

# 2. Define ansatz polynomial
ansatz = :(1 + u + v)  # Complete linear in 2D

# 3. Build Vandermonde matrix
V[i,j] = eval_polynomial_term(ansatz_terms[j], coords[i])

# 4. For each node i:
coeffs = V \ e_i  # Solve linear system
N_i = sum(coeffs[j] * ansatz_terms[j])  # Construct basis function

# 5. Symbolic differentiation
∂N_i/∂ξ = differentiate(N_i, :u)
∂N_i/∂η = differentiate(N_i, :v)
```

### Why This Works

1. **Completeness:** Ansatz spans full polynomial space of given order
2. **Linear Independence:** Vandermonde matrix is non-singular for distinct nodes
3. **Interpolation Property:** Follows directly from $\mathbf{V} \boldsymbol{\alpha}_i = \mathbf{e}_i$

### Derivatives

Once we have $N_i(\xi, \eta, \zeta)$ symbolically, derivatives are straightforward:

$$\frac{\partial N_i}{\partial \xi}, \frac{\partial N_i}{\partial \eta}, \frac{\partial N_i}{\partial \zeta}$$

These are computed **once** symbolically, then **pre-compiled** into efficient Julia code.

## Standard Lagrange Elements in JuliaFEM

### 1D Elements
- **Seg2**: Linear (2 nodes)
- **Seg3**: Quadratic (3 nodes, mid-edge node)

### 2D Elements
- **Tri3**: Linear triangle (3 corner nodes)
- **Tri6**: Quadratic triangle (6 nodes: 3 corners + 3 mid-edges)
- **Quad4**: Bilinear quadrilateral (4 corner nodes)
- **Quad8**: Serendipity quadrilateral (8 nodes: 4 corners + 4 mid-edges)
- **Quad9**: Biquadratic quadrilateral (9 nodes: 4 corners + 4 mid-edges + 1 center)

### 3D Elements
- **Tet4**: Linear tetrahedron (4 corner nodes)
- **Tet10**: Quadratic tetrahedron (10 nodes: 4 corners + 6 mid-edges)
- **Hex8**: Trilinear hexahedron (8 corner nodes)
- **Hex20**: Serendipity hexahedron (20 nodes: 8 corners + 12 mid-edges)
- **Hex27**: Triquadratic hexahedron (27 nodes: full tensor product)
- **Pyr5**: Linear pyramid (5 nodes)
- **Wedge6**: Linear wedge/prism (6 nodes)
- **Wedge15**: Quadratic wedge (15 nodes)

## Pre-Generation vs Runtime Generation

### Historical Approach (JuliaFEM ≤ 0.5.1)

```julia
# At package load time:
create_basis_and_eval(:Tet10, "...", coords, ansatz)
# - Builds Vandermonde matrix
# - Solves n linear systems
# - Symbolic differentiation
# - Simplification
# - Code generation with eval()
# Result: __precompile__(false) - slow loading
```

**Problems:**
- ❌ Symbolic math every package load (100+ ms)
- ❌ Cannot precompile (`eval()` at module scope)
- ❌ Opaque code generation
- ❌ Hard to debug

### Modern Approach (JuliaFEM ≥ 1.0)

```julia
# Once, during development:
scripts/generate_lagrange_basis.jl
# - Computes all bases symbolically
# - Writes clean Julia code to src/basis/lagrange_generated.jl

# At package load time:
include("basis/lagrange_generated.jl")
# - Just parses pre-written Julia code
# - Fully precompilable
# - Zero symbolic computation
```

**Benefits:**
- ✅ Instant package loading
- ✅ Full precompilation
- ✅ Readable generated code
- ✅ Easy to debug
- ✅ Version controlled (can review changes)

## Numerical Stability

### Vandermonde Matrix Conditioning

The Vandermonde matrix can be ill-conditioned for:
- High-order polynomials ($p > 5$)
- Poorly distributed nodes
- Reference elements far from unit cube/simplex

**JuliaFEM's approach:**
- Use canonical reference elements (unit cube $[-1,1]^d$ or unit simplex)
- Lagrange elements rarely exceed order 3 in practice
- For high-order: Consider hierarchical bases (not Lagrange)

### Verification

Generated basis functions are verified by:
1. **Kronecker delta property:** $N_i(\mathbf{x}_j) = \delta_{ij}$
2. **Partition of unity:** $\sum_i N_i(\mathbf{x}) = 1$ everywhere
3. **Derivative correctness:** Compare symbolic vs AD

See `test/test_basis_functions.jl` for comprehensive tests.

## References

1. Hughes, T.J.R., "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis", Dover, 2000
2. Zienkiewicz, O.C. and Taylor, R.L., "The Finite Element Method", Volumes 1-3, Butterworth-Heinemann, 2000
3. Szabó, B. and Babuška, I., "Finite Element Analysis", Wiley, 1991

## See Also

- `scripts/generate_lagrange_basis.jl` - Generation script
- `src/basis/lagrange_generated.jl` - Generated code (do not edit manually)
- `src/basis/lagrange_generator.jl` - Generator functions (symbolic engine)
- `benchmarks/tet10_derivatives_benchmark.jl` - Performance analysis (manual vs AD)
