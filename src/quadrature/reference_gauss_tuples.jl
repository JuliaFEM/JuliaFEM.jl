# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

# Reference-domain Gauss-Legendre tuples for hand-coded integrals. Values match
# `get_quadrature_points` for the same rule (see `gl_tensor_product.jl`, `gl_triangles.jl`),
# including the 3-point triangle rule (`REF_GAUSS_TRIANGLE_ORDER2`).

"Two-point Gauss-Legendre on reference segment `[-1, 1]`; each tuple is `(ξ, w)`; weights sum to `2`."
const REF_GAUSS_SEGMENT_ORDER2 = ((-1.0 / √3, 1.0), (1.0 / √3, 1.0))

"Tensor-product 2×2 Gauss-Legendre on reference square `[-1, 1]²`; each tuple is `(ξ, η, w)`."
const REF_GAUSS_QUAD_2X2 = (
    (-1.0 / √3, -1.0 / √3, 1.0),
    (1.0 / √3, -1.0 / √3, 1.0),
    (1.0 / √3, 1.0 / √3, 1.0),
    (-1.0 / √3, 1.0 / √3, 1.0),
)

"One-point centroid rule on the unit-area-right-triangle natural coordinates; tuple `(ξ, η, w)` with `w = 1/2`."
const REF_GAUSS_TRIANGLE_CENTROID = ((1.0 / 3.0, 1.0 / 3.0, 0.5),)

"""
Three-point Gauss-Legendre rule (degree 2) on the reference triangle `ξ, η ≥ 0`,
`ξ + η ≤ 1` (area `1/2`). Each tuple is `(ξ, η, w)` with `w = 1/6`.

Same weights as `get_quadrature_points(Triangle{3}, GaussLegendre{2}())` in
`gl_triangles.jl` (default 3-point rule); point order matches
`MixedDarcyTet4BoundaryNormalFluxLoad` in `matrix_free/loads.jl`.
"""
const REF_GAUSS_TRIANGLE_ORDER2 = (
    (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
)
