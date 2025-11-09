#!/usr/bin/env julia

# ==============================================================================
# LAGRANGE BASIS FUNCTION GENERATION SCRIPT
# ==============================================================================
#
# This script generates pre-computed Lagrange basis functions for all standard
# finite element types and writes them to src/basis/lagrange_generated.jl
#
# USAGE:
#   cd /path/to/JuliaFEM.jl
#   julia --project=. scripts/generate_lagrange_basis.jl
#
# OUTPUT:
#   src/basis/lagrange_generated.jl (commit this to git!)
#
# WHEN TO RUN:
#   - Adding new element types
#   - Fixing bugs in generation logic
#   - Changing polynomial ansatz strategy
#   - After modifying lagrange_generator.jl
#
# THEORY:
#   See docs/book/lagrange_basis_functions.md for full mathematical details
#
# ==============================================================================

using Pkg
Pkg.activate(".")

using Dates  # For timestamp in generated file header

println("="^80)
println("LAGRANGE BASIS FUNCTION GENERATOR")
println("="^80)
println()

# Load the symbolic generation engine
println("Loading symbolic generator...")
include("../src/basis/lagrange_generator.jl")
println("✓ Generator loaded")
println()

# ==============================================================================
# ELEMENT CATALOG
# ==============================================================================
#
# For each standard Lagrange element, define:
# 1. Name (e.g., "Seg2")
# 2. Description
# 3. Node coordinates in reference element
# 4. Polynomial ansatz (monomial basis)
#
# Reference element conventions:
# - 1D (Seg): u ∈ [-1, 1]
# - 2D (Tri): (u, v) where u,v ≥ 0, u+v ≤ 1
# - 2D (Quad): (u, v) ∈ [-1, 1]²
# - 3D (Tet): (u, v, w) where u,v,w ≥ 0, u+v+w ≤ 1
# - 3D (Hex): (u, v, w) ∈ [-1, 1]³
#
# NOTE: Variable names are u, v, w (not ξ, η, ζ)
#
# ==============================================================================

elements = []

# ------------------------------------------------------------------------------
# 1D ELEMENTS
# ------------------------------------------------------------------------------

# Seg2: 2-node linear segment
push!(elements, (
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
push!(elements, (
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

# ------------------------------------------------------------------------------
# 2D TRIANGULAR ELEMENTS
# ------------------------------------------------------------------------------

# Tri3: 3-node linear triangle
push!(elements, (
    name="Tri3",
    description="3-node linear triangular element",
    coordinates=[
        [0.0, 0.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [0.0, 1.0]   # Node 3
    ],
    ansatz=[
        :(1),    # 1
        :(u),    # u
        :(v)     # v
    ]
))

# Tri6: 6-node quadratic triangle
push!(elements, (
    name="Tri6",
    description="6-node quadratic triangular element",
    coordinates=[
        [0.0, 0.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [0.0, 1.0],  # Node 3
        [0.5, 0.0],  # Node 4 (edge 1-2)
        [0.5, 0.5],  # Node 5 (edge 2-3)
        [0.0, 0.5]   # Node 6 (edge 3-1)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # u
        :(v),      # v
        :(u^2),    # u²
        :(u * v),    # uv
        :(v^2)     # v²
    ]
))

# Tri6: 6-node quadratic triangle
push!(elements, (
    name="Tri6",
    description="6-node quadratic triangular element",
    coordinates=[
        [0.0, 0.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [0.0, 1.0],  # Node 3
        [0.5, 0.0],  # Node 4 (edge 1-2)
        [0.5, 0.5],  # Node 5 (edge 2-3)
        [0.0, 0.5]   # Node 6 (edge 3-1)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # ξ
        :(v),      # η
        :(u^2),    # ξ²
        :(ξ * η),    # ξη
        :(v^2)     # η²
    ]
))

# ------------------------------------------------------------------------------
# 2D QUADRILATERAL ELEMENTS
# ------------------------------------------------------------------------------

# Quad4: 4-node bilinear quadrilateral
# Quad4: 4-node bilinear quadrilateral
push!(elements, (
    name="Quad4",
    description="4-node bilinear quadrilateral element",
    coordinates=[
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0]   # Node 4
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # u
        :(v),      # v
        :(u * v)     # uv
    ]
))

# Quad8: 8-node serendipity quadrilateral (quadratic edges, no center node)
push!(elements, (
    name="Quad8",
    description="8-node serendipity quadrilateral element",
    coordinates=[
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0],  # Node 4
        [0.0, -1.0],  # Node 5 (edge 1-2)
        [1.0, 0.0],  # Node 6 (edge 2-3)
        [0.0, 1.0],  # Node 7 (edge 3-4)
        [-1.0, 0.0]   # Node 8 (edge 4-1)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # u
        :(v),      # v
        :(u^2),    # u²
        :(u * v),    # uv
        :(v^2),    # v²
        :(u^2 * v),  # u²v
        :(u * v^2)   # uv²
    ]
))

# Quad9: 9-node biquadratic quadrilateral (complete quadratic)
push!(elements, (
    name="Quad9",
    description="9-node biquadratic quadrilateral element",
    coordinates=[
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0],  # Node 4
        [0.0, -1.0],  # Node 5 (edge 1-2)
        [1.0, 0.0],  # Node 6 (edge 2-3)
        [0.0, 1.0],  # Node 7 (edge 3-4)
        [-1.0, 0.0],  # Node 8 (edge 4-1)
        [0.0, 0.0]   # Node 9 (center)
    ],
    ansatz=[
        :(1),        # 1
        :(u),        # u
        :(v),        # v
        :(u^2),      # u²
        :(u * v),      # uv
        :(v^2),      # v²
        :(u^2 * v),    # u²v
        :(u * v^2),    # uv²
        :(u^2 * v^2)   # u²v²
    ]
))

# Quad8: 8-node serendipity quadrilateral (quadratic edges, no center node)
push!(elements, (
    name="Quad8",
    description="8-node serendipity quadrilateral element",
    coordinates=[
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0],  # Node 4
        [0.0, -1.0],  # Node 5 (edge 1-2)
        [1.0, 0.0],  # Node 6 (edge 2-3)
        [0.0, 1.0],  # Node 7 (edge 3-4)
        [-1.0, 0.0]   # Node 8 (edge 4-1)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # ξ
        :(v),      # η
        :(u^2),    # ξ²
        :(ξ * η),    # ξη
        :(v^2),    # η²
        :(u^2 * η),  # ξ²η
        :(ξ * v^2)   # ξη²
    ]
))

# Quad9: 9-node biquadratic quadrilateral (complete quadratic)
push!(elements, (
    name="Quad9",
    description="9-node biquadratic quadrilateral element",
    coordinates=[
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [-1.0, 1.0],  # Node 4
        [0.0, -1.0],  # Node 5 (edge 1-2)
        [1.0, 0.0],  # Node 6 (edge 2-3)
        [0.0, 1.0],  # Node 7 (edge 3-4)
        [-1.0, 0.0],  # Node 8 (edge 4-1)
        [0.0, 0.0]   # Node 9 (center)
    ],
    ansatz=[
        :(1),        # 1
        :(u),        # ξ
        :(v),        # η
        :(u^2),      # ξ²
        :(ξ * η),      # ξη
        :(v^2),      # η²
        :(u^2 * η),    # ξ²η
        :(ξ * v^2),    # ξη²
        :(u^2 * v^2)   # ξ²η²
    ]
))

# ------------------------------------------------------------------------------
# 3D TETRAHEDRAL ELEMENTS
# ------------------------------------------------------------------------------

# Tet4: 4-node linear tetrahedron
# Tet4: 4-node linear tetrahedron
push!(elements, (
    name="Tet4",
    description="4-node linear tetrahedral element",
    coordinates=[
        [0.0, 0.0, 0.0],  # Node 1
        [1.0, 0.0, 0.0],  # Node 2
        [0.0, 1.0, 0.0],  # Node 3
        [0.0, 0.0, 1.0]   # Node 4
    ],
    ansatz=[
        :(1),    # 1
        :(u),    # u
        :(v),    # v
        :(w)     # w
    ]
))

# Tet10: 10-node quadratic tetrahedron
push!(elements, (
    name="Tet10",
    description="10-node quadratic tetrahedral element",
    coordinates=[
        [0.0, 0.0, 0.0],  # Node 1
        [1.0, 0.0, 0.0],  # Node 2
        [0.0, 1.0, 0.0],  # Node 3
        [0.0, 0.0, 1.0],  # Node 4
        [0.5, 0.0, 0.0],  # Node 5 (edge 1-2)
        [0.5, 0.5, 0.0],  # Node 6 (edge 2-3)
        [0.0, 0.5, 0.0],  # Node 7 (edge 3-1)
        [0.0, 0.0, 0.5],  # Node 8 (edge 1-4)
        [0.5, 0.0, 0.5],  # Node 9 (edge 2-4)
        [0.0, 0.5, 0.5]   # Node 10 (edge 3-4)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # u
        :(v),      # v
        :(w),      # w
        :(u^2),    # u²
        :(u * v),    # uv
        :(u * w),    # uw
        :(v^2),    # v²
        :(v * w),    # vw
        :(w^2)     # w²
    ]
))

# Tet10: 10-node quadratic tetrahedron
push!(elements, (
    name="Tet10",
    description="10-node quadratic tetrahedral element",
    coordinates=[
        [0.0, 0.0, 0.0],  # Node 1
        [1.0, 0.0, 0.0],  # Node 2
        [0.0, 1.0, 0.0],  # Node 3
        [0.0, 0.0, 1.0],  # Node 4
        [0.5, 0.0, 0.0],  # Node 5 (edge 1-2)
        [0.5, 0.5, 0.0],  # Node 6 (edge 2-3)
        [0.0, 0.5, 0.0],  # Node 7 (edge 3-1)
        [0.0, 0.0, 0.5],  # Node 8 (edge 1-4)
        [0.5, 0.0, 0.5],  # Node 9 (edge 2-4)
        [0.0, 0.5, 0.5]   # Node 10 (edge 3-4)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # ξ
        :(v),      # η
        :(w),      # ζ
        :(u^2),    # ξ²
        :(ξ * η),    # ξη
        :(ξ * ζ),    # ξζ
        :(v^2),    # η²
        :(η * ζ),    # ηζ
        :(w^2)     # ζ²
    ]
))

# ------------------------------------------------------------------------------
# 3D HEXAHEDRAL ELEMENTS
# ------------------------------------------------------------------------------

# Hex8: 8-node trilinear hexahedron (brick)
push!(elements, (
    name="Hex8",
    description="8-node trilinear hexahedral element",
    coordinates=[
        [-1.0, -1.0, -1.0],  # Node 1
        [1.0, -1.0, -1.0],  # Node 2
        [1.0, 1.0, -1.0],  # Node 3
        [-1.0, 1.0, -1.0],  # Node 4
        [-1.0, -1.0, 1.0],  # Node 5
        [1.0, -1.0, 1.0],  # Node 6
        [1.0, 1.0, 1.0],  # Node 7
        [-1.0, 1.0, 1.0]   # Node 8
    ],
    ansatz=[
        :(1),        # 1
        :(u),        # ξ
        :(v),        # η
        :(w),        # ζ
        :(ξ * η),      # ξη
        :(ξ * ζ),      # ξζ
        :(η * ζ),      # ηζ
        :(ξ * η * ζ)     # ξηζ
    ]
))

# Hex20: 20-node serendipity hexahedron (quadratic edges, no face/center nodes)
push!(elements, (
    name="Hex20",
    description="20-node serendipity hexahedral element",
    coordinates=[
        # Corner nodes
        [-1.0, -1.0, -1.0],  # 1
        [1.0, -1.0, -1.0],  # 2
        [1.0, 1.0, -1.0],  # 3
        [-1.0, 1.0, -1.0],  # 4
        [-1.0, -1.0, 1.0],  # 5
        [1.0, -1.0, 1.0],  # 6
        [1.0, 1.0, 1.0],  # 7
        [-1.0, 1.0, 1.0],  # 8
        # Edge nodes (bottom face)
        [0.0, -1.0, -1.0],  # 9
        [1.0, 0.0, -1.0],  # 10
        [0.0, 1.0, -1.0],  # 11
        [-1.0, 0.0, -1.0],  # 12
        # Edge nodes (top face)
        [0.0, -1.0, 1.0],  # 13
        [1.0, 0.0, 1.0],  # 14
        [0.0, 1.0, 1.0],  # 15
        [-1.0, 0.0, 1.0],  # 16
        # Edge nodes (vertical)
        [-1.0, -1.0, 0.0],  # 17
        [1.0, -1.0, 0.0],  # 18
        [1.0, 1.0, 0.0],  # 19
        [-1.0, 1.0, 0.0]   # 20
    ],
    ansatz=[
        :(1),          # 1
        :(u), :(v), :(w),  # linear
        :(u^2), :(v^2), :(w^2),  # pure quadratic
        :(ξ * η), :(ξ * ζ), :(η * ζ),  # bilinear
        :(u^2 * η), :(u^2 * ζ), :(v^2 * ξ), :(v^2 * ζ), :(w^2 * ξ), :(w^2 * η),  # quadratic-linear
        :(ξ * η * ζ),      # trilinear
        :(u^2 * η * ζ), :(ξ * v^2 * ζ), :(ξ * η * w^2)  # quadratic-bilinear
    ]
))

# Hex27: 27-node triquadratic hexahedron (complete quadratic)
push!(elements, (
    name="Hex27",
    description="27-node triquadratic hexahedral element",
    coordinates=[
        # Corner nodes
        [-1.0, -1.0, -1.0],  # 1
        [1.0, -1.0, -1.0],  # 2
        [1.0, 1.0, -1.0],  # 3
        [-1.0, 1.0, -1.0],  # 4
        [-1.0, -1.0, 1.0],  # 5
        [1.0, -1.0, 1.0],  # 6
        [1.0, 1.0, 1.0],  # 7
        [-1.0, 1.0, 1.0],  # 8
        # Edge nodes (bottom face)
        [0.0, -1.0, -1.0],  # 9
        [1.0, 0.0, -1.0],  # 10
        [0.0, 1.0, -1.0],  # 11
        [-1.0, 0.0, -1.0],  # 12
        # Edge nodes (top face)
        [0.0, -1.0, 1.0],  # 13
        [1.0, 0.0, 1.0],  # 14
        [0.0, 1.0, 1.0],  # 15
        [-1.0, 0.0, 1.0],  # 16
        # Edge nodes (vertical)
        [-1.0, -1.0, 0.0],  # 17
        [1.0, -1.0, 0.0],  # 18
        [1.0, 1.0, 0.0],  # 19
        [-1.0, 1.0, 0.0],  # 20
        # Face center nodes
        [0.0, 0.0, -1.0],  # 21 (bottom)
        [0.0, 0.0, 1.0],  # 22 (top)
        [0.0, -1.0, 0.0],  # 23 (front)
        [1.0, 0.0, 0.0],  # 24 (right)
        [0.0, 1.0, 0.0],  # 25 (back)
        [-1.0, 0.0, 0.0],  # 26 (left)
        # Volume center node
        [0.0, 0.0, 0.0]   # 27 (center)
    ],
    ansatz=[
        :(1),          # 1
        :(u), :(v), :(w),  # linear (3)
        :(u^2), :(v^2), :(w^2),  # pure quadratic (3)
        :(ξ * η), :(ξ * ζ), :(η * ζ),  # bilinear (3)
        :(u^2 * η), :(u^2 * ζ), :(v^2 * ξ), :(v^2 * ζ), :(w^2 * ξ), :(w^2 * η),  # quad-linear (6)
        :(ξ * η * ζ),      # trilinear (1)
        :(u^2 * η * ζ), :(ξ * v^2 * ζ), :(ξ * η * w^2),  # quad-bilinear (3)
        :(u^2 * v^2), :(u^2 * w^2), :(v^2 * w^2),  # biquadratic (3)
        :(u^2 * v^2 * ζ), :(u^2 * η * w^2), :(ξ * v^2 * w^2),  # biquad-linear (3)
        :(u^2 * v^2 * w^2)  # triquadratic (1)
    ]
))

# ------------------------------------------------------------------------------
# 3D PYRAMID ELEMENTS
# ------------------------------------------------------------------------------

# Pyr5: 5-node linear pyramid
push!(elements, (
    name="Pyr5",
    description="5-node linear pyramid element",
    coordinates=[
        [-1.0, -1.0, 0.0],  # Node 1 (base)
        [1.0, -1.0, 0.0],  # Node 2 (base)
        [1.0, 1.0, 0.0],  # Node 3 (base)
        [-1.0, 1.0, 0.0],  # Node 4 (base)
        [0.0, 0.0, 1.0]   # Node 5 (apex)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # ξ
        :(v),      # η
        :(w),      # ζ
        :(ξ * η)     # ξη (needed for symmetry)
    ]
))

# ------------------------------------------------------------------------------
# 3D WEDGE ELEMENTS (Prism/Pentahedron)
# ------------------------------------------------------------------------------

# Wedge6: 6-node linear wedge (triangular prism)
push!(elements, (
    name="Wedge6",
    description="6-node linear wedge element (triangular prism)",
    coordinates=[
        [0.0, 0.0, -1.0],  # Node 1 (bottom triangle)
        [1.0, 0.0, -1.0],  # Node 2 (bottom triangle)
        [0.0, 1.0, -1.0],  # Node 3 (bottom triangle)
        [0.0, 0.0, 1.0],  # Node 4 (top triangle)
        [1.0, 0.0, 1.0],  # Node 5 (top triangle)
        [0.0, 1.0, 1.0]   # Node 6 (top triangle)
    ],
    ansatz=[
        :(1),      # 1
        :(u),      # ξ
        :(v),      # η
        :(w),      # ζ
        :(ξ * ζ),    # ξζ
        :(η * ζ)     # ηζ
    ]
))

# Wedge15: 15-node quadratic wedge
push!(elements, (
    name="Wedge15",
    description="15-node quadratic wedge element",
    coordinates=[
        # Bottom triangle
        [0.0, 0.0, -1.0],  # 1
        [1.0, 0.0, -1.0],  # 2
        [0.0, 1.0, -1.0],  # 3
        # Top triangle
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [0.0, 1.0, 1.0],  # 6
        # Mid-edge nodes (bottom triangle)
        [0.5, 0.0, -1.0],  # 7
        [0.5, 0.5, -1.0],  # 8
        [0.0, 0.5, -1.0],  # 9
        # Mid-edge nodes (top triangle)
        [0.5, 0.0, 1.0],  # 10
        [0.5, 0.5, 1.0],  # 11
        [0.0, 0.5, 1.0],  # 12
        # Mid-edge nodes (vertical)
        [0.0, 0.0, 0.0],  # 13
        [1.0, 0.0, 0.0],  # 14
        [0.0, 1.0, 0.0]   # 15
    ],
    ansatz=[
        :(1),        # 1
        :(u), :(v), :(w),  # linear (3)
        :(u^2), :(v^2), :(w^2),  # pure quadratic (3)
        :(ξ * η),      # triangle bilinear (1)
        :(ξ * ζ), :(η * ζ),  # prism bilinear (2)
        :(u^2 * ζ), :(v^2 * ζ), :(ξ * η * ζ),  # mixed (3)
        :(ξ * w^2), :(η * w^2)  # mixed (2)
    ]
))

println("Element catalog loaded: $(length(elements)) element types")
println()

# ==============================================================================
# GENERATION LOOP
# ==============================================================================

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
println(output, "#   scripts/generate_lagrange_basis.jl")
println(output, "#")
println(output, "# To regenerate (e.g., after adding new element types):")
println(output, "#   cd /path/to/JuliaFEM.jl")
println(output, "#   julia --project=. scripts/generate_lagrange_basis.jl")
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

for (i, elem) in enumerate(elements)
    println("[$i/$(length(elements))] Generating $(elem.name)...")

    try
        # Convert coordinates to proper format (tuples, not vectors)
        coords = [tuple(coord...) for coord in elem.coordinates]

        # Build polynomial ansatz as single expression: term1 + term2 + ...
        # The ansatz is a vector of terms [:1, :ξ, :η, ...] → combine with +
        if length(elem.ansatz) == 1
            ansatz_expr = elem.ansatz[1]
        else
            ansatz_expr = Expr(:call, :+, elem.ansatz...)
        end

        # Generate basis code using symbolic engine
        # Returns an Expr (quoted code)
        basis_code_expr = create_basis(
            Symbol(elem.name),
            elem.description,
            coords,
            ansatz_expr
        )

        # Convert Expr to readable Julia code string
        basis_code_str = string(basis_code_expr)

        # Pretty-print: The generated code is in a quote block, extract inner code
        # Handle both `quote ... end` and `begin ... end` forms
        basis_code_str = replace(basis_code_str, r"^(quote|begin)\s+" => "")
        basis_code_str = replace(basis_code_str, r"\s*end$" => "")

        # Write to output with nice formatting
        println(output, "# " * "─"^78)
        println(output, "# $(elem.name): $(elem.description)")
        println(output, "# " * "─"^78)
        println(output)
        println(output, basis_code_str)
        println(output)

    catch e
        println("  ⚠ Error generating $(elem.name):")
        println("     $e")
        if isa(e, ErrorException)
            # Print stacktrace for debugging
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

# ==============================================================================
# WRITE OUTPUT FILE
# ==============================================================================

output_path = joinpath(@__DIR__, "..", "src", "basis", "lagrange_generated.jl")
println("Writing to: $output_path")

output_content = String(take!(output))
write(output_path, output_content)

println("✓ Generation complete!")
println()
println("Generated $(length(elements)) element types:")
for elem in elements
    println("  - $(elem.name): $(elem.description)")
end
println()
println("Next steps:")
println("  1. Review: src/basis/lagrange_generated.jl")
println("  2. Update module to include this file")
println("  3. Remove __precompile__(false) from main module")
println("  4. Test: julia --project=. -e 'using JuliaFEM'")
println("  5. Run tests: julia --project=. -e 'using Pkg; Pkg.test()'")
println("  6. Commit: git add src/basis/lagrange_generated.jl && git commit")
println()
println("="^80)
