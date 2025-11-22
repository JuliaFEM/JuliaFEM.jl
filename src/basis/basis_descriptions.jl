# Catalog of basis descriptions used by the generator.
#
# Each entry provides:
# - name: legacy short name (e.g., "Tri3")
# - description: human-readable text
# - topology: reference topology type (with node count parameter)
# - basis: basis family and order (Lagrange{P}, Serendipity{P}, etc.)
# - ansatz: polynomial terms used to build the Vandermonde system

const BASIS_DESCRIPTIONS = VandermondeBasisDescription[]

# 1D SEGMENTS
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Seg2",
    description="2-node linear segment element",
    family=Lagrange{1},
    topology=Segment{2},
    ansatz=(:(1), :(u))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Seg3",
    description="3-node quadratic segment element",
    family=Lagrange{2},
    topology=Segment{3},
    ansatz=(:(1), :(u), :(u^2))
))

# 2D TRIANGLES
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tri3",
    description="3-node linear triangular element",
    family=Lagrange{1},
    topology=Triangle{3},
    ansatz=(:(1), :(u), :(v))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tri6",
    description="6-node quadratic triangular element",
    family=Lagrange{2},
    topology=Triangle{6},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u * v), :(v^2))
))

# 2D QUADS
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Quad4",
    description="4-node bilinear quadrilateral element",
    family=Lagrange{1},
    topology=Quadrilateral{4},
    ansatz=(:(1), :(u), :(v), :(u * v))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Quad8",
    description="8-node serendipity quadrilateral element",
    family=Serendipity{2},
    topology=Quadrilateral{8},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u * v), :(v^2), :(u^2 * v), :(u * v^2))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Quad9",
    description="9-node biquadratic quadrilateral element",
    family=Lagrange{2},
    topology=Quadrilateral{9},
    ansatz=(:(1), :(u), :(v), :(u^2), :(u * v), :(v^2), :(u^2 * v), :(u * v^2), :(u^2 * v^2))
))

# 3D TETS
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tet4",
    description="4-node linear tetrahedral element",
    family=Lagrange{1},
    topology=Tetrahedron{4},
    ansatz=(:(1), :(u), :(v), :(w))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Tet10",
    description="10-node quadratic tetrahedral element",
    family=Lagrange{2},
    topology=Tetrahedron{10},
    ansatz=(:(1), :(u), :(v), :(w), :(u^2), :(v^2), :(w^2), :(u * v), :(u * w), :(v * w))
))

# 3D HEXES
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Hex8",
    description="8-node trilinear hexahedral element",
    family=Lagrange{1},
    topology=Hexahedron{8},
    ansatz=(:(1), :(u), :(v), :(w), :(u * v), :(u * w), :(v * w), :(u * v * w))
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Hex20",
    description="20-node serendipity hexahedral element",
    family=Serendipity{2},
    topology=Hexahedron{20},
    ansatz=(
        :(1), :(u), :(v), :(w),
        :(u * v), :(u * w), :(v * w),
        :(u^2), :(v^2), :(w^2),
        :(u^2 * v), :(u * v^2), :(u^2 * w), :(u * w^2), :(v^2 * w), :(v * w^2)
    )
))

push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Hex27",
    description="27-node triquadratic hexahedral element",
    family=Lagrange{2},
    topology=Hexahedron{27},
    ansatz=(
        :(1), :(u), :(v), :(w),
        :(u^2), :(v^2), :(w^2),
        :(u * v), :(u * w), :(v * w),
        :(u^2 * v), :(u * v^2), :(u^2 * w), :(u * w^2), :(v^2 * w), :(v * w^2),
        :(u^2 * v^2), :(u^2 * w^2), :(v^2 * w^2),
        :(u^2 * v * w), :(u * v^2 * w), :(u * v * w^2), :(u^2 * v^2 * w^2)
    )
))

# 3D PYRAMID
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Pyr5",
    description="5-node linear pyramid element",
    family=Lagrange{1},
    topology=Pyramid{5},
    ansatz=(:(1), :(u), :(v), :(w))
))

# 3D WEDGE / PRISM
push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
    name="Wedge6",
    description="6-node linear wedge (prism) element",
    family=Lagrange{1},
    topology=Wedge{6},
    ansatz=(:(1), :(u), :(v), :(w), :(u * w), :(v * w))
))

# TODO: Wedge15 ansatz needs proper polynomial basis for triangular prism
# The standard Lagrange tensor product doesn't work - need special formulation
# push!(BASIS_DESCRIPTIONS, VandermondeBasisDescription(
#     name="Wedge15",
#     description="15-node quadratic wedge (prism) element",
#     family=Lagrange{2},
#     topology=Wedge{15},
#     ansatz=(
#         :(1), :(u), :(v), :(w),
#         :(u^2), :(v^2), :(w^2), :(u * v), :(u * w), :(v * w),
#         :(u^2 * w), :(v^2 * w), :(u * v * w), :(u^2 * v), :(u * v^2)
#     )
# ))
