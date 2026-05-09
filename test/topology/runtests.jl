# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Topology" begin
    include("test_segments.jl")
    include("test_triangles.jl")
    include("test_quadrilaterals.jl")
    include("test_topology_entities_tetrahedron.jl")
    include("test_hexahedra.jl")
    include("test_pyramids.jl")
    include("test_wedges.jl")
    # include("test_topology_type_extraction.jl") -- archived to
    # llm/design/legacy-tests/topology/, targets the older API where
    # entities carried their own topology
    include("test_topological_invariance.jl")
    include("test_entity_dimensions.jl")
    include("test_triangle_reference_coordinates.jl")
end
