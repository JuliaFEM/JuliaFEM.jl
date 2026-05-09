using Test
using JuliaFEM

@testset "DOF system" begin
    # End-to-end multi-field exercise: DOFSet -> Element template ->
    # DOFHandler -> DOFConnectivity -> field_dof_range / element_dofs.
    include("test_global_field_ranges.jl")
    include("test_multifield_dof_system.jl")
    include("test_hex8_face_dofs.jl")
    include("test_tet4_facet_maps.jl")
    include("test_wedge6_facet_maps.jl")
    include("test_pyr5_facet_maps.jl")
    include("test_tet10_facet_maps.jl")
    include("test_hex20_facet_maps.jl")
    include("test_multicomponent_facet_dofs.jl")
end
