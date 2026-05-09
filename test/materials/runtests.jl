# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Materials" begin
    include("test_state_variables.jl")
    include("test_orthotropic_linear_elastic.jl")
    include("test_traits.jl")
    include("test_global_material_cache.jl")
    include("test_field_traits.jl")
    include("test_plasticity_integration.jl")
    include("test_j2_isotropic_plasticity.jl")
    include("test_advanced_materials.jl")
    include("test_cantilever_material_cache_zero_allocations.jl")
    include("test_assembly_workspace_refactor.jl")
end
