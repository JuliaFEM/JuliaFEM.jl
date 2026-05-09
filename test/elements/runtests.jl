# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Element Interpolation" begin
    include("test_interpolate_local_fields.jl")
    include("test_interpolate_field_value.jl")
    include("test_extract_element_dofs.jl")
end
