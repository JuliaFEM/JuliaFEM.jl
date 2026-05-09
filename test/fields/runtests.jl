# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Fields" begin
    include("test_local_field.jl")
    include("test_field_specs.jl")
    include("test_fields_api.jl")
end
