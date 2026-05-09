# SPDX-FileCopyrightText: 2015-2026 Jukka Aho
# SPDX-License-Identifier: MIT

using Test
using JuliaFEM

@testset "Reference benchmarks (analytical / literature)" begin
    include("test_analytical_elasticity.jl")
    include("test_analytical_heat.jl")
end
