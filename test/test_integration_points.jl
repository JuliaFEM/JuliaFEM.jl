# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test

@testset "test integration point" begin
    ip = IP(1, 1.0, sqrt(1.0/3.0)*[-1.0, -1.0])
    strain = [1.0 2.0; 3.0 4.0]
    update!(ip, "strain", 0.0 => strain)
    @test isapprox(ip("strain", 0.0), strain)
    @test isapprox(ip("strain"), strain)
end

