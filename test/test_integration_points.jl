# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

@testset "test integration point" begin
    a = sqrt(1.0/3.0)
    ip = IP(1, 1.0, (a, -a))
    strain = [1.0 2.0; 3.0 4.0]
    update!(ip, "strain", 0.0 => strain)
    @test isapprox(ip("strain", 0.0), strain)
    @test isapprox(ip("strain"), strain)
end

