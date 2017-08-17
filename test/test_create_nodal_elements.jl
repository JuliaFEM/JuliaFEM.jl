# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Base.Test

using JuliaFEM
using JuliaFEM.Preprocess

@testset "test create nodal elements" begin
    m = Mesh()
    add_node!(m, 1, [0.0, 0.0])
    add_node_to_node_set!(m, :test, 1)
    els = create_nodal_elements(m, "test")
    fel = first(els)
    @test fel.connectivity == [1]
end
