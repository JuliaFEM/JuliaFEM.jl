# This file is a part of JuliaFEM.
# SPDX-License-Identifier: MIT

using SparseArrays
using Test

"""
Extract fenced ```julia blocks from README.md (in order).
"""
function _readme_julia_blocks(readme_path::AbstractString)
    readme = read(readme_path, String)
    blocks = String[]
    for m in eachmatch(r"(?s)```julia\s*\r?\n(.*?)```", readme)
        push!(blocks, String(strip(m.captures[1])))
    end
    return blocks
end

@testset "README.md minimal example executes" begin
    root = joinpath(@__DIR__, "..", "..")
    readme = joinpath(root, "README.md")
    @test isfile(readme)
    blocks = _readme_julia_blocks(readme)
    @test length(blocks) >= 2
    # First fence is the Pkg.install snippet; second is "A modern minimal example".
    M = Module()
    Base.include_string(M, blocks[2])
    K = Core.eval(M, :K)
    f = Core.eval(M, :f)
    @test size(K) == (375, 375)
    @test length(f) == 375
    @test nnz(K) == 19773
end
