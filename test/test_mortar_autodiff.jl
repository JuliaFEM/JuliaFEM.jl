# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Testing

function get_testproblems(u, la)
    nodes = Dict{Int64, Node}(
        1 => [0.0, 0.0],
        2 => [2.0, 0.0],
        3 => [1.0, 2.0],
        4 => [0.0, 2.0],
        5 => [2.5, 0.0],
        6 => [4.5, 0.0],
        7 => [4.5, 1.0],
        8 => [2.5, 1.0])
    displacement = Dict{Int64, Vector{Float64}}()
    reaction_force = Dict{Int64, Vector{Float64}}()
    for i=1:8
        displacement[i] = u[:,i]
        reaction_force[i] = la[:,i]
    end
    bc5 = Seg2([3, 2])
    bc6 = Seg2([8, 5])
    update!([bc5, bc6], "geometry", nodes)
    update!([bc5, bc6], "displacement", displacement)
    update!([bc5, bc6], "reaction force", reaction_force)
    bc5["master elements"] = [bc6]
    contact1 = Problem(Mortar, "contact between bodies", 2, "displacement")
    contact2 = Problem(Mortar, "contact between bodies", 2, "displacement")
    contact2.properties.formulation = :forwarddiff
    contact2.assembly.u = vec(u)
    contact2.assembly.la = vec(la)
    push!(contact1, bc5, bc6)
    push!(contact2, bc5, bc6)
    return contact1, contact2
end

#= TODO: Fix test
@testset "test linearization of contact force in undeformed state" begin
    u = zeros(2, 8)
    la = zeros(2, 8)
    contact1, contact2 = get_testproblems(u, la)
    assemble!(contact1, 0.0)
    assemble!(contact2, 0.0)
    @test isapprox(full(contact1.assembly.C1), full(contact2.assembly.C1))
    @test isapprox(full(contact1.assembly.K), full(contact2.assembly.K))
end
=#

