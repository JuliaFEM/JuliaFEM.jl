# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Test
importall Base
import JuliaFEM: get_basis, get_dbasis, get_integration_points

type MyQuad4 <: AbstractElement
end

function get_basis(element::Element{MyQuad4}, ip, time)
    1/4*[(1-ip[1])*(1-ip[2]) (1+ip[1])*(1-ip[2]) (1+ip[1])*(1+ip[2]) (1-ip[1])*(1+ip[2])]
end

function get_dbasis(element::Element{MyQuad4}, ip, time)
    1/4*[-(1-ip[2])    (1-ip[2])   (1+ip[2])  -(1+ip[2])
         -(1-ip[1])   -(1+ip[1])   (1+ip[1])   (1-ip[1])]
end

function get_integration_points(element::MyQuad4)
    [
        (1.0, 1.0/sqrt(3.0)*[-1, -1]),
        (1.0, 1.0/sqrt(3.0)*[ 1, -1]),
        (1.0, 1.0/sqrt(3.0)*[ 1,  1]),
        (1.0, 1.0/sqrt(3.0)*[-1,  1])
    ]
end

function length(element::Element{MyQuad4})
    return 4
end

function size(element::Element{MyQuad4})
    return (2, 4)
end

@testset "test new element" begin
    el = Element(MyQuad4)
    el["geometry"] = Vector{Float64}[[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]]
    el["displacement"] = Vector{Float64}[[0.0,0.0], [0.0,0.0], [1.0,0.0], [0.0,0.0]]
    @test isapprox(el("geometry", [0.0, 0.0], 0.0), [0.5, 0.5])
    @test isapprox(el("displacement", [0.0, 0.0], 0.0), [0.25, 0.0])
    el["temperature thermal conductivity"] = 6.0
    dim = length(el)
    K = zeros(dim, dim)
    A = 0.0
    time = 0.0
    for ip in get_integration_points(el)
        dN = el(ip, time, Val{:Grad})
        detJ = el(ip, time, Val{:detJ})
        w = ip.weight*detJ
        c = el("temperature thermal conductivity", ip, time)
        K += w*c*dN'*dN
        A += w
    end
    @test isapprox(A, 1.0)
    K_expected = [
        4.0  -1.0  -2.0  -1.0
       -1.0   4.0  -1.0  -2.0
       -2.0  -1.0   4.0  -1.0
       -1.0  -2.0  -1.0   4.0]
    @test isapprox(K, K_expected)
end

