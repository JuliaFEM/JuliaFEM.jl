# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Test

@testset "renumber element nodes" begin
    mesh = Mesh()
    add_element!(mesh, 1, :Tet10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mapping = Dict{Symbol, Vector{Int}}(
        :Tet10 => [1, 2, 4, 3, 5, 6, 7, 8, 9, 10])
    reorder_element_connectivity!(mesh, mapping)
    @test mesh.elements[1] == [1, 2, 4, 3, 5, 6, 7, 8, 9, 10]
    invmapping = Dict{Symbol, Vector{Int}}()
    invmapping[:Tet10] = invperm(mapping[:Tet10])
    reorder_element_connectivity!(mesh, invmapping)
    @test mesh.elements[1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
end

function get_volume(element::Element, time=0.0)
    V = 0.0
    for ip in get_integration_points(element)
        V += ip.weight*element(ip, time, Val{:detJ})
    end
    return V
end

function get_volume(elements::Vector{Element}, time=0.0)
    return sum([get_volume(element, time) for element in elements])
end

#=
@testset "Hex8 element connectivity order" begin
    fn = Pkg.dir("JuliaFEM") * "/test/testdata/rod_short.med"
    mesh = aster_read_mesh(fn, "SHORT_ROD_RECTANGLE_HE8_1ELEM")
    # 1. check volume of element
    rod = create_elements(mesh, "ROD")
    V = get_volume(rod)
    V_expected = 0.01^2*0.2
    info("Volume of rod = $V, expected = $V_expected")
    @test isapprox(V, V_expected)
    # 2. put some field value and calculate flux in gauss points
    T = Dict{Int64, Float64}(
        1 => 100.0, 2 => 100.0, 3 => 100.0, 4 => 100.0,
        5 => 200.0, 6 => 300.0, 7 => 400.0, 8 => 500.0)
    update!(rod, "temperature", T)
    # it has been verified using code aster that flux in integration
    # points is
    FLUX_ELGA = Dict{Int, Vector{Float64}}(
        1 => [-4.08493649053890E+04, -2.11324865405187E+05,  1.05662432702594E+05],
        2 => [-4.08493649053890E+04, -7.88675134594813E+05,  3.94337567297406E+05],
        3 => [-6.97168783648703E+04, -2.11324865405187E+05,  1.05662432702594E+05],
        4 => [-6.97168783648703E+04, -7.88675134594813E+05,  3.94337567297406E+05],
        5 => [-5.52831216351297E+04, -2.11324865405187E+05,  1.05662432702594E+05],
        6 => [-5.52831216351297E+04, -7.88675134594813E+05,  3.94337567297406E+05],
        7 => [-8.41506350946110E+04, -2.11324865405187E+05,  1.05662432702594E+05],
        8 => [-8.41506350946110E+04, -7.88675134594813E+05,  3.94337567297406E+05])
    # flux is q̄(ξ) = -k∇T
    element = first(rod)
    k = -50.0
    flux(xi, time) = -k*vec(element("temperature", xi, time, Val{:Grad}))
    weights = ones(8)
    # code aster integration points (FPG8)
    a = -1.0/sqrt(3.0)
    points = Vector{Float64}[
        [-a, -a, -a],
        [-a, -a,  a],
        [-a,  a, -a],
        [-a,  a,  a],
        [ a, -a, -a],
        [ a, -a,  a],
        [ a,  a, -a],
        [ a,  a,  a]]
    for i=1:8
        q1 = flux(points[i], 0.0)
        q2 = FLUX_ELGA[i]
        rtol = norm(q1-q2)/max(norm(q1),norm(q2))*100.0
        @printf "ip   %i flux, JF: (% e,% e,% e), CA: (% e,% e,% e), rtol: %10.6f %%\n" i q1... q2... rtol
        @test rtol < 0.05
    end
end
=#

