# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/FEMQuad.jl/blob/master/LICENSE

# Tensorial quadrature rules in 1, 2, 3 dimensions

function tensor_product(w::Tuple, p::Tuple, dim::Int)
    @assert length(w) == length(p)

    N = length(w)
    weights = Float64[]
    points = NTuple{dim, Float64}[]
    for i in CartesianIndices(ntuple(i -> 1:N, dim))
        push!(weights, prod(w[k] for k in Tuple(i)))
        push!(points, ntuple(k -> p[i[k]], dim))
    end
    return zip(Tuple(weights), Tuple(points))
end


names = [:GLSEG, :GLQUAD, :GLHEX]
names2 = ["segment", "quadrilateral", "hexahedron"]

for n in 1:length(QUAD_DATA)
    points, weights = QUAD_DATA[n]
    order = 2(n-1)+1
    for dim in (1, 2, 3)
        n_points = length(points)^dim
        quadname = QuoteNode(Symbol(string(names[dim], n_points)))
        z = tensor_product(weights, points, dim)
        @eval begin
            @doc """
                get_quadrature_points(::Type{Val{:$($(quadname))})

            Gauss-Legendre quadrature, $($(n_points)) point rule on $($(names2[dim]))."""
            get_quadrature_points(::Type{Val{$(quadname)}}) = $z
        end
        @eval get_order(::Type{Val{$(quadname)}}) = $order
    end
end

