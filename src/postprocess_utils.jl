# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

importall Base

using JuliaFEM
using DataFrames
using HDF5
using LightXML
using Formatting

"""
Calculate field values to nodal points from Gauss points using least-squares fitting.
"""
function calc_nodal_values!(elements::Vector, field_name, field_dim, time;
                            F=nothing, nz=nothing, b=nothing, return_F_and_nz=false)

    if F == nothing
        A = SparseMatrixCOO()
        for element in elements
            gdofs = get_connectivity(element)
            for ip in get_integration_points(element)
                detJ = element(ip, time, Val{:detJ})
                w = ip.weight*detJ
                N = element(ip, time)
                add!(A, gdofs, gdofs, w*kron(N', N))
            end
        end
        A = sparse(A)
        nz = get_nonzero_rows(A)
        A = 1/2*(A + A')
        F = ldltfact(A[nz,nz])
    end

    if b == nothing
        b = SparseMatrixCOO()
        for element in elements
            gdofs = get_connectivity(element)
            for ip in get_integration_points(element)
                if !haskey(ip, field_name)
                    info("warning: integration point does not have field $field_name")
                    continue
                end
                detJ = element(ip, time, Val{:detJ})
                w = ip.weight*detJ
                f = ip(field_name, time)
                N = element(ip, time)
                for dim=1:field_dim
                    add!(b, gdofs, w*f[dim]*N, dim)
                end
            end
        end
        b = sparse(b)
    end

    x = zeros(size(b)...)
    x[nz, :] = F \ b[nz, :]
    nodal_values = Dict()
    for i=1:size(x,1)
        nodal_values[i] = vec(x[i,:])
    end
    update!(elements, field_name, time => nodal_values)
    if return_F_and_nz
        return F, nz
    end
end

"""
Return node ids + vector of values
"""
function get_nodal_vector(elements::Vector, field_name::AbstractString, time::Float64)
    f = Dict()
    for element in elements
        for (c, v) in zip(get_connectivity(element), element[field_name](time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    node_ids = sort(collect(keys(f)))
    field = [f[nid] for nid in node_ids]
    return node_ids, field
end


function to_dataframe(u::Dict, abbreviation::Symbol)
    length(u) != 0 || return DataFrame()
    node_ids = collect(keys(u))
    column_names = [:NODE]
    n = length(u[first(node_ids)])
    index = [Symbol("N$id") for id in node_ids]
    result = Any[index]
    for dof=1:n
        push!(result, [u[id][dof] for id in node_ids])
        push!(column_names, Symbol("$abbreviation$dof"))
    end
    df = DataFrame(result, column_names)
    sort!(df, cols=[:NODE])
    return df
end

function (solver::Solver)(::Type{DataFrame}, field_name::AbstractString,
              abbreviation::Symbol, time::Float64=0.0)
    fields = [problem(field_name, time) for problem in get_problems(solver)]
    fields = filter(f -> f != nothing, fields)
    if length(fields) != 0
        u = merge(fields...)
    else
        u = Dict()
    end
    return to_dataframe(u, abbreviation)
end

""" Interpolate field from a set of elements. """
function (problem::Problem)(field_name::AbstractString, X::Vector, time::Float64=0.0; fillna=NaN)
    for element in get_elements(problem)
        if inside(element, X, time)
            xi = get_local_coordinates(element, X, time)
            return element(field_name, xi, time)
        end
    end
    return fillna
end

""" Interpolate field from a set of elements. """
function (problem::Problem)(field_name::AbstractString, X::Vector, time::Float64, ::Type{Val{:Grad}}; fillna=NaN)
    for element in get_elements(problem)
        if inside(element, X, time)
            xi = get_local_coordinates(element, X, time)
            return element(field_name, xi, time, Val{:Grad})
        end
    end
    return fillna
end

function (solver::Solver)(field_name::AbstractString, X::Vector, time::Float64; fillna=NaN)
    for problem in get_problems(solver)
        for element in get_elements(problem)
            if inside(element, X, time)
                xi = get_local_coordinates(element, X, time)
                return element(field_name, xi, time)
            end
        end
    end
    return fillna
end

""" Calculate area of cross-section. """
function calculate_area(problem::Problem, X=[0.0, 0.0], time=0.0)
    A = 0.0
    for element in get_elements(problem)
        elsize = size(element)
        elsize[1] == 2 || error("wrong dimension of problem for area calculation, element size = $elsize")
        for ip in get_integration_points(element)
            w = ip.weight*element(ip, time, Val{:detJ})
            A += w
        end
    end
    return A
end

""" Calculate volume of body. """
function calculate_volume(problem::Problem, X=[0.0, 0.0, 0.0], time=0.0)
    V = 0.0
    for element in get_elements(problem)
        elsize = size(element)
        elsize[1] == 3 || error("wrong dimension of problem for area calculation, element size = $elsize")
        for ip in get_integration_points(element)
            w = ip.weight*element(ip, time, Val{:detJ})
            V += w
        end
    end
    return V
end

""" Calculate center of mass of body with respect to X.
https://en.wikipedia.org/wiki/Center_of_mass
"""
function calculate_center_of_mass(problem::Problem, X=[0.0, 0.0, 0.0], time=0.0)
    M = 0.0
    Xc = zeros(X)
    for element in get_elements(problem)
        for ip in get_integration_points(element)
            w = ip.weight*element(ip, time, Val{:detJ})
            M += w
            rho = haskey(element, "density") ? element("density", ip, time) : 1.0
            Xp = element("geometry", ip, time)
            Xc += w*rho*(Xp-X)
        end
    end
    return 1.0/M * Xc
end

""" Calculate second moment of mass with respect to X.
https://en.wikipedia.org/wiki/Second_moment_of_area
"""
function calculate_second_moment_of_mass(problem::Problem, X=[0.0, 0.0, 0.0], time=0.0)
    n = length(X)
    I = zeros(n, n)
    for element in get_elements(problem)
        for ip in get_integration_points(element)
            w = ip.weight*element(ip, time, Val{:detJ})
            rho = haskey(element, "density") ? element("density", ip, time) : 1.0
            Xp = element("geometry", ip, time) - X
            I += w*rho*Xp*Xp'
        end
    end
    return I
end
