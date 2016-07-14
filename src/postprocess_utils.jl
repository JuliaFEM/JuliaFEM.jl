# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

importall Base

using JuliaFEM
using DataFrames
using HDF5
using LightXML
using StringUtils

import HDF5: h5read, h5write

function h5read{T<:DataFrame}(::Type{T}, filename, name::ByteString)
    raw_data = h5read(filename, name)
    index = raw_data["index"]
    column_names = raw_data["column_names"]
    column_names = map(parse, column_names)
    n, m = size(raw_data["data"])
    data = Any[index]
    for i=1:m
        push!(data, raw_data["data"][:,i])
    end
    return DataFrame(data, column_names)
end

function h5write(filename, name::ByteString, data::DataFrame)
    column_names = DataFrames._names(data)
    column_names = map(string, column_names)
    index = convert(Vector, data[:,1])
    data = convert(Matrix, data[:,2:end])
    h5write(filename, "$name/column_names", column_names)
    h5write(filename, "$name/index", index)
    h5write(filename, "$name/data", data)
end

function convert(::Type{AbstractString}, df::DataFrame)
    fn = tempname()
    writetable(fn, df)
    return readall(fn)
end

function convert(::Type{DataFrame}, dfs::AbstractString)
    fn = tempname()
    fid = open(fn, "w")
    write(fid, dfs)
    close(fid)
    return readtable(fn)
end

function extract(df::DataFrame, args...; kwargs...)
    result = copy(df)
    for (k,v) in kwargs
        rows = find(df[k] .== v)
        result = result[rows, :]
    end
    foo = Symbol[si for si in args]
    return result[foo]
end

function vec(df::DataFrame)
    return vec(convert(Matrix{Float64}, df))
end

function isapprox(d1::DataFrame, d2::Vector)
    return isapprox(vec(d1), d2)
end

""" A more appropriate representation for floats in results. """
function DataFrames.ourshowcompact(io::IO, x::Float64)
    print(io, u"\% 0.4E(x)")
    return
end

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
        nz = get_nonzero_rows(A)
        A = sparse(A)
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

function calc_nodal_values!(problem::Problem, field_name, field_dim, time)
    # after all, it's just a mass matrix ...
#   isempty(problem.assembly.M) && assemble!(problem, time, Val{:mass_matrix}; density=1.0, dual_basis=false, dim=1)
#   M = sparse(problem.assembly.M)
    # TODO: make test before implementation
    calc_nodal_values!(problem.elements, field_name, field_dim, time)
end

"""
Return node ids + vector of values 
"""
function get_nodal_vector(elements, field_name, time)
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

""" Return nodal values in Dict format. """
function get_nodal_dict(T::DataType, elements, field_name, time)
    f = T()
    for element in elements
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    return f
end

""" Update nodal field values from set of elements to another. Can be used to
transform e.g. reaction force from boundary element set to surface of
volume elements for easier postprocess.
"""
function copy_field!(src_elements::Vector, dst_elements::Vector, field_name, time)
    dst_nodes = Set{Int64}()
    for element in dst_elements
        push!(dst_nodes, get_connectivity(element)...)
    end
    node_ids, field = get_nodal_vector(src_elements, field_name, time)
    z = 0.0*first(field)
    d = Dict()
    for j in dst_nodes
        d[j] = z
    end
    for (j, f) in zip(node_ids, field)
        d[j] = f
    end
    for element in dst_elements
        c = get_connectivity(element)
        f = [d[j] for j in c]
        update!(element, field_name, time => f)
    end
end

function copy_field!(src_problem::Problem, dst_problem::Problem, field_name, time)
    copy_field!(src_problem.elements, dst_problem.elements, field_name, time)
end

""" Return field calculated to nodal points for elements in problem p. """
function call(problem::Problem, field_name::AbstractString, time::Float64=0.0)
    f = Dict()
    for element in get_elements(problem)
        haskey(element, field_name) || continue
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                if !isapprox(f[c], v)
                    info("several values for single node when returning field $field_name")
                    info("already have: $(f[c]), and trying to set $v")
                end
            end
            f[c] = v
        end
    end
    return f
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

function call(problem::Problem, ::Type{DataFrame}, field_name::AbstractString,
              abbreviation::Symbol, time::Float64=0.0)
    u = problem(field_name, time)
    return to_dataframe(u, abbreviation)
end

function call(solver::Solver, ::Type{DataFrame}, field_name::AbstractString,
              abbreviation::Symbol, time::Float64=0.0)
    u = Dict()
    for problem in get_problems(solver)
        u = merge(u, problem(field_name, time))
    end
    return to_dataframe(u, abbreviation)
end

function get_components(n, m)
    if n == m
        if n == 1
            return Vector{Int}[[1,1]]
        end
        if n == 2
            return Vector{Int}[[1,1], [2,2], [1,2]]
        elseif n == 3
            return Vector{Int}[[1,1], [2,2], [3,3], [1,2], [1,3], [2,3]]
        else
            error("get_components, n=$n, m=$m!")
        end
    end
end

""" Return T in integration points. """
function call{T}(problem::Problem, ::Type{DataFrame}, element::Element, time::Float64,
                 ::Type{Val{T}})
    column_names = [:ELEMENT, :IP]
    ips = get_integration_points(element)
    field = Any[problem(element, ip, time, Val{T}) for ip in ips]
    # FIXME, handle better ..?
    first(field) == nothing && return DataFrame()
    m = length(field)
    n = length(first(field))
    result = Any[]
    push!(result, [Symbol("E$(element.id)") for i=1:m])
    push!(result, [Symbol("P$i") for i=1:m])
    is_tensor_field = isa(first(field), Matrix)
    if is_tensor_field
        n, m = size(first(field))
        components = get_components(n, m)
        for (j, k) in components
            push!(column_names, Symbol("$T$j$k"))
            push!(result, [S[j,k] for S in field])
        end
    else
        components = collect(1:n)
        for j in components
            push!(column_names, Symbol("$T$j"))
            push!(result, [S[j] for S in field])
        end
    end
    df = DataFrame(result, column_names)
    sort!(df, cols=[:IP])
end

function call{T}(problem::Problem, ::Type{DataFrame}, time::Float64, ::Type{Val{T}})
    tables = [problem(DataFrame, element, time, Val{T}) for element in get_elements(problem)]
    results = [tables...;]
    return results
end

function call{T}(solver::Solver, ::Type{DataFrame}, time::Float64, ::Type{Val{T}})
    problems = get_problems(solver)
    tables = Any[]
    for problem in get_problems(solver)
        try
            push!(tables, problem(DataFrame, time, Val{T}))
        catch
            warn("Unable to obtain results $T for problem $(problem.name)")
        end
    end
    results = [tables...;]
    return results
end

""" Interpolate field from a set of elements. """
function call(problem::Problem, field_name::AbstractString, X::Vector, time::Float64=0.0; fillna=NaN)
    for element in get_elements(problem)
        if inside(element, X, time)
            xi = get_local_coordinates(element, X, time)
            return element(field_name, xi, time)
        end
    end
    return fillna
end

""" Interpolate field from a set of elements. """
function call(problem::Problem, field_name::AbstractString, X::Vector, time::Float64, ::Type{Val{:Grad}}; fillna=NaN)
    for element in get_elements(problem)
        if inside(element, X, time)
            xi = get_local_coordinates(element, X, time)
            return element(field_name, xi, time, Val{:Grad})
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

function getindex(problem::Problem, field_name::AbstractString)
    info("fetching result $field_name")
    timeframes = []
    for frame in first(problem.elements)[field_name].data
        push!(timeframes, frame.time)
    end
    info("time frames: $timeframes")
    conn = get_connectivity(problem)
    increments = Increment[]
    for time in timeframes
        p = problem(field_name, time)
        data = [p[id] for id in conn]
        push!(increments, Increment(time, data))
    end
    return DVTV(increments)
end

