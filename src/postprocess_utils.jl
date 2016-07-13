# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

importall Base

using JuliaFEM
using DataFrames
using HDF5
using LightXML

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
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
            if haskey(f, c)
                @assert isapprox(f[c], v)
            end
            f[c] = v
        end
    end
    return f
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

