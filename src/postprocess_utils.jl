# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

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
        F = ldlt(A[nz,nz])
    end

    if b == nothing
        b = SparseMatrixCOO()
        for element in elements
            gdofs = get_connectivity(element)
            for ip in get_integration_points(element)
                if !haskey(ip, field_name)
                    @warn("integration point does not have field $field_name")
                    continue
                end
                detJ = element(ip, time, Val{:detJ})
                w = ip.weight*detJ
                f = ip(field_name, time)
                N = element(ip, time)
                for dim=1:field_dim
                    add!(b, gdofs, [dim], w*f[dim]*N')
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
        for (c, v) in zip(get_connectivity(element), element(field_name, time))
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

"""
    problem(field_name, X, time)

Interpolate field from a set of elements defined in problem. Here, `X` is the
location inside domain described by elements.

Internally, function loops through all the elements, finding the one containing
the point `X`. After that, using inverse isoparametric mapping, first find
dimensionless coordinates (ξ,η,ζ) of that element corresponding to the location
of point `X` and after that interpolate the values of field under investigation.
Algorithm can be expected to be somewhat slow for big models, but for tests
models the performance is good.

# Examples

Having a problem called `body`, one can query the field `displacement` at
position `X = (1.0, 2.0, 3.0)` and time `t = 1.0`, with the command
```julia
X = (1.0, 2.0, 3.0)
time = 1.0
u = body("displacement", X, time)
```
"""
function (problem::Problem)(field_name, X, time; fillna=NaN)
    for element in get_elements(problem)
        if inside(element, X, time)
            xi = get_local_coordinates(element, X, time)
            return element(field_name, xi, time)
        end
    end
    return fillna
end

function (problem::Problem)(field_name, X, time, ::Type{Val{:Grad}}; fillna=NaN)
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

""" Calculate center of mass of body with respect to X.
https://en.wikipedia.org/wiki/Center_of_mass
"""
function calculate_center_of_mass(problem::Problem, X=[0.0, 0.0, 0.0], time=0.0)
    M = 0.0
    Xc = zero(X)
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
