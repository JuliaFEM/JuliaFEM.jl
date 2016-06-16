# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function gen_rtic_grid!{S<:Union{Seg2,Seg3}}(thetas, element::Element{S}; npts=2)
    xi1 = midpoints(linspace(-1, 1, npts+1))
    for xi in xi1
        push!(thetas, [0.0, xi])
    end
end

function gen_rtic_grid!(thetas, element::Element{NSeg}; npts=2)
    knots_u = element.properties.knots
    u = midpoints(linspace(minimum(knots_u), maximum(knots_u), npts+1))
    for ui in u
        push!(thetas, [0.0, ui])
    end
end

function gen_rtic_grid!(thetas, element::Element{NSurf}; npts=2)
    knots_u = element.properties.knots_u
    knots_v = element.properties.knots_v
    u = midpoints(linspace(minimum(knots_u), maximum(knots_u), npts+1))
    v = midpoints(linspace(minimum(knots_v), maximum(knots_v), npts+1))
    for ui in u
        for vi in v
            push!(thetas, [0.0, ui, vi])
        end
    end
end

"""
Find intersection between element surface X(ξ) and ray x(t) = s + t*n by
solving equation F(t, ξ) = x(t) - X(ξ) = 0

Returns
-------
t, ξ

References
----------
[1] https://en.wikipedia.org/wiki/Ray_tracing_%28graphics%29

"""
function find_intersection{S<:Union{Seg2, Seg3, NSeg, NSurf}}(element::Element{S}, s, n, time;
    max_iterations=10, tolerance=1.0e-12, info_output=false, deformed=true, secondary=false, npts=2)

    x = element["geometry"](time)
    if deformed && haskey(element, "displacement")
        x += element["displacement"](time)
    end

    function calc_intersection!(theta)
        i = 0
        dtheta = zeros(theta)
        for i=1:max_iterations
            t = theta[1]
            xi = theta[2:end]
            N = get_basis(element, xi, time)
            dN = get_dbasis(element, xi, time)
            A = [n -dN*x]
            r = s + t*n - N*x
            try
                dtheta = A \ -r
            catch err
                info("failed to solve equation.")
                dump(theta)
                dump(A)
                dump(r)
                throw(err)
            end
            theta[:] += dtheta
            info_output && info("iter $i, theta = $theta, norm(dtheta) = $(norm(dtheta))")
            norm(dtheta) < tolerance && return theta
            isnan(theta[1]) && break
        end
        theta[1] = NaN
        return theta
        # error("didn't converge in $i iterations")
    end

    thetas = Vector{Float64}[]
    gen_rtic_grid!(thetas, element; npts=npts)
    map(calc_intersection!, thetas)
    filter!(t -> !isnan(t[1]), thetas)
    info_output && info("thetas: $thetas")
    secondary && filter!(t -> t[1] > 1.0e-12, thetas)
    # error("failure finding ray trace, thetas vec is empty")
    length(thetas) == 0 && return NaN, [NaN, NaN]
    sort!(thetas, alg=MergeSort, lt=(a,b)->a[1]<b[1])
    theta = thetas[1]
    return theta[1], theta[2:end]
end

function calc_normal{S<:Union{Seg2, Seg3, NSeg}}(element::Element{S}, xi, time; deformed=true)
    x = element["geometry"](time)
    if deformed && haskey(element, "displacement")
        x += element["displacement"](time)
    end
    Q = [0.0 1.0; -1.0 0.0]
    dN = get_dbasis(element, xi, time)
    n = Q*(dN*x)
    n /= norm(n)
    return n
end

function calc_normal{S<:Union{Quad4, NSurf}}(element::Element{S}, xi, time; deformed=true)
    x = element["geometry"](time)
    if deformed && haskey(element, "displacement")
        x += element["displacement"](time)
    end
    dN = get_dbasis(element, xi, time)
    J = transpose(sum([kron(dN[:,i], x[i]') for i=1:length(x)]))
    n = cross(J[:,1], J[:,2])
    n /= norm(n)
    return n
end

"""
References
----------
[1] http://fp.optics.arizona.edu/optomech/Fall13/Notes/6%20Mirror%20matrices.pdf
"""
function calc_reflection(element::Element, xi, k, time; deformed=true)
    n = calc_normal(element, xi, time; deformed=deformed)
    k2 = k - 2*vecdot(k, n)*n
    return k2
end

