"""
calculate "local" normals in elements, in a way that
n = Nᵢnᵢ gives some reasonable results for ξ ∈ [-1, 1]
"""
function calculate_normals!(el::Element, t, field_name=symbol("normals"))
    new_field!(el, field_name, Vector)
    for xi in Vector[[-1.0], [1.0]]
        t = dinterpolate(el, :Geometry, xi)
        n = [0 -1; 1 0]*t
        n /= norm(n)
        push_field!(el, field_name, n)
    end
end

"""
Alter normal field such that normals of adjacent elements are averaged.
"""
function average_normals!(elements, normal_field=symbol("normals"))
    d = Dict()
    for el in elements
        c = get_connectivity(el)
        n = get_field(el, normal_field)
        for (ci, ni) in zip(c, n)
            d[ci] = haskey(d, ci) ? d[ci] + ni : ni
        end
    end
    for (ci, ni) in d
        d[ci] /= norm(d[ci])
    end
    for el in elements
        c = get_connectivity(el)
        new_normals = [d[ci] for ci in c]
        set_field(el, normal_field, new_normals)
    end
end


""" Find projection from slave nodes to master element. """
function calc_projection_slave_nodes_to_master_element(sel, mel)
    X1 = get_field(sel, :Geometry)
    N1 = get_field(sel, :Normals)
    X2(xi) = interpolate(mel, :Geometry, xi)
    dX2(xi) = dinterpolate(mel, :Geometry, xi)
    R(xi, k) = det([X2(xi) - X1[k] N1[k]]')
    dR(xi, k) = det([dX2(xi) N1[k]]')
    xi2 = Vector[[0.0], [0.0]]
    for k=1:2
        xi = xi2[k]
        for i=1:3
            dxi = -R(xi, k)/dR(xi, k)
            xi += dxi
            if abs(dxi) < 1.0e-9
                break
            end
        end
        xi2[k] = xi
    end
    clamp!(xi2, -1, 1)
    return xi2
end

""" Find projection from master nodes to slave element. """
function calc_projection_master_nodes_to_slave_element(sel, mel)
    X1(xi) = interpolate(sel, :Geometry, xi)
    dX1(xi) = dinterpolate(sel, :Geometry, xi)
    N1(xi) = interpolate(sel, :Normals, xi)
    dN1(xi) = dinterpolate(sel, :Normals, xi)
    X2 = get_field(mel, :Geometry)
    R(xi, k) = det([X1(xi) - X2[k] N1(xi)]')
    dR(xi, k) = det([dX1(xi) N1(xi)]') + det([X1(xi) - X2[k] dN1(xi)]')
    xi1 = Vector[[0.0], [0.0]]
    for k=1:2
        xi = xi1[k]
        for i=1:3
            dxi = -R(xi, k)/dR(xi, k)
            xi += dxi
            if abs(dxi) < 1.0e-9
                break
            end
        end
        xi1[k] = xi
    end
    clamp!(xi1, -1, 1)
    return xi1
end

function has_projection(sel, mel)
    xi1 = calc_projection_master_nodes_to_slave_element(sel, mel)
    l = abs(xi1[2]-xi1[1])[1]
    return l > 1.0e-9
end

"""
Calculate projection between 1d boundary elements
"""
function calc_projection(sel, mel)
    xi1 = calc_projection_master_nodes_to_slave_element(sel, mel)
    xi2 = calc_projection_slave_nodes_to_master_element(sel, mel)
    return xi1, xi2
end

