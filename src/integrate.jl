# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Let's drop here all integration schemes and some defaults for different element types

function get_integration_points(Quad4::Element)
    [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)
    ]
end

typealias LineElement Union{Seg2, Seg3}

function get_integration_points(element::LineElement, ::Type{Val{1}})
    [
        IntegrationPoint([0.0], 2.0)
    ]
end

function get_integration_points(element::LineElement, ::Type{Val{2}})
    [
        IntegrationPoint([-sqrt(1/3)], 1)
        IntegrationPoint([+sqrt(1/3)], 1)
    ]
end

function get_integration_points(element::LineElement, ::Type{Val{3}})
    [
        IntegrationPoint([0.0], 8/9),
        IntegrationPoint([-sqrt(3/5)], 5/9),
        IntegrationPoint([+sqrt(3/5)], 5/9)
    ]
end

function get_integration_points(element::LineElement, ::Type{Val{4}})
    [
        IntegrationPoint([+sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
        IntegrationPoint([-sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
        IntegrationPoint([+sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
        IntegrationPoint([-sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
    ]
end

function get_integration_points(element::LineElement, ::Type{Val{5}})
    [
        IntegrationPoint([-1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900),
        IntegrationPoint([-1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([0.0], 128/225),
        IntegrationPoint([ 1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([ 1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900)
    ]
end

function get_integration_points(element::Seg2)
    return get_integration_points(element, Val{1})
end

function get_integration_points(element::Seg3)
    return get_integration_points(element, Val{2})
end

### 3D elements

function get_integration_points(element::Tet10, ::Type{Val{4}})
    a = .585410196624969
    b = .138196601125011
    w = .041666666666667
    integration_points = [
        IntegrationPoint([a, b, b], w),
        IntegrationPoint([b, a, b], w),
        IntegrationPoint([b, b, a], w),
        IntegrationPoint([b, b, b], w)]
end

function get_integration_points(element::Tet10)
    return get_integration_points(element, Val{4})
end

