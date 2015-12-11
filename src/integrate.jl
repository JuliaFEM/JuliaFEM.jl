# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Let's drop here all integration schemes and some defaults for different element types
# maybe parse from txt file ..?

### 1d elements

function get_integration_points(::Type{Seg2}, ::Type{Val{1}})
    [
        IntegrationPoint([0.0], 2.0)
    ]
end

function get_integration_points(::Type{Seg2}, ::Type{Val{2}})
    [
        IntegrationPoint([-sqrt(1/3)], 1)
        IntegrationPoint([+sqrt(1/3)], 1)
    ]
end

function get_integration_points(::Type{Seg3}, ::Type{Val{3}})
    [
        IntegrationPoint([0.0], 8/9),
        IntegrationPoint([-sqrt(3/5)], 5/9),
        IntegrationPoint([+sqrt(3/5)], 5/9)
    ]
end

function get_integration_points(::Type{Seg3}, ::Type{Val{4}})
    [
        IntegrationPoint([+sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
        IntegrationPoint([-sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
        IntegrationPoint([+sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
        IntegrationPoint([-sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
    ]
end

function get_integration_points(::Union{Type{Seg2}, Type{Seg3}}, ::Type{Val{5}})
    [
        IntegrationPoint([-1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900),
        IntegrationPoint([-1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([0.0], 128/225),
        IntegrationPoint([ 1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([ 1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900)
    ]
end

function get_integration_points(::Type{Seg2})
    return get_integration_points(Seg2, Val{2})
end

function get_integration_points(::Type{Seg3})
    return get_integration_points(Seg3, Val{3})
end

### 2d triangular elements

# http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF

typealias TriangularElements Union{Type{Tri3}, Type{Tri6}}

function get_integration_points(::TriangularElements, ::Type{Val{1}})
    # http://libmesh.github.io/doxygen/quadrature__gauss__2D_8C_source.html
    [
        IntegrationPoint([1.0/3.0, 1.0/3.0], 0.5)
    ]
end

function get_integration_points(::TriangularElements, ::Type{Val{2}})
    # http://libmesh.github.io/doxygen/quadrature__gauss__2D_8C_source.html
    [
        IntegrationPoint([2.0/3.0, 1.0/6.0], 1.0/6.0),
        IntegrationPoint([1.0/6.0, 2.0/3.0], 1.0/6.0),
        IntegrationPoint([1.0/6.0, 1.0/6.0], 1.0/6.0)
    ]
end

function get_integration_points(::TriangularElements, ::Type{Val{5}})
    # http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
    # FIXME: something wrong here with weights ..?
    [
        IntegrationPoint([0.33333333333333, 0.33333333333333], 0.5*0.22500000000000),
        IntegrationPoint([0.47014206410511, 0.47014206410511], 0.5*0.13239415278851),
        IntegrationPoint([0.47014206410511, 0.05971587178977], 0.5*0.13239415278851),
        IntegrationPoint([0.05971587178977, 0.47014206410511], 0.5*0.13239415278851),
        IntegrationPoint([0.10128650732346, 0.10128650732346], 0.5*0.12593918054483),
        IntegrationPoint([0.10128650732346, 0.79742698535309], 0.5*0.12593918054483),
        IntegrationPoint([0.79742698535309, 0.10128650732346], 0.5*0.12593918054483)
    ]
end

function get_integration_points(::Type{Tri3})
    return get_integration_points(Tri3, Val{1})
end


function get_integration_points(::Type{Quad4})
    [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)
    ]
end

### 3d elements

function get_integration_points(::Type{Tet4})
    # http://libmesh.github.io/doxygen/quadrature__gauss__3D_8C_source.html
    [
        IntegrationPoint([0.25, 0.25, 0.25], 1.0/6.0)
    ]
end

function get_integration_points(::Type{Tet10})
    # http://libmesh.github.io/doxygen/quadrature__gauss__3D_8C_source.html
    a = .585410196624969
    b = .138196601125011
    w = .041666666666667
    [
        IntegrationPoint([a, b, b], w),
        IntegrationPoint([b, a, b], w),
        IntegrationPoint([b, b, a], w),
        IntegrationPoint([b, b, b], w)
    ]
end
