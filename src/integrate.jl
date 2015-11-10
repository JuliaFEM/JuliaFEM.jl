# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


function get_default_integration_points(element::Quad4)
    [
        IntegrationPoint(1.0/sqrt(3.0)*[-1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1, -1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[ 1,  1], 1.0),
        IntegrationPoint(1.0/sqrt(3.0)*[-1,  1], 1.0)
    ]
end

function get_default_integration_points(element::Seg2)
    [
        IntegrationPoint([0.0], 2.0)
    ]
end

function line3()
    [
        IntegrationPoint([0.0], 8/9),
        IntegrationPoint([-sqrt(3/5)], 5/9),
        IntegrationPoint([+sqrt(3/5)], 5/9)
    ]
end

function line5()
    [
        IntegrationPoint([-1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900),
        IntegrationPoint([-1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([0.0], 128/225),
        IntegrationPoint([ 1/3*sqrt(5 - 2*sqrt(10/7))], (322+13*sqrt(70))/900),
        IntegrationPoint([ 1/3*sqrt(5 + 2*sqrt(10/7))], (322-13*sqrt(70))/900)
    ]
end

    #integration_points = [
    #    IntegrationPoint([ 0.0000000000000000], 0.5688888888888889),
    #    IntegrationPoint([-0.5384693101056831], 0.4786286704993665),
    #    IntegrationPoint([ 0.5384693101056831], 0.4786286704993665),
    #    IntegrationPoint([-0.9061798459386640], 0.2369268850561891),
    #    IntegrationPoint([ 0.9061798459386640], 0.2369268850561891)
    #]
    #integration_points = [
    #    IntegrationPoint([+sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
    #    IntegrationPoint([-sqrt(3/7 - 2/7*sqrt(6/5))], (18+sqrt(30))/36)
    #    IntegrationPoint([+sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
    #    IntegrationPoint([-sqrt(3/7 + 2/7*sqrt(6/5))], (18-sqrt(30))/36)
    #]
    #integration_points = [
    #    IntegrationPoint([0.0], 8/9),
    #    IntegrationPoint([-sqrt(3/5)], 5/9),
    #    IntegrationPoint([+sqrt(3/5)], 5/9)
    #]
    #integration_points = [
    #    IntegrationPoint([-sqrt(1/3)], 1)
    #    IntegrationPoint([+sqrt(1/3)], 1)
    #]
    #integration_points = [
    #    IntegrationPoint([0.0], 2)
    #]

