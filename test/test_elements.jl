# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using FactCheck

facts("test set and interpolate field variable") do
    el = Quad4(1, [1, 2, 3, 4])
    set_coordinates(el, [0.0 0.0; 10.0 0.0; 10.0 1.0; 0.0 1.0]')
    set_field(el, "displacement", [0.0 0.0; 0.0 0.0; 0.5 0.0; 0.0 0.0]'')
    fval = interpolate(el, "displacement", [0.0, 1.0])
    Logging.debug(fval)
    @fact fval --> roughly([0.25 0.0]')
end

