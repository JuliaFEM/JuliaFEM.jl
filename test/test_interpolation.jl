# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM: get_basis, grad, FieldSet, Field, Quad4
using FactCheck

element = Quad4([1, 2, 3, 4])

geometry_field = Field(0.0, Vector[])  # Create empty field at time t=0.0
push!(geometry_field, [ 0.0, 0.0])  # push some values for field
push!(geometry_field, [ 1.0, 0.0])
push!(geometry_field, [ 1.0, 1.0])
push!(geometry_field, [ 0.0, 1.0])
geometry_fieldset = FieldSet("geometry")  # create fieldset "geometry"
push!(geometry_fieldset, geometry_field)  # add field to fieldset
push!(element, geometry_fieldset) # add fieldset to element

temperature_fieldset = FieldSet("temperature")
push!(temperature_fieldset, Field(0.0, [0.0, 0.0, 0.0, 0.0]))
push!(temperature_fieldset, Field(1.0, [1.0, 2.0, 3.0, 4.0]))
push!(element, temperature_fieldset)

displacement_fieldset = FieldSet("displacement")
push!(displacement_fieldset, Field(0.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
push!(displacement_fieldset, Field(1.0, Vector[[0.0, 0.0], [0.0, 0.0], [0.25, 0.0], [0.0, 0.0]]))
push!(element, displacement_fieldset)

facts("basic continuum interpolations") do
    # from my old home works
    basis = get_basis(element)
    dbasis = grad(basis)
    @fact basis("geometry", [0.0, 0.0], 1.0) + basis("displacement", [0.0, 0.0], 1.0) --> [9/16, 1/2]
    gradu = dbasis("displacement", [0.0, 0.0], 1.0)
    epsilon = 1/2*(gradu + gradu')
    rotation = 1/2*(gradu - gradu')
    X = basis("geometry", [0.0, 0.0], 1.0)
    k = 0.25
    epsilon_wanted = [X[2]*k 1/2*X[1]*k; 1/2*X[1]*k 0]
    rotation_wanted = [0 k/2*X[1]; -k/2*X[1] 0]
    @fact epsilon --> roughly(epsilon_wanted)
    @fact rotation --> roughly(rotation_wanted)
    F = I + gradu
    @fact F --> [X[2]*k+1 X[1]*k; 0 1]
    C = F'*F
    @fact C --> [(X[2]*k+1)^2  (X[2]*k+1)*X[1]*k; (X[2]*k+1)*X[1]*k  X[1]^2*k^2+1]
    E = 1/2*(F'*F - I)
    @fact E --> [1/2*(X[2]*k + 1)^2-1/2  1/2*(X[2]*k+1)*X[1]*k; 1/2*(X[2]*k + 1)*X[1]*k  1/2*X[1]^2*k^2]
    U = 1/sqrt(trace(C) + 2*sqrt(det(C)))*(C + sqrt(det(C))*I)
    #@fact U --> roughly([1.24235 0.13804; 0.13804 1.02149])
end
