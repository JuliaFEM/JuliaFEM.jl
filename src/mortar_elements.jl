# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# Mortar elements

# slave element = non-mortar element where integration happens
# master element = mortar element projected to non-mortar side

abstract MortarElement <: Element

type MSeg2 <: MortarElement
    connectivity :: Vector{Int}
    basis :: Basis
    fields :: FieldSet
    master_elements :: Vector{MortarElement}
end

function MSeg2(connectivity, master_elements=[], biorthogonal=false)
    basis(xi) = [(1-xi[1])/2 (1+xi[1])/2]
    dbasisdxi(xi) = [-1/2 1/2]
    return MSeg2(connectivity, Basis(basis, dbasisdxi), FieldSet(), master_elements)
end


