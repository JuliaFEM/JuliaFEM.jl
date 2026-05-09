# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

"""
    piola_covariant(J::Tensor{2,3}, u_ref::Vec{3}) -> Vec{3}

Covariant Piola transform (push-forward for `H(curl)`-type fields):

``u^{\\mathrm{phys}} = J^{-T} u^{\\mathrm{ref}}``,

with `J = ∂x/∂ξ` the Jacobian of the reference-to-physical map.

See also [`piola_contravariant`](@ref) for `H(\\mathrm{div})`.
"""
@inline function piola_covariant(J::Tensor{2, 3}, u_ref::Vec{3})
    return inv(J)' ⋅ u_ref
end

"""
    piola_contravariant(J::Tensor{2,3}, u_ref::Vec{3}) -> Vec{3}

Contravariant Piola transform (push-forward for `H(\\mathrm{div})`-type fields):

``u^{\\mathrm{phys}} = (\\det J)^{-1} J \\, u^{\\mathrm{ref}}``.
"""
@inline function piola_contravariant(J::Tensor{2, 3}, u_ref::Vec{3})
    return (J ⋅ u_ref) / det(J)
end
