# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

"""
A very simple debugging macro. It executes commands if environment variable DEBUG is set.

Usage
-----
Instead of starting session `julia file.jl`, do `DEBUG=1 julia file.jl`.
Or set `export DEBUG=1` for your `.bashrc`.

Running inside code
-------------------

julia> @debug info("moimoi heihei")

will get executed iff environment variable DEBUG is set. 

Examples
--------
julia> @debug info("moimoi")
(empty)
julia> ENV["DEBUG"] = 1
julia> @debug info("moimoi")
INFO: moimoi
julia> @debug begin
...        info("matrix is")
...        dump([1 2; 3 4])
...    end
INFO: matrix is
Array(Int64(2,2)) 2x2 Array{Int64,2}:
 1  2
 3  4

"""
macro debug(msg)
    haskey(ENV, "DEBUG") || return
    return msg
end

function set_debug_on!()
    ENV["DEBUG"] = 1;
end

function set_debug_off!()
    pop!(ENV, "DEBUG");
end

#=
""" Simple linspace extension to arrays.

Examples
--------
>>> linspace([0.0], [1.0], 3)
3-element Array{Array{Float64,1},1}:
 [0.0]
 [0.5]
 [1.0]

"""
function linspace{T<:Array}(X1::T, X2::T, n)
    [1/2*(1-ti)*X1 + 1/2*(1+ti)*X2 for ti in linspace(-1, 1, n)]
end
=#

function resize!(A::SparseMatrixCSC, m::Int64, n::Int64)
    (n == A.n) && (m == A.m) && return
    @assert n >= A.n
    @assert m >= A.m
    append!(A.colptr, A.colptr[end]*ones(Int, m-A.m))
    A.n = n
    A.m = m
end

function ForwardDiff.derivative{T}(f::Function, S::Matrix{T}, args...)
    shape = size(S)
    wrapper(S::Vector) = f(reshape(S, shape))
    deriv = ForwardDiff.gradient(wrapper, vec(S), args...)
    return reshape(deriv, shape)
end

export @debug, set_debug_on!, set_debug_off!

