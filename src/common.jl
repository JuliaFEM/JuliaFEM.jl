# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

# common routines

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

export @debug, set_debug_on!, set_debug_off!

