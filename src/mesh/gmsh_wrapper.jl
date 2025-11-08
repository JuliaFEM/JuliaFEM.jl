# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Gmsh.jl/blob/master/LICENSE
#
# Gmsh wrapper - consolidated from Gmsh.jl
# Provides convenient access to Gmsh API via gmsh_jll

import gmsh_jll
include(gmsh_jll.gmsh_api)
import .gmsh

"""
    gmsh_initialize(argv=String[]; finalize_atexit=true)

Wrapper around `gmsh.initialize` which make sure to only call it if `gmsh` is not already
initialized. Return `true` if `gmsh.initialize` was called, and `false` if `gmsh` was
already initialized.

The argument vector `argv` is passed to `gmsh.initialize`. `argv` can be used to pass
command line options to Gmsh, see [Gmsh documentation for more
details](https://gmsh.info/doc/texinfo/gmsh.html#index-Command_002dline-options). Note that
this wrapper prepends the program name to `argv` since Gmsh expects that to be the first
entry.

If `finalize_atexit` is `true` a Julia exit hook is added, which calls `finalize()`.

**Example**
```julia
Gmsh.initialize(["-v", "0"]) # initialize with decreased verbosity
```
"""
function gmsh_initialize(argv=String[]; finalize_atexit=true)
    if Bool(gmsh.isInitialized())
        return false
    end
    # Prepend a dummy program name in case argv only contains options
    # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/2112
    if length(argv) > 0 && startswith(first(argv), "-")
        argv = pushfirst!(copy(argv), "gmsh")
    end
    gmsh.initialize(argv)
    if finalize_atexit
        atexit(finalize)
    end
    return true


"""
    Gmsh.finalize()

Wrapper around `gmsh.finalize` which make sure to only call it if `gmsh` is initialized.
"""
function gmsh_finalize()
    if Bool(gmsh.isInitialized())
        gmsh.finalize()
    end



