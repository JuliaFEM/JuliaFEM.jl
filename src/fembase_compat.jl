# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE

"""
Compatibility shim for vendor packages that expect FEMBase types.

Since we've consolidated FEMBase into JuliaFEM, we need to provide
the FEMBase module namespace for backward compatibility with code
that uses FEMBase.function_name().

This creates a minimal FEMBase module with function forwarding.
Type aliases are added after all types are defined.
"""
module FEMBase

# Note: We can only create function aliases here, not type aliases,
# because not all types have been defined yet when this module is included.

# Forward declarations for functions that exist at this point
# We'll add more after types are defined

end # module FEMBase
