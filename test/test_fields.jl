# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md


module FieldTests

using JuliaFEM
using JuliaFEM.Test

using JuliaFEM: Field

function test_create_field()
    f = Field(1.0)
end

end
