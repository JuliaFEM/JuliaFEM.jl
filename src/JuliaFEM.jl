# This file is a part of JuliaFEM. License is MIT: https://github.com/ovainola/JuliaFEM/blob/master/README.md
module JuliaFEM

# package code goes here
type Node
    node_id
    coords
end
type Element
    element_id
    node_ids
end
function test()
    return 1
end

end # module
