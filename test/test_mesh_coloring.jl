using JuliaFEM, LinearAlgebra, Test

datadir = first(splitext(basename(@__FILE__)))

@testset "Test that mesh coloring provides a good coloring"
    fn = joinpath(datadir, "cube_tet4.inp")
    mesh = JuliaFEM.Mesh(open(parse_abaqus, fn))

    JuliaFEM.create_coloring!(mesh)
    for colors in mesh.coloring
        for ele_i in colors
            for ele_j in colors
                if ele_i == ele_j
                    continue
                end
                @test intersect(Set(mesh.elements[ele_i]), Set(mesh.elements[ele_j])) |> isempty
            end
        end
    end
end