# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Base.Test
using TimerOutputs

pkg_dir = Pkg.dir("JuliaFEM")
maybe_test_files = readdir(joinpath(pkg_dir, "test"))
is_test_file(fn) = startswith(fn, "test_") & endswith(fn, ".jl")
test_files = filter(is_test_file, maybe_test_files)
info("$(length(test_files)) test files found.")

const to = TimerOutput()
@testset "JuliaFEM.jl" begin
    for fn in test_files
        info("----- Running tests from file $fn -----")
        t0 = time()
        timeit(to, fn) do
            include(fn)
        end
        dt = round(time() - t0, 2)
        info("----- Testing file $fn completed in $dt seconds -----")
    end
end
println()
println("Test statistics:")
println(to)
