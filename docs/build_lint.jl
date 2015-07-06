using Lint

include("badges.jl")

d = lintpkg("JuliaFEM", returnMsgs=true)
cd(dirname(@__FILE__)) do
    k = 0
    open("quality/lint_report.rst", "w") do fid
        for i in d
            i = string(i)
            i = i[search(i, "JuliaFEM")[2]-1:end]
            write(fid, "| "*i*"\n")
            k += 1
        end
    end
    make_badge("code quality", k, "badges/lint-status.svg")
end
