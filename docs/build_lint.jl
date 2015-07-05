using Lint
d = lintpkg("JuliaFEM", returnMsgs=true)
cd(dirname(@__FILE__)) do
    open("quality/lint_report.rst", "w") do fid
        for i in d
            i = string(i)
            i = i[search(i, "JuliaFEM")[2]-1:end]
            write(fid, "| "*i*"\n")
        end
    end
end