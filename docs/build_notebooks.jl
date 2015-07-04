cd(dirname(@__FILE__)) do

results = Dict[]
for ipynb in readdir("tutorials")
    tic()
    if !endswith(ipynb, "ipynb")
        continue
    end
    runtime = 0
    status = 1
    println("Running notebook tutorials/$ipynb")
    try
        run(`runipy -o tutorials/$ipynb --kernel=julia-0.4 --port=34211`)
        status = 0
    catch
        println("did not work")
    end
    runtime = toc()
    run(`ipython nbconvert tutorials/$ipynb --to rst --output=tutorials/$(ipynb[1:end-6])`)
    println("took $runtime seconds")
    push!(results, Dict("filename" => ipynb, "status" => status, "runtime" => runtime))
end

function write_login_markdown(preamble, results, footer,filename="tutorials/README.md")
    open(filename, "w") do f
        write(f, preamble)
        for (index, each) in enumerate(results)
            write(f, """|$(index)|$(each["filename"])|$(round(each["runtime"], 2))|$(each["status"])|
""")
        end
        write(f, footer)
    end
end


preamble = """##Notebooks

| id | Notebook | Runtime | Status|
|----|----------|---------|-------|
"""
footer = """
"""

write_login_markdown(preamble, results, footer)

end
