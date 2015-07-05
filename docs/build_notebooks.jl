# Automatically run notebooks and generate rst table for results

function run_notebooks()
    k = 0
    results = Dict[]
    cd(dirname(@__FILE__)) do
        for ipynb in readdir("tutorials")
            tic()
            if !endswith(ipynb, "ipynb")
                continue
            end
            runtime = 0
            status = 1
            println("Running notebook tutorials/$ipynb")
            port = 34211+k  # we're having some weird port issue with zmq
            k += 1
            try
                run(`runipy -o tutorials/$ipynb --kernel=julia-0.4 --port=$port`)
                status = 0
            catch
                println("did not work")
            end
            runtime = toc()
            run(`ipython nbconvert tutorials/$ipynb --to rst --output=tutorials/$(ipynb[1:end-6])`)
            #println("took $runtime seconds")
            data = Dict("author" => "unknown", "status" => status, "runtime" => runtime,
                        "filename" => ipynb, "last_run" => time(), "description"=>"")
            push!(results, data)
        end
    end
    return results
end

"""
Make rows from results ready to tabular form
"""
function makerows(results)
    rows = ["id" "author" "description" "last run" "runtime (s)" "status" "html" "pdf"]
    for (i, data) in enumerate(results)
        row = ["na" "unknown" "unknown" "unknown" "unknown" "unknown" "na" "na"]
        println(i)
        row[1] = string(i)
        desc = data["description"]
        if desc == ""
            desc = data["filename"]
        end
        row[3] = desc
        row[4] = Libc.strftime("%Y-%m-%d %H:%M:%S", data["last_run"])
        row[5] = string(round(data["runtime"], 2))
        if data["status"] == 0
            row[6] = "PASS"
        else
            row[6] = "FAIL"
        end
        fn = data["filename"][1:end-6]*".rst"
        row[7] = ":doc:`html <$fn>`"
        fn = data["filename"][1:end-6]*".pdf"
        row[8] = ":download:`html <$fn>`"
        rows = vcat(rows, row)
    end
    return rows
end


"""
write rst table. rows is array of arrays.

Expected output

    +----+-----------+------------------------------+------------+---------+--------+------+-----+
    | id | author    | description                  | last run   | runtime  | status | html | pdf |
    +====+===========+==============================+============+=========+========+======+=====+
    |  1 | Jukka Aho | This is placeholder notebook | 2015-07-05 | 123.45  | |pass| | html | pdf |
    +----+-----------+------------------------------+------------+---------+--------+------+-----+


"""
function write_rst_table(rows)
    # determine cell lenghts
    lenghts = zeros(Int, size(rows))
    for i=1:size(rows, 1)
        for j=1:size(rows, 2)
            lenghts[i,j] = length(rows[i,j])+3
        end
    end
    #println(lenghts)
    cl = maximum(lenghts, 1)  # maximum column lenghts
    tl = sum(cl)+1  # total table width
    mp = [1 cumsum(cl, 2)+1] # marker points "+"
    println(mp)
    #println(cl)
    #println(tl)

    function sep(l, m; s="+", d="-")
        p = ""
        for j=1:l
            if j in m
                p *= s
            else
                p *= d
            end
        end
        p *= "\n"
        return p
    end

    s = ""
    # write rows
    s *= sep(tl, mp)
    for i = 1:size(rows, 1)
        s *= "|"
        for j=1:size(rows, 2)
            s *= " " * rows[i, j] * " "
            s *= repeat(" ", cl[j]-lenghts[i,j])
            s *= "|"
        end
        s *= "\n"
        if i == 1
            s *= sep(tl, mp; d="=")
        else
            s *= sep(tl, mp)
        end
    end
    return s
end


function main(;filename="tutorials/notebooks.rst")
    results = run_notebooks()
    rows = makerows(results)
    table = write_rst_table(rows)
    println(table)
    cd(dirname(@__FILE__)) do
        open(filename, "w") do fid
            write(fid, table)
        end
    end
end


main()
